from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
from flax import nnx


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 32000
    d_model: int = 1536
    n_head: int = 24
    n_layers: int = 24
    eps: float = 1e-5
    use_gradient_checkpointing: bool = True

    @property
    def d_head(self):
        return self.d_model // self.n_head

    @classmethod
    def dev(cls):
        return cls(block_size=64, d_model=256, n_head=4, n_layers=4, use_gradient_checkpointing=False)


class RMSNorm(nnx.Module):
    def __init__(self, dim: int, eps: float = 1e-6, *, rngs: nnx.Rngs):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((dim,)))

    def __call__(self, x):
        rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True))
        return self.weight[...] * (x / (rms + self.eps))


class RotaryEmbedding(nnx.Module):
    """Standard RoPE. Precomputes cos/sin up to block_size for efficiency under jit."""

    def __init__(self, dim: int, max_seq_len: int, base: int = 10000):
        assert dim % 2 == 0
        self.dim = dim
        inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
        t = jnp.arange(max_seq_len, dtype=jnp.float32)
        freqs = jnp.einsum("i,j->ij", t, inv_freq)  # (max_seq_len, dim/2)
        self.cos = nnx.Variable(jnp.cos(freqs))
        self.sin = nnx.Variable(jnp.sin(freqs))

    def apply_rotary(self, x, cos, sin):
        # x: (B, T, n_head, d_head). cos/sin: (T, d_head/2)
        x_part = x[..., 0::2]
        y_part = x[..., 1::2]

        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]

        x_rot = x_part * cos - y_part * sin
        y_rot = x_part * sin + y_part * cos

        out = jnp.stack((x_rot, y_rot), axis=-1)
        return out.reshape(*x.shape)

    def __call__(self, q, k):
        seq_len = q.shape[1]
        cos = self.cos[...][:seq_len].astype(q.dtype)
        sin = self.sin[...][:seq_len].astype(q.dtype)
        return self.apply_rotary(q, cos, sin), self.apply_rotary(k, cos, sin)


class MultiHeadAttention(nnx.Module):
    def __init__(self, config: GPTConfig, *, rngs: nnx.Rngs):
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.d_head = config.d_head

        self.q = nnx.Linear(self.d_model, self.d_model, use_bias=False, rngs=rngs)
        self.k = nnx.Linear(self.d_model, self.d_model, use_bias=False, rngs=rngs)
        self.v = nnx.Linear(self.d_model, self.d_model, use_bias=False, rngs=rngs)
        self.Wo = nnx.Linear(self.d_model, self.d_model, use_bias=False, rngs=rngs)

        self.rope = RotaryEmbedding(config.d_head, max_seq_len=config.block_size)

    def __call__(self, x):
        B, T, D = x.shape

        # jax.nn.dot_product_attention expects (B, T, n_head, d_head) — no transpose needed.
        q = self.q(x).reshape(B, T, self.n_head, self.d_head)
        k = self.k(x).reshape(B, T, self.n_head, self.d_head)
        v = self.v(x).reshape(B, T, self.n_head, self.d_head)

        q, k = self.rope(q, k)

        out = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        out = out.reshape(B, T, D)
        return self.Wo(out)


class FeedForward(nnx.Module):
    def __init__(self, config: GPTConfig, *, rngs: nnx.Rngs):
        hidden = 4 * config.d_model
        self.w1 = nnx.Linear(config.d_model, hidden, use_bias=False, rngs=rngs)
        self.w3 = nnx.Linear(config.d_model, hidden, use_bias=False, rngs=rngs)
        self.w2 = nnx.Linear(hidden, config.d_model, use_bias=False, rngs=rngs)

    def __call__(self, x):
        return self.w2(jax.nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nnx.Module):
    def __init__(self, config: GPTConfig, *, rngs: nnx.Rngs):
        self.norm1 = RMSNorm(config.d_model, config.eps, rngs=rngs)
        self.attn = MultiHeadAttention(config, rngs=rngs)
        self.norm2 = RMSNorm(config.d_model, config.eps, rngs=rngs)
        self.mlp = FeedForward(config, rngs=rngs)
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

    def _attn_block(self, x):
        return self.attn(self.norm1(x))

    def _mlp_block(self, x):
        return self.mlp(self.norm2(x))

    def __call__(self, x):
        if self.use_gradient_checkpointing:
            x = x + nnx.remat(type(self)._attn_block)(self, x)
            x = x + nnx.remat(type(self)._mlp_block)(self, x)
        else:
            x = x + self._attn_block(x)
            x = x + self._mlp_block(x)
        return x


class GPT(nnx.Module):
    def __init__(self, config: GPTConfig, *, rngs: nnx.Rngs):
        self.config = config

        self.embed = nnx.Embed(
            config.vocab_size,
            config.d_model,
            embedding_init=nnx.initializers.normal(stddev=0.02),
            rngs=rngs,
        )
        self.layers = nnx.data([TransformerBlock(config, rngs=rngs) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, config.eps, rngs=rngs)

        # Match PyTorch init_weights: kaiming_normal_ on all Linear kernels.
        self._reinit_linear_kernels(rngs)
        self._scale_residuals()

    def _reinit_linear_kernels(self, rngs: nnx.Rngs):
        he = nnx.initializers.he_normal()
        for _, m in nnx.iter_modules(self):
            if isinstance(m, nnx.Linear):
                kernel = m.kernel[...]
                m.kernel[...] = he(rngs.params(), kernel.shape, kernel.dtype)

    def _scale_residuals(self):
        scale = math.sqrt(2 * self.config.n_layers)
        for block in self.layers:
            block.attn.Wo.kernel[...] = block.attn.Wo.kernel[...] / scale
            block.mlp.w2.kernel[...] = block.mlp.w2.kernel[...] / scale

    def __call__(self, idx):
        x = self.embed(idx)
        for block in self.layers:
            x = block(x)
        x = self.final_norm(x)
        # Weight tying: reuse the embedding matrix as the LM head.
        logits = x @ self.embed.embedding[...].T
        return logits


def count_parameters(model: nnx.Module):
    state = nnx.state(model, nnx.Param)
    total = sum(int(x.size) for x in jax.tree_util.tree_leaves(state))
    print(f"Total parameters: {total:,}")
    return total


if __name__ == "__main__":
    config = GPTConfig.dev()
    model = GPT(config, rngs=nnx.Rngs(0))
    count_parameters(model)

    idx = jnp.zeros((2, config.block_size), dtype=jnp.int32)
    logits = model(idx)
    print("logits shape:", logits.shape)
