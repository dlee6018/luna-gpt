from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import math


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 32000
    d_model: int = 768 #1536
    n_head: int = 12 #24
    n_layers: int = 12 #24
    eps: float = 1e-5
    use_gradient_checkpointing: bool = False

    @property
    def d_head(self):
        return self.d_model // self.n_head

    @classmethod
    def dev(cls):
        return cls(block_size=64, d_model=256, n_head=4, n_layers=4, use_gradient_checkpointing=False)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # learnable scale parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Compute RMS along last dimension
        # rms = sqrt(mean(x^2))
        rms = x.norm(2, dim=-1, keepdim=True) / math.sqrt(x.size(-1))
        return self.weight * (x / (rms + self.eps))


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.d_head = config.d_head  # dimension per head

        # Project to q, k, v (full projection)
        self.q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v = nn.Linear(self.d_model, self.d_model, bias=False)

        # Output projection back to d_model
        self.Wo = nn.Linear(self.d_model, self.d_model, bias=False)

        # RoPE
        self.rope = RotaryEmbedding(config.d_head)

    def forward(self, x):
        B, T, D = x.shape

        # Compute q,k,v
        q = self.q(x)  # (B, T, D)
        k = self.k(x)
        v = self.v(x)

        # Reshape into heads: (B, n_head, T, d_head)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        q, k = self.rope(q, k)  # apply RoPE after reshaping

        # PyTorch 2.0+ built-in Flash Attention / SDPA
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True   # autoregressive mask
        )  # shape: (B, n_head, T, d_head)

        # Merge heads back: (B, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        return self.Wo(out)


class RotaryEmbedding(nn.Module):
    """
    Standard RoPE (Rotary Positional Embedding)
    Used in LLaMA / Qwen / Mistral (without NTK scaling).
    """
    def __init__(self, dim, base=10000):
        """
        dim: rotary dimension (must be even), head dimension
        base: frequency base (default 10,000, same as GPT/LLaMA)
        """
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

        # Create inverse frequency: shape [dim/2]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)  # stays on same device as model

    def _compute_angles(self, seq_len, device, dtype):
        """
        Compute cos/sin for all positions.
        returns: cos, sin with shape [seq_len, dim/2]
        """
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # pos * inv_freq
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)
        return cos, sin

    def apply_rotary(self, x, cos, sin):
        """
        Apply RoPE to x.
        x:   [batch, heads, seq, dim]
        cos: [seq, dim/2]
        sin: [seq, dim/2]
        """
        # Split even / odd
        x_part = x[..., ::2]
        y_part = x[..., 1::2]

        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        x_rot = x_part * cos - y_part * sin
        y_rot = x_part * sin + y_part * cos

        x_out = torch.stack((x_rot, y_rot), dim=-1)
        return x_out.flatten(-2)

    def forward(self, q, k):
        seq_len = q.size(-2)
        cos, sin = self._compute_angles(seq_len, q.device, q.dtype)
        return self.apply_rotary(q, cos, sin), self.apply_rotary(k, cos, sin)


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = 4 * config.d_model  # standard hidden expansion

        # SwiGLU = silu(W1x) * (W3x)
        self.w1 = nn.Linear(config.d_model, hidden, bias=False)
        self.w3 = nn.Linear(config.d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, config.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, config.eps)
        self.attn = MultiHeadAttention(config)
        self.norm2 = RMSNorm(config.d_model, config.eps)
        self.mlp = FeedForward(config)
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

    def forward(self, x):
        if self.use_gradient_checkpointing and self.training:
            # Gradient checkpointing: trade compute for memory
            x = x + checkpoint(self._attn_block, x, use_reentrant=False)
            x = x + checkpoint(self._mlp_block, x, use_reentrant=False)
        else:
            # Pre-norm residual attention
            x = x + self.attn(self.norm1(x))
            # Pre-norm residual MLP
            x = x + self.mlp(self.norm2(x))
        return x

    def _attn_block(self, x):
        return self.attn(self.norm1(x))

    def _mlp_block(self, x):
        return self.mlp(self.norm2(x))



class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        # Positional embedding (absolute, not RoPE)
        # self.pos = nn.Embedding(config.block_size, config.d_model)

        # Transformer layers
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Final RMSNorm
        self.final_norm = RMSNorm(config.d_model, config.eps)

        # LM head (Language Modeling)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)  # no bias b.c RMS norm right before

        # weight tying (GPT-style)
        self.lm_head.weight = self.embed.weight

        # apply initialization
        self.apply(init_weights)
        self._scale_residuals()

    def _scale_residuals(self):
        # GPT-2 / LLaMA-style residual scaling for deep stability
        scale = math.sqrt(2 * self.config.n_layers)
        for block in self.layers:
            block.attn.Wo.weight.data /= scale
            block.mlp.w2.weight.data /= scale

    def forward(self, idx):
        B, T = idx.shape

        # Token embedding + positional embedding
        x = self.embed(idx)

        # Run through Transformer blocks
        for block in self.layers:
            x = block(x)

        # Final norm & LM logits
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits



def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, RMSNorm):
        nn.init.ones_(m.weight)



def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return total, trainable

if __name__ == "__main__":
    config = GPTConfig()
    model = GPT(config)
    count_parameters(model)