"""Parity tests: PyTorch model == JAX model (same weights, same input, same output).

Run: `uv run python jax/test_parity.py`
"""

import importlib.util
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pt = _load("pt_model", os.path.join(ROOT, "model", "model.py"))
jx = _load("jx_model", os.path.join(ROOT, "jax", "model.py"))


# ------------------------- helpers -------------------------

def t2j(x: torch.Tensor) -> jnp.ndarray:
    return jnp.asarray(x.detach().cpu().numpy())


def j2np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def assert_close(a, b, atol=1e-5, rtol=1e-5, name=""):
    a = j2np(a)
    b = j2np(b)
    if a.shape != b.shape:
        raise AssertionError(f"[{name}] shape mismatch: {a.shape} vs {b.shape}")
    diff = np.max(np.abs(a - b))
    ok = np.allclose(a, b, atol=atol, rtol=rtol)
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {name}: max|Δ|={diff:.3e}  shape={a.shape}")
    if not ok:
        raise AssertionError(f"[{name}] outputs differ (max|Δ|={diff:.3e})")


def dev_config_pt():
    # Disable gradient checkpointing to keep a clean comparison.
    return pt.GPTConfig(block_size=32, vocab_size=256, d_model=64, n_head=4, n_layers=2,
                        use_gradient_checkpointing=False)


def dev_config_jx():
    return jx.GPTConfig(block_size=32, vocab_size=256, d_model=64, n_head=4, n_layers=2,
                        use_gradient_checkpointing=False)


# --------------------- weight transfer ---------------------

def copy_linear(pt_lin: torch.nn.Linear, jx_lin: nnx.Linear):
    # PyTorch Linear.weight: (out, in). Flax nnx.Linear.kernel: (in, out).
    jx_lin.kernel[...] = t2j(pt_lin.weight.T)


def copy_rmsnorm(pt_n, jx_n):
    jx_n.weight[...] = t2j(pt_n.weight)


def copy_attention(pt_a, jx_a):
    copy_linear(pt_a.q, jx_a.q)
    copy_linear(pt_a.k, jx_a.k)
    copy_linear(pt_a.v, jx_a.v)
    copy_linear(pt_a.Wo, jx_a.Wo)


def copy_mlp(pt_m, jx_m):
    copy_linear(pt_m.w1, jx_m.w1)
    copy_linear(pt_m.w2, jx_m.w2)
    copy_linear(pt_m.w3, jx_m.w3)


def copy_block(pt_b, jx_b):
    copy_rmsnorm(pt_b.norm1, jx_b.norm1)
    copy_rmsnorm(pt_b.norm2, jx_b.norm2)
    copy_attention(pt_b.attn, jx_b.attn)
    copy_mlp(pt_b.mlp, jx_b.mlp)


def copy_gpt(pt_m, jx_m):
    # Embedding: both stored as (vocab, d_model).
    jx_m.embed.embedding[...] = t2j(pt_m.embed.weight)
    copy_rmsnorm(pt_m.final_norm, jx_m.final_norm)
    for pt_b, jx_b in zip(pt_m.layers, jx_m.layers):
        copy_block(pt_b, jx_b)


# ----------------------- individual tests -----------------------

def test_rmsnorm():
    print("test_rmsnorm")
    dim = 64
    pt_m = pt.RMSNorm(dim, eps=1e-5)
    torch.nn.init.normal_(pt_m.weight, mean=1.0, std=0.1)

    jx_m = jx.RMSNorm(dim, eps=1e-5, rngs=nnx.Rngs(0))
    copy_rmsnorm(pt_m, jx_m)

    x_np = np.random.randn(2, 5, dim).astype(np.float32)
    pt_out = pt_m(torch.from_numpy(x_np))
    jx_out = jx_m(jnp.asarray(x_np))

    assert_close(pt_out, jx_out, name="RMSNorm")


def test_rope():
    print("test_rope")
    # PyTorch: (B, H, T, d). JAX: (B, T, H, d). Feed the same underlying (B, T, H, d)
    # and reshape for PyTorch.
    B, T, H, d = 2, 8, 4, 16
    x_np = np.random.randn(B, T, H, d).astype(np.float32)
    x_pt = torch.from_numpy(x_np).transpose(1, 2).contiguous()  # (B, H, T, d)
    x_jx = jnp.asarray(x_np)  # (B, T, H, d)

    pt_rope = pt.RotaryEmbedding(d)
    q_pt, k_pt = pt_rope(x_pt, x_pt)  # same q/k
    # transpose back: (B, T, H, d)
    q_pt = q_pt.transpose(1, 2).contiguous()

    jx_rope = jx.RotaryEmbedding(d, max_seq_len=T)
    q_jx, k_jx = jx_rope(x_jx, x_jx)

    assert_close(q_pt, q_jx, name="RoPE")


def test_feedforward():
    print("test_feedforward")
    cfg_pt = dev_config_pt()
    cfg_jx = dev_config_jx()
    pt_m = pt.FeedForward(cfg_pt)
    jx_m = jx.FeedForward(cfg_jx, rngs=nnx.Rngs(0))
    copy_mlp(pt_m, jx_m)

    x_np = np.random.randn(2, 5, cfg_pt.d_model).astype(np.float32)
    pt_out = pt_m(torch.from_numpy(x_np))
    jx_out = jx_m(jnp.asarray(x_np))

    assert_close(pt_out, jx_out, name="FeedForward", atol=1e-5)


def test_attention():
    print("test_attention")
    cfg_pt = dev_config_pt()
    cfg_jx = dev_config_jx()
    pt_m = pt.MultiHeadAttention(cfg_pt)
    jx_m = jx.MultiHeadAttention(cfg_jx, rngs=nnx.Rngs(0))
    copy_attention(pt_m, jx_m)

    B, T, D = 2, cfg_pt.block_size, cfg_pt.d_model
    x_np = np.random.randn(B, T, D).astype(np.float32)
    pt_out = pt_m(torch.from_numpy(x_np))
    jx_out = jx_m(jnp.asarray(x_np))

    # SDPA backends can differ slightly from manual attention — use a looser tol.
    assert_close(pt_out, jx_out, name="MultiHeadAttention", atol=1e-4)


def test_transformer_block():
    print("test_transformer_block")
    cfg_pt = dev_config_pt()
    cfg_jx = dev_config_jx()
    pt_b = pt.TransformerBlock(cfg_pt)
    pt_b.eval()  # disables gradient_checkpointing (also already off in dev cfg)
    jx_b = jx.TransformerBlock(cfg_jx, rngs=nnx.Rngs(0))
    copy_block(pt_b, jx_b)

    B, T, D = 2, cfg_pt.block_size, cfg_pt.d_model
    x_np = np.random.randn(B, T, D).astype(np.float32)
    with torch.no_grad():
        pt_out = pt_b(torch.from_numpy(x_np))
    jx_out = jx_b(jnp.asarray(x_np))

    assert_close(pt_out, jx_out, name="TransformerBlock", atol=1e-4)


def test_gpt_full():
    print("test_gpt_full")
    cfg_pt = dev_config_pt()
    cfg_jx = dev_config_jx()

    pt_m = pt.GPT(cfg_pt)
    pt_m.eval()
    jx_m = jx.GPT(cfg_jx, rngs=nnx.Rngs(0))
    copy_gpt(pt_m, jx_m)

    B, T = 2, cfg_pt.block_size
    idx_np = np.random.randint(0, cfg_pt.vocab_size, size=(B, T)).astype(np.int64)
    with torch.no_grad():
        pt_logits = pt_m(torch.from_numpy(idx_np))
    jx_logits = jx_m(jnp.asarray(idx_np, dtype=jnp.int32))

    # Accumulated over many layers — loosen tol further.
    assert_close(pt_logits, jx_logits, name="GPT(full)", atol=5e-4)

    # Also test argmax agreement (more robust against small numerical drift).
    pt_arg = pt_logits.argmax(dim=-1).cpu().numpy()
    jx_arg = np.asarray(jnp.argmax(jx_logits, axis=-1))
    agree = (pt_arg == jx_arg).mean()
    print(f"  argmax agreement: {agree*100:.2f}%")
    assert agree > 0.99, f"argmax disagreement too high: {agree}"


# ----------------------- runner -----------------------

def main():
    np.random.seed(0)
    torch.manual_seed(0)

    tests = [
        test_rmsnorm,
        test_rope,
        test_feedforward,
        test_attention,
        test_transformer_block,
        test_gpt_full,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  !! {t.__name__} FAILED: {e}")
            failures.append(t.__name__)
        print()

    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    print("All parity tests passed.")


if __name__ == "__main__":
    main()
