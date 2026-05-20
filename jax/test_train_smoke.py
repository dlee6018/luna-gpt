"""End-to-end smoke test for the JAX training loop.

Uses synthetic random-token batches (no network access / HF auth needed) to verify:
  - compute_grads + apply_grads are jit-compatible and run
  - loss decreases when overfitting a tiny fixed batch
  - params actually change

Run: `uv run python jax/test_train_smoke.py`
"""

import os
import sys
import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_file_location("jx_model", os.path.join(ROOT, "jax", "model.py"))
jx_model = importlib.util.module_from_spec(_spec)
sys.modules["jx_model"] = jx_model
_spec.loader.exec_module(jx_model)

_spec2 = importlib.util.spec_from_file_location("jx_train", os.path.join(ROOT, "jax", "train.py"))
# We only need the jitted fns from train.py, not main(); avoid importing it to not
# pull the HF tokenizer at import-time. Copy them inline instead.

GPT = jx_model.GPT
GPTConfig = jx_model.GPTConfig


def loss_fn(model, tokens, pad_id):
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    logits = model(x)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    B, T, V = logits.shape
    y_flat = y.reshape(-1)
    lp_flat = log_probs.reshape(B * T, V)
    nll = -lp_flat[jnp.arange(B * T), y_flat]
    mask = (y_flat != pad_id).astype(nll.dtype)
    return (nll * mask).sum() / jnp.maximum(mask.sum(), 1.0)


@nnx.jit
def compute_grads(model, tokens, pad_id):
    return nnx.value_and_grad(loss_fn)(model, tokens, pad_id)


@nnx.jit
def apply_grads(model, optimizer, grads, scale):
    grads = jax.tree.map(lambda g: g * scale, grads)
    optimizer.update(model, grads)


def main():
    # Tiny model.
    config = GPTConfig(
        block_size=16,
        vocab_size=64,
        d_model=32,
        n_head=4,
        n_layers=2,
        use_gradient_checkpointing=False,
    )
    model = GPT(config, rngs=nnx.Rngs(0))

    # High-ish LR + no schedule; we want to see loss move fast on a tiny fixed batch.
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=3e-3, b1=0.9, b2=0.95, weight_decay=0.0),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # Fixed synthetic batch (overfit target).
    rng = np.random.default_rng(0)
    B, T = 4, config.block_size
    tokens_np = rng.integers(0, config.vocab_size, size=(B, T)).astype(np.int32)
    tokens = jnp.asarray(tokens_np)
    pad_id = -1  # no padding in synthetic data; any unused id works

    # Snapshot a param before training.
    w_before = np.asarray(model.layers[0].attn.q.kernel[...]).copy()

    losses = []
    N_STEPS = 20
    for step in range(N_STEPS):
        loss, grads = compute_grads(model, tokens, pad_id)
        apply_grads(model, optimizer, grads, 1.0)
        losses.append(float(loss))

    w_after = np.asarray(model.layers[0].attn.q.kernel[...])

    first, last = losses[0], losses[-1]
    w_diff = float(np.max(np.abs(w_after - w_before)))

    print(f"loss[0]={first:.4f}  loss[{N_STEPS-1}]={last:.4f}  Δ={first - last:+.4f}")
    print(f"max |Δparam| (layers[0].attn.q.kernel) = {w_diff:.4e}")

    # Assertions.
    assert np.isfinite(first) and np.isfinite(last), "non-finite loss"
    assert last < first - 0.5, f"loss did not decrease enough: {first:.4f} -> {last:.4f}"
    assert w_diff > 1e-4, f"params barely changed: {w_diff}"

    # Test gradient accumulation path: same batch applied twice, scaled by 0.5,
    # should equal one full update (up to adamw state differences).
    model_a = GPT(config, rngs=nnx.Rngs(1))
    opt_a = nnx.Optimizer(model_a, tx, wrt=nnx.Param)
    _, g = compute_grads(model_a, tokens, pad_id)
    apply_grads(model_a, opt_a, g, 1.0)

    model_b = GPT(config, rngs=nnx.Rngs(1))
    opt_b = nnx.Optimizer(model_b, tx, wrt=nnx.Param)
    _, g1 = compute_grads(model_b, tokens, pad_id)
    _, g2 = compute_grads(model_b, tokens, pad_id)
    g_sum = jax.tree.map(jnp.add, g1, g2)
    apply_grads(model_b, opt_b, g_sum, 0.5)

    a = np.asarray(model_a.layers[0].attn.q.kernel[...])
    b = np.asarray(model_b.layers[0].attn.q.kernel[...])
    grad_accum_diff = float(np.max(np.abs(a - b)))
    print(f"accum-equivalence max|Δ| = {grad_accum_diff:.3e}")
    assert grad_accum_diff < 1e-5, f"grad accumulation not equivalent: {grad_accum_diff}"

    print("\nOK: train loop smoke test passed.")


if __name__ == "__main__":
    main()
