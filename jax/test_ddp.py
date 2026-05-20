"""Multi-device / data-parallel parity test.

Uses JAX's CPU-device emulation to simulate N devices, then checks that:
  1. Replicated model state is actually sharded across N devices (each slice equal).
  2. A batch-sharded forward+grad produces the SAME loss and gradient values as
     running the exact same batch on 1 device (data-parallel is mathematically
     identical to single-device, just split across devices).

Run: `uv run python jax/test_ddp.py`
"""

import os

# MUST be set before `import jax` — this is how JAX picks up the flag.
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import sys
import importlib.util

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P, AxisType
import numpy as np
import optax
from flax import nnx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_file_location("jx_model", os.path.join(ROOT, "jax", "model.py"))
jx_model = importlib.util.module_from_spec(_spec)
sys.modules["jx_model"] = jx_model
_spec.loader.exec_module(jx_model)

GPT = jx_model.GPT
GPTConfig = jx_model.GPTConfig


# Copied from train.py to avoid pulling in its HF imports.
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


def build(cfg, seed):
    model = GPT(cfg, rngs=nnx.Rngs(seed))
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1e-3, b1=0.9, b2=0.95, weight_decay=0.0),
    )
    opt = nnx.Optimizer(model, tx, wrt=nnx.Param)
    return model, opt


def test_replication_and_parity():
    devices = jax.devices()
    num_devices = len(devices)
    print(f"Simulated devices ({num_devices}): {devices}")
    assert num_devices >= 2, "expected >=2 simulated devices for this test"

    cfg = GPTConfig(
        block_size=16,
        vocab_size=64,
        d_model=32,
        n_head=4,
        n_layers=2,
        use_gradient_checkpointing=False,
    )

    # Global batch of size B_global, split evenly across num_devices.
    B_global, T = 4 * num_devices, cfg.block_size
    rng = np.random.default_rng(0)
    tokens_np = rng.integers(0, cfg.vocab_size, size=(B_global, T)).astype(np.int32)

    # ---- Run 1: single device (reference) ----
    single_dev_sharding = NamedSharding(
        jax.make_mesh((1,), ("data",), axis_types=(AxisType.Auto,)), P("data")
    )
    model_ref, opt_ref = build(cfg, seed=42)
    batch_ref = jax.device_put(jnp.asarray(tokens_np), single_dev_sharding)
    loss_ref, grads_ref = compute_grads(model_ref, batch_ref, pad_id=-1)
    apply_grads(model_ref, opt_ref, grads_ref, 1.0)
    w_ref = np.asarray(model_ref.layers[0].attn.q.kernel[...])

    # ---- Run 2: data-parallel across num_devices ----
    mesh = jax.make_mesh((num_devices,), ("data",), axis_types=(AxisType.Auto,))
    rep = NamedSharding(mesh, P())
    data = NamedSharding(mesh, P("data"))

    model_dp, opt_dp = build(cfg, seed=42)
    # Replicate.
    nnx.update(model_dp, jax.device_put(nnx.state(model_dp), rep))
    nnx.update(opt_dp, jax.device_put(nnx.state(opt_dp), rep))

    # Verify replication: embedding should show same bytes on every device.
    emb = model_dp.embed.embedding[...]
    assert emb.sharding.is_fully_replicated, f"params should be replicated, got {emb.sharding}"

    # Shard the batch.
    batch_dp = jax.device_put(jnp.asarray(tokens_np), data)
    assert not batch_dp.sharding.is_fully_replicated, "batch must be sharded"

    loss_dp, grads_dp = compute_grads(model_dp, batch_dp, pad_id=-1)
    # Check grads come back replicated.
    sample_grad = jax.tree.leaves(grads_dp)[0]
    assert sample_grad.sharding.is_fully_replicated, (
        f"grads must be replicated (all-reduced), got {sample_grad.sharding}"
    )

    apply_grads(model_dp, opt_dp, grads_dp, 1.0)
    w_dp = np.asarray(model_dp.layers[0].attn.q.kernel[...])

    # ---- Compare ----
    loss_diff = abs(float(loss_ref) - float(loss_dp))
    grad_diff = float(
        max(
            np.max(np.abs(np.asarray(a) - np.asarray(b)))
            for a, b in zip(jax.tree.leaves(grads_ref), jax.tree.leaves(grads_dp))
        )
    )
    w_diff = float(np.max(np.abs(w_ref - w_dp)))

    print(f"loss  : single={float(loss_ref):.6f}  dp={float(loss_dp):.6f}  Δ={loss_diff:.2e}")
    print(f"grads : max|Δ|={grad_diff:.2e}")
    print(f"params: max|Δ|={w_diff:.2e}")

    assert loss_diff < 1e-5, f"loss mismatch: {loss_diff}"
    assert grad_diff < 1e-5, f"grad mismatch: {grad_diff}"
    assert w_diff < 1e-5, f"updated param mismatch: {w_diff}"

    print("\nOK: data-parallel matches single-device exactly.")


if __name__ == "__main__":
    test_replication_and_parity()
