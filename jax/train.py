"""JAX/Flax NNX training loop — port of train.py.

Reuses the existing PyTorch DataLoader in data/loader.py (tokenization + packing +
streaming dataset mixing). Each batch is converted to a jnp array at the step boundary.

Run: `uv run python jax/train.py`
"""

import os
import sys
import time
import pickle
import importlib.util
from dataclasses import dataclass, asdict

# Project root on sys.path for data/ and friends.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P, AxisType
import optax
from flax import nnx
from transformers import AutoTokenizer

from data.loader import get_dataloader

# Load jax/model.py by explicit path (avoids colliding with the top-level `model/` package).
_spec = importlib.util.spec_from_file_location(
    "jx_model", os.path.join(ROOT, "jax", "model.py")
) # allow you to load module as jx_model. Doing this b.c there is already model/
jx_model = importlib.util.module_from_spec(_spec)
sys.modules["jx_model"] = jx_model # correct single initialization
_spec.loader.exec_module(jx_model) # runs it actually

GPT = jx_model.GPT
GPTConfig = jx_model.GPTConfig


# ----------------------
# TRAIN CONFIG
# ----------------------
@dataclass
class TrainConfig:
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 700
    micro_batch_size: int = 6
    accum_steps: int = 192
    max_iters: int = 15000
    log_interval: int = 10
    num_workers: int = 8
    grad_clip: float = 1.0
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.1

    @classmethod
    def dev(cls):
        return cls(
            micro_batch_size=2,
            accum_steps=2,
            max_iters=32,
            log_interval=4,
            warmup_steps=10,
            min_lr=1e-5,
            num_workers=0,  # avoid macOS spawn overhead in dev
        )


@dataclass
class CheckpointConfig:
    path: str = "ckpt_jax.pkl"
    save_every_steps: int = 3000

    @classmethod
    def dev(cls):
        return cls(path="ckpt_jax_dev.pkl", save_every_steps=4)


DEV_MODE = True


# ----------------------
# LOSS + JITTED STEP FUNCTIONS
# ----------------------
def _loss_fn(model, tokens, pad_id):
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    logits = model(x)  # (B, T, V)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    B, T, V = logits.shape
    y_flat = y.reshape(-1) # (B,T) -> (B * T), -1 = flatten everything
    lp_flat = log_probs.reshape(B * T, V) # (B,T,V) -> (B * T, V)
    nll = -lp_flat[jnp.arange(B * T), y_flat] # (B * T, V) -> (B * T), where y_flat is the index of the target token
    mask = (y_flat != pad_id).astype(nll.dtype) # zero out pad token
    return (nll * mask).sum() / jnp.maximum(mask.sum(), 1.0)

# use nnx.jit when it interacts with nnx.Module, jax.jit only for pure mathematical functions.
# jax.jit only deals with PyTrees which are nested list/dicts of immutable JAX arrays
@nnx.jit # compiles a python callable to XLA
def compute_grads(model, tokens, pad_id):
    """
    Computes the loss and gradients of the loss with respect to the model parameters.

    Returns:
        loss: Scalar loss value.
        grads: Gradients with respect to model parameters.
    """
    loss, grads = nnx.value_and_grad(_loss_fn)(model, tokens, pad_id)
    return loss, grads
# Trace - running a JAX-transformed function builds a jaxpr
# becomes StableHLO -> XLA (compiler for array programs which optimizes to specific backend, e.g: GPU/CUDA)

@nnx.jit
def apply_grads(model, optimizer, grads, scale):
    grads = jax.tree.map(lambda g: g * scale, grads) # tree.map applies recursively to all leaves
    optimizer.update(model, grads)


def to_jnp(torch_tokens):
    return jnp.asarray(torch_tokens.numpy(), dtype=jnp.int32) # convert torch tensor to jnp array


# ----------------------
# MAIN
# ----------------------
def main():
    if DEV_MODE:
        train_cfg = TrainConfig.dev()
        ckpt_cfg = CheckpointConfig.dev()
        config = GPTConfig.dev()
    else:
        train_cfg = TrainConfig()
        ckpt_cfg = CheckpointConfig()
        config = GPTConfig()

    devices = jax.devices()
    num_devices = len(devices)
    print(f"[init] JAX devices ({num_devices}): {devices}", flush=True)

    # Data-parallel mesh: one axis "data", spanning all local devices.
    # Single-device runs still go through this path with a 1-device mesh (no-op cost).
    # AxisType.Auto — GSPMD infers collectives automatically (vs. Explicit, which is
    # stricter and rejects ops like jnp.take over a sharded index).
    mesh = jax.make_mesh((num_devices,), ("data",), axis_types=(AxisType.Auto,))
    rep_sharding = NamedSharding(mesh, P())            # replicated across all devices
    data_sharding = NamedSharding(mesh, P("data"))     # sharded along batch axis

    if train_cfg.micro_batch_size % num_devices != 0:
        raise ValueError(
            f"micro_batch_size={train_cfg.micro_batch_size} must be divisible by "
            f"num_devices={num_devices} for data parallelism."
        )

    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-30b", model_max_length=int(1e9))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = int(tokenizer.pad_token_id)

    model = GPT(config, rngs=nnx.Rngs(0))
    # Optax = library for optimization on pytrees. 
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=train_cfg.lr,
        warmup_steps=train_cfg.warmup_steps,
        decay_steps=train_cfg.max_iters,
        end_value=train_cfg.min_lr,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(train_cfg.grad_clip),
        optax.adamw(
            learning_rate=lr_schedule,
            b1=train_cfg.betas[0],
            b2=train_cfg.betas[1],
            weight_decay=train_cfg.weight_decay,
        ),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param) # only model parameters should be optimized.

    # --- Data-parallel: replicate model + optimizer state across all devices. ---
    # After this, every device holds an identical copy of params and Adam state.
    # Gradients computed from a batch-sharded input will be automatically
    # all-reduced by XLA (because the output — grad w.r.t. replicated params —
    # must be replicated).
    nnx.update(model, jax.device_put(nnx.state(model), rep_sharding))
    nnx.update(optimizer, jax.device_put(nnx.state(optimizer), rep_sharding))

    step = 0
    epoch = 0

    if os.path.exists(ckpt_cfg.path):
        with open(ckpt_cfg.path, "rb") as f:
            ckpt = pickle.load(f)
        nnx.update(model, ckpt["model_state"])
        nnx.update(optimizer, ckpt["opt_state"])
        step = ckpt["step"]
        epoch = ckpt.get("epoch", 0)
        print(f"[resume] loaded checkpoint at step {step}, epoch {epoch}", flush=True)

    def save_checkpoint(step, epoch):
        with open(ckpt_cfg.path, "wb") as f:
            pickle.dump(
                {
                    "model_state": nnx.state(model),
                    "opt_state": nnx.state(optimizer),
                    "step": step,
                    "epoch": epoch,
                    "config": asdict(config),
                },
                f,
            )

    train_loader = get_dataloader(
        tokenizer=tokenizer,
        batch_size=train_cfg.micro_batch_size,
        block_size=config.block_size,
        train=True,
        num_workers=train_cfg.num_workers,
    )

    log_file = "train_jax_log.csv"
    if step == 0:
        with open(log_file, "w") as f:
            f.write("step,loss,lr\n")

    total_micro_steps = train_cfg.max_iters * train_cfg.accum_steps
    print(f"[train] starting training loop, {total_micro_steps} micro-steps", flush=True)

    t0 = time.time()
    data_iter = iter(train_loader)
    grads_accum = None
    micro_in_window = 0
    last_raw_loss = None

    for _ in range(total_micro_steps):
        try:
            tokens = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            tokens = next(data_iter)
            epoch += 1

        batch = jax.device_put(to_jnp(tokens), data_sharding)
        loss_val, grads = compute_grads(model, batch, pad_id)

        if grads_accum is None:
            grads_accum = grads
        else:
            grads_accum = jax.tree.map(jnp.add, grads_accum, grads)
        micro_in_window += 1
        last_raw_loss = loss_val

        if micro_in_window == train_cfg.accum_steps:
            apply_grads(model, optimizer, grads_accum, 1.0 / train_cfg.accum_steps)
            grads_accum = None
            micro_in_window = 0
            step += 1

            if step % train_cfg.log_interval == 0:
                raw_loss_float = float(last_raw_loss)  # forces device->host sync
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                tokens_per_sec = (
                    train_cfg.log_interval
                    * train_cfg.accum_steps
                    * train_cfg.micro_batch_size
                    * config.block_size
                ) / dt
                lr_now = float(lr_schedule(step - 1))
                print(
                    f"step {step:5d} | loss {raw_loss_float:.4f} | "
                    f"lr {lr_now:.2e} | {tokens_per_sec:.0f} tok/s",
                    flush=True,
                )
                with open(log_file, "a") as f:
                    f.write(f"{step},{raw_loss_float},{lr_now}\n")

            if step % ckpt_cfg.save_every_steps == 0:
                save_checkpoint(step, epoch)
                print(f"[checkpoint] saved at step {step}, epoch {epoch}", flush=True)

    print("[train] done.", flush=True)


if __name__ == "__main__":
    main()
