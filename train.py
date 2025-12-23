from dataclasses import dataclass
import os
import torch
from torch.nn import functional as F
from torch.amp import autocast

import time
import math

from model.model import GPT, GPTConfig
from data.loader import get_training_corpus, tokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"



# ----------------------
# TRAIN CONFIG
# ----------------------
@dataclass
class TrainConfig:
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 700
    micro_batch_size: int = 20
    accum_steps: int = 10
    max_iters: int = 15000
    log_interval: int = 10

    @classmethod
    def dev(cls):
        return cls(
            micro_batch_size=2, 
            accum_steps=2, 
            max_iters=32, 
            log_interval=4, 
            warmup_steps=10, 
            min_lr=1e-5
        )


# ----------------------
# CHECKPOINT CONFIG
# ----------------------
@dataclass
class CheckpointConfig:
    path: str = "ckpt.pt"
    best_path: str = "best_model.pt"
    save_every_steps: int = 3000   # optimizer steps
    save_best: bool = True
    
    @classmethod
    def dev(cls):
        return cls(
            path="ckpt_dev.pt",
            best_path="best_model_dev.pt",
            save_every_steps=4,
            save_best=False,   # usually unnecessary in dev
        )


device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

if device == "cuda":
    torch.set_float32_matmul_precision('high')  # enables TF32
    torch.backends.cudnn.allow_tf32 = True
# ----------------------
# MODE SELECTION
# ----------------------
DEV_MODE = False  # Set to True for quick iteration, False for real training

if DEV_MODE:
    train_cfg = TrainConfig.dev()
    ckpt_cfg = CheckpointConfig.dev()
    config = GPTConfig.dev()
else:
    train_cfg = TrainConfig()
    ckpt_cfg = CheckpointConfig()
    config = GPTConfig()

model = GPT(config).to(device)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_cfg.lr,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    fused=(device == "cuda"),
)




# ----------------------
# LOAD CHECKPOINT (IF EXISTS)
# ----------------------
step = 0
epoch = 0
best_loss = float("inf")

if os.path.exists(ckpt_cfg.path):
    ckpt = torch.load(ckpt_cfg.path, map_location="cpu")
    state_dict = ckpt["model"]
    unwanted_prefix = '_orig_mod.'
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(ckpt["optimizer"])
    step = ckpt["step"]
    epoch = ckpt.get("epoch", 0)
    best_loss = ckpt.get("best_loss", float("inf"))
    print(f"[resume] loaded checkpoint at step {step}, epoch {epoch}", flush=True)

print("[init] compiling model with torch.compile...", flush=True)
model = torch.compile(model)
print("[init] model compiled, starting data generator...", flush=True)

# ----------------------
# TRAINING STEP
# ----------------------
def training_step(tokens):
    # tokens: pre-tokenized tensor of shape (B, block_size)
    # Move to device if not already there
    tokens = tokens.to(device)

    # input and target
    x = tokens[:, :-1]
    y = tokens[:, 1:]

    with autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
        logits = model(x)
        B, T, V = logits.shape

        loss = F.cross_entropy(
            logits.view(B * T, V),
            y.reshape(B * T),  # y is NOT contiguous so need reshape > view
            ignore_index=tokenizer.pad_token_id,  # ignore padding in loss
        )
    return loss


# ----------------------
# CHECKPOINT SAVE HELPER
# ----------------------
def save_checkpoint(step, epoch, loss=None):
    global best_loss
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "best_loss": best_loss,
        },
        ckpt_cfg.path,
    )

    if ckpt_cfg.save_best and loss is not None:
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), ckpt_cfg.best_path)
            print(f"[checkpoint] new best model saved (loss={loss:.4f})", flush=True)

warmup_steps = 1000
total_steps = train_cfg.max_iters
min_lr = 1e-5

def get_lr(current_step):
    # Linear warmup
    if current_step < train_cfg.warmup_steps:
        return train_cfg.lr * (current_step + 1) / train_cfg.warmup_steps

    # If we go beyond max_iters, return min_lr
    if current_step > train_cfg.max_iters:
        return train_cfg.min_lr

    # Cosine decay
    decay_ratio = (current_step - train_cfg.warmup_steps) / (train_cfg.max_iters - train_cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return train_cfg.min_lr + coeff * (train_cfg.lr - train_cfg.min_lr)

# ----------------------
# TRAIN LOOP
# ----------------------
optimizer.zero_grad()
data_generator, data_state = get_training_corpus(
    batch_size=train_cfg.micro_batch_size,
    block_size=config.block_size,
    train=True,
    start_epoch=epoch,
)

# Initialize CSV log file
log_file = "train_log.csv"
if step == 0:
    with open(log_file, "w") as f:
        f.write("step,loss\n")

total_micro_steps = train_cfg.max_iters * train_cfg.accum_steps
print(f"[train] starting training loop, {total_micro_steps} micro-steps", flush=True)

t0 = time.time()

for iter in range(total_micro_steps):
    tokens = next(data_generator)
    if data_state["epoch"] > epoch:
        epoch = data_state["epoch"]

    raw_loss = training_step(tokens)
    loss = raw_loss / train_cfg.accum_steps
    loss.backward()

    if (iter + 1) % train_cfg.accum_steps == 0:
        # t0 = time.time()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param in optimizer.param_groups:
            param['lr'] = lr
        optimizer.step()
        optimizer.zero_grad()
        step += 1

        # Logging
        if step % train_cfg.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            # Approximate tokens per sec
            tokens_per_sec = (train_cfg.log_interval * train_cfg.accum_steps * train_cfg.micro_batch_size * config.block_size) / dt
            
            print(f"step {step:5d} | loss {raw_loss.item():.4f} | lr {lr:.2e} | {tokens_per_sec:.0f} tok/s", flush=True)
            with open(log_file, "a") as f:
                f.write(f"{step},{raw_loss.item()}\n")

        # Periodic checkpoint
        if step % ckpt_cfg.save_every_steps == 0:
            save_checkpoint(step, epoch, raw_loss.item())
            print(f"[checkpoint] saved at step {step}, epoch {epoch}", flush=True)

# plot_loss()

# Latency notes:
# 620ms (start)
# 400ms bf16
# 360ms torch.compile
# 330ms fused adam + TF32
