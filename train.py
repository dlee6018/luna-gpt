from dataclasses import dataclass
import os
import torch
from torch.nn import functional as F
from torch.amp import autocast
import pandas as pd
import matplotlib.pyplot as plt
import time
import math

from model.model import GPT, GPTConfig
from data.loader import get_training_corpus, tokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# ----------------------
# PLOTTING
# ----------------------
def plot_loss(log_file="train_log.csv"):
    """Plot training loss vs step from CSV log file."""
    df = pd.read_csv(log_file)
    plt.figure(figsize=(10, 6))
    plt.plot(df["step"], df["loss"], linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_plot.png", dpi=150)
    plt.show()
    print(f"Plot saved to loss_plot.png")


# ----------------------
# TRAIN CONFIG
# ----------------------
@dataclass
class TrainConfig:
    lr: float = 2e-4
    micro_batch_size: int = 8
    accum_steps: int = 32
    max_iters: int = 50000
    log_interval: int = 32

    @classmethod
    def dev(cls):
        return cls(micro_batch_size=2, accum_steps=2, max_iters=32, log_interval=4)


# ----------------------
# CHECKPOINT CONFIG
# ----------------------
@dataclass
class CheckpointConfig:
    path: str = "ckpt.pt"
    best_path: str = "best_model.pt"
    save_every_steps: int = 2000   # optimizer steps
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
best_loss = float("inf")

if os.path.exists(ckpt_cfg.path):
    ckpt = torch.load(ckpt_cfg.path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    step = ckpt["step"]
    best_loss = ckpt.get("best_loss", float("inf"))
    print(f"[resume] loaded checkpoint at step {step}")

model = torch.compile(model)

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
def save_checkpoint(step, loss=None):
    global best_loss
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_loss": best_loss,
        },
        ckpt_cfg.path,
    )

    if ckpt_cfg.save_best and loss is not None:
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), ckpt_cfg.best_path)
            print(f"[checkpoint] new best model saved (loss={loss:.4f})")

warmup_steps = 1000
total_steps = train_cfg.max_iters
min_lr = 1e-5

def get_lr(step, *, base_lr, warmup_steps, total_steps, min_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + cosine_decay * (base_lr - min_lr)

# ----------------------
# TRAIN LOOP
# ----------------------
optimizer.zero_grad()
data_generator = get_training_corpus(
    batch_size=train_cfg.micro_batch_size,
    block_size=config.block_size,
)

# Initialize CSV log file
log_file = "train_log.csv"
if step == 0:
    with open(log_file, "w") as f:
        f.write("step,loss\n")

total_micro_steps = train_cfg.max_iters * train_cfg.accum_steps
for iter in range(total_micro_steps):
    tokens = next(data_generator)

    raw_loss = training_step(tokens)
    loss = raw_loss / train_cfg.accum_steps
    loss.backward()

    if (iter + 1) % train_cfg.accum_steps == 0:
        # t0 = time.time()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step, base_lr=train_cfg.lr, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=min_lr)
        for param in optimizer.param_groups:
            param['lr'] = lr
        optimizer.step()
        optimizer.zero_grad()
        step += 1

        # Logging
        if step % train_cfg.log_interval == 0:
            print(f"step {step:6d} | loss {raw_loss.item():.4f} | lr {lr:.2e}")
            with open(log_file, "a") as f:
                f.write(f"{step},{raw_loss.item()}\n")

        # Periodic checkpoint
        if step % ckpt_cfg.save_every_steps == 0:
            save_checkpoint(step, raw_loss.item())
            print(f"[checkpoint] saved at step {step}")

# plot_loss()

# Latency notes:
# 620ms (start)
# 400ms bf16
# 360ms torch.compile
# 330ms fused adam + TF32
