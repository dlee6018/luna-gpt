from dataclasses import dataclass
import os
from pickle import TRUE
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from torch.amp import autocast
# DataLoader is now handled by get_dataloader in loader.py

import time
import math
from transformers import AutoTokenizer

from model.model import GPT, GPTConfig
from data.loader import get_dataloader
from inference import evaluate_val_loss

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank



    

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
    val_check_interval: int = 1000
    log_interval: int = 10
    num_workers: int = 8

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


# Device will be set after DDP setup
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')  # enables TF32
    torch.backends.cudnn.allow_tf32 = True
# ----------------------
# MODE SELECTION
# ----------------------
DEV_MODE = TRUE  # Set to True for quick iteration, False for real training

if DEV_MODE:
    train_cfg = TrainConfig.dev()
    ckpt_cfg = CheckpointConfig.dev()
    config = GPTConfig.dev()
else:
    train_cfg = TrainConfig()
    ckpt_cfg = CheckpointConfig()
    config = GPTConfig()

local_rank = setup_ddp()
device = f"cuda:{local_rank}"

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-30b", model_max_length=int(1e9))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = GPT(config).to(device)
model = DDP(model, device_ids=[local_rank])

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_cfg.lr,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    fused=(device.startswith("cuda")),
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

model.train()
# ----------------------
# TRAINING STEP
# ----------------------
def training_step(tokens):
    # tokens: pre-tokenized tensor of shape (B, block_size)
    # Move to device if not already there
    tokens = tokens.to(device, non_blocking=True)

    # input and target
    x = tokens[:, :-1]
    y = tokens[:, 1:]

    with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.startswith("cuda"))):
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
    return train_cfg.min_lr + (coeff * (train_cfg.lr - train_cfg.min_lr))

# ----------------------
# TRAIN LOOP
# ----------------------
optimizer.zero_grad()
train_loader = get_dataloader(
    tokenizer=tokenizer,
    batch_size=train_cfg.micro_batch_size,
    block_size=config.block_size,
    train=True,
    num_workers=train_cfg.num_workers
)


# Initialize CSV log file
log_file = "train_log.csv"
if step == 0:
    with open(log_file, "w") as f:
        f.write("step,loss\n")

total_micro_steps = train_cfg.max_iters * train_cfg.accum_steps
print(f"[train] starting training loop, {total_micro_steps} micro-steps", flush=True)

t0 = time.time()

data_iter = iter(train_loader)
for micro_step in range(total_micro_steps):
    try:
        tokens = next(data_iter)
    except StopIteration:
        # Restart iterator when dataset is exhausted
        data_iter = iter(train_loader)
        tokens = next(data_iter)
        epoch += 1

    raw_loss = training_step(tokens)
    loss = raw_loss / train_cfg.accum_steps
    loss.backward()

    if (micro_step + 1) % train_cfg.accum_steps == 0:
        # t0 = time.time()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        # lr = lr * 0.1
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
            save_checkpoint(step, epoch)
            print(f"[checkpoint] saved at step {step}, epoch {epoch}", flush=True)
        
        # log validation loss, for the first 1k, log every 100
        if (step % train_cfg.val_check_interval == 0):
            val_loss = evaluate_val_loss(model, num_batches=10, batch_size=20, block_size=1024)
            print(f"[val] loss: {val_loss:.4f} (over {step} steps)", flush=True)
            val_log_file = "val_log.csv"
            if step == 0:
                with open(val_log_file, "w") as f:
                    f.write("step,loss\n")
            with open(val_log_file, "a") as f:
                f.write(f"{step},{val_loss:.4f}\n")
            # Save best model based on validation loss
            save_checkpoint(step, epoch, val_loss)

# plot_loss()

# Latency notes:
# 620ms (start)
# 400ms bf16
# 360ms torch.compile
# 330ms fused adam + TF32
