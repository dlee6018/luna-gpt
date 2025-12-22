from dataclasses import dataclass
import os
import torch
from torch.nn import functional as F

from model.model import GPT, GPTConfig
from model.tokenizer import tokenizer
from data.loader import get_training_corpus


# ----------------------
# TRAIN CONFIG
# ----------------------
@dataclass
class TrainConfig:
    lr: float = 3e-4
    micro_batch_size: int = 8
    accum_steps: int = 32
    max_iters: int = 200000
    log_interval: int = 50

    @classmethod
    def dev(cls):
        return cls(micro_batch_size=4, accum_steps=4, max_iters=32, log_interval=4)


# ----------------------
# CHECKPOINT CONFIG
# ----------------------
@dataclass
class CheckpointConfig:
    path: str = "ckpt.pt"
    best_path: str = "best_model.pt"
    save_every_steps: int = 500   # optimizer steps
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

train_cfg = TrainConfig.dev()
ckpt_cfg = CheckpointConfig.dev()
config = GPTConfig.dev()

model = GPT(config).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_cfg.lr,
    betas=(0.9, 0.95),
    weight_decay=0.1,
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
    print(f"[resume] loaded checkpoint at step {step}")


# ----------------------
# TRAINING STEP
# ----------------------
def training_step(batch):
    # batch: list of raw text

    # tokenize
    encoded = [tokenizer.encode(x, max_length=config.block_size, truncation=True) for x in batch]

    # pad / trim to block_size
    block = config.block_size
    tokens = []
    for t in encoded:
        if len(t) < block:
            t = t + [tokenizer.pad_token_id] * (block - len(t))
        else:
            t = t[:block]
        tokens.append(t)

    tokens = torch.tensor(tokens, dtype=torch.long, device=device)

    # input and target
    x = tokens[:, :-1]
    y = tokens[:, 1:]

    logits = model(x)
    B, T, V = logits.shape

    loss = F.cross_entropy(
        logits.view(B * T, V),
        y.reshape(B * T), # y is NOT contiguous so need reshape > view
    )
    return loss


# ----------------------
# CHECKPOINT SAVE HELPER
# ----------------------
def save_checkpoint(step, loss=None):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        ckpt_cfg.path,
    )

    if ckpt_cfg.save_best and loss is not None:
        global best_loss
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), ckpt_cfg.best_path)
            print(f"[checkpoint] new best model saved (loss={loss:.4f})")


# ----------------------
# TRAIN LOOP
# ----------------------
optimizer.zero_grad()
data_generator = get_training_corpus(train_cfg.micro_batch_size)

for iter in range(train_cfg.max_iters):
    batch = next(data_generator)

    raw_loss = training_step(batch)
    loss = raw_loss / train_cfg.accum_steps
    loss.backward()

    if (iter + 1) % train_cfg.accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        step += 1

        # periodic checkpoint
        if step % ckpt_cfg.save_every_steps == 0:
            save_checkpoint(step, loss.item())
            print(f"[checkpoint] saved at step {step}")

    if iter % train_cfg.log_interval == 0:
        print(f"iter {iter} step {step} | loss {raw_loss.item():.4f}")
