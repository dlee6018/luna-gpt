import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model.model import GPT, GPTConfig
from data.loader import get_dataloader
import matplotlib.pyplot as plt
import pandas as pd


def plot_loss(log_file="train_log.csv", output_file="loss_plot.png", smoothing=0.9):
    """Plot training loss from CSV log file with optional smoothing."""
    df = pd.read_csv(log_file)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Raw loss (faint)
    ax.plot(df["step"], df["loss"], alpha=0.3, color="blue", label="Raw")
    
    # Smoothed loss (EMA)
    smoothed = df["loss"].ewm(alpha=1 - smoothing).mean()
    ax.plot(df["step"], smoothed, color="blue", linewidth=2, label=f"Smoothed (α={smoothing})")
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"[plot] saved to {output_file}")

def load_inference_model(
    ckpt_path="best_model.pt",
    model_class=GPT,
    config_class=GPTConfig,
    tokenizer_name="huggyllama/llama-30b"
):
    """Loads the model and tokenizer for inference, including weights from checkpoint."""
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=int(1e9))
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    config = config_class()
    model = model_class(config).to(device)

    # Load checkpoint weights
    state_dict = torch.load(ckpt_path, map_location=device)
    # If loading a checkpoint with just the weights dict, or a full dict with {"model": ...}
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    
    # Remove unwanted prefixes (DDP adds 'module.', compiled models add '_orig_mod.')
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        # Remove '_orig_mod.' prefix (from torch.compile)
        if new_key.startswith('_orig_mod.'):
            new_key = new_key[len('_orig_mod.'):]
        # Remove 'module.' prefix (from DDP)
        if new_key.startswith('module.'):
            new_key = new_key[len('module.'):]
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.eval()
    print(f"[inference] loaded weights from {ckpt_path}")
    return model, tokenizer, config, device



@torch.no_grad()
def generate(model, tokenizer, config, device, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50) -> str:
    """Generate text from a prompt string."""
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    for _ in range(max_new_tokens):
        # Crop to block_size if needed
        idx = tokens if tokens.size(1) <= config.block_size else tokens[:, -config.block_size:]
        
        logits = model(idx)
        logits = logits[:, -1, :] / temperature
        
        # Top-k sampling
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(tokens[0], skip_special_tokens=True)


@torch.no_grad()
def evaluate_val_loss(model, num_batches=50, batch_size=8, block_size=1024):
    """Evaluate average validation loss over num_batches."""
    # Get device from model
    device = next(model.parameters()).device
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-30b", model_max_length=int(1e9))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    val_loader = get_dataloader(
        tokenizer=tokenizer,
        batch_size=batch_size,
        block_size=block_size,
        train=False,
        num_workers=2
    )
    
    total_loss = 0.0
    data_iter = iter(val_loader)
    for _ in range(num_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(val_loader)
            batch = next(data_iter)
        
        batch = batch.to(device, non_blocking=True)
        x, y = batch[:, :-1], batch[:, 1:]
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            y.reshape(-1),
            ignore_index=tokenizer.pad_token_id
        )
        total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    # print(f"[val] loss: {avg_loss:.4f} (over {num_batches} batches)")
    return avg_loss


@torch.no_grad()
def evaluate_gpt2_val_loss(num_batches=50, batch_size=8, block_size=1024):
    """Evaluate GPT-2 from HuggingFace on validation set for comparison."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load GPT-2
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    gpt2_model.eval()
    
    # Initialize LLaMA tokenizer for decoding
    llama_tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-30b", model_max_length=int(1e9))
    if llama_tokenizer.pad_token is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token
    
    # Get validation data (tokenized with LLaMA tokenizer)
    val_loader = get_dataloader(
        tokenizer=llama_tokenizer,
        batch_size=batch_size,
        block_size=block_size,
        train=False,
        num_workers=2
    )
    
    total_loss = 0.0
    data_iter = iter(val_loader)
    for _ in range(num_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(val_loader)
            batch = next(data_iter)
        
        # Decode from LLaMA tokens to text, then re-encode with GPT-2 tokenizer
        texts = [llama_tokenizer.decode(seq, skip_special_tokens=True) for seq in batch]
        
        # Tokenize with GPT-2
        gpt2_inputs = gpt2_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=block_size,
        ).to(device)
        
        input_ids = gpt2_inputs["input_ids"]
        attention_mask = gpt2_inputs["attention_mask"]
        
        # Forward pass
        outputs = gpt2_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        
        # Compute loss (ignore padding)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=gpt2_tokenizer.pad_token_id,
        )
        total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    print(f"[gpt2-val] loss: {avg_loss:.4f} (over {num_batches} batches)")
    return avg_loss


if __name__ == "__main__":
    # plot_loss()
    model, tokenizer, config, device = load_inference_model()
    val_loss = evaluate_val_loss(model, num_batches=20, batch_size=10, block_size=1024)
    print(f"[val] loss: {val_loss:.4f} (over {20} batches)")
    
    # evaluate_gpt2_val_loss(num_batches=10, batch_size=20, block_size=1024)
    # prompt = "My name is "
    # output = generate(model, tokenizer, config, device, prompt)
    # print(output)
