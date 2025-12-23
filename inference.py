import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model.model import GPT, GPTConfig
from data.loader import get_training_corpus

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-30b", model_max_length=int(1e9))
tokenizer.pad_token = tokenizer.eos_token

# Load model
config = GPTConfig()
model = GPT(config).to(device)

# Load checkpoint weights
ckpt_path = "best_model.pt"
unwanted_prefix = '_orig_mod.'
state_dict = torch.load(ckpt_path, map_location=device)
for key in list(state_dict.keys()):
    if key.startswith(unwanted_prefix):
        # Create the new key name (e.g., "embed.weight")
        new_key = key[len(unwanted_prefix):] 
        # Move the data to the new key
        state_dict[new_key] = state_dict.pop(key)

# 3. Load the fixed state dict
model.load_state_dict(state_dict)
# best_model.pt is just state_dict, ckpt.pt has {"model": state_dict, ...}
if isinstance(state_dict, dict) and "model" in state_dict:
    model.load_state_dict(state_dict["model"])
else:
    model.load_state_dict(state_dict)
model.eval()
print(f"[inference] loaded weights from {ckpt_path}")


@torch.no_grad()
def generate(prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50) -> str:
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
def evaluate_val_loss(num_batches=50, batch_size=8, block_size=1024):
    """Evaluate average validation loss over num_batches."""
    model.eval()
    val_loader = get_training_corpus(batch_size=batch_size, block_size=block_size, train=False)
    
    total_loss = 0.0
    for _ in range(num_batches):
        batch = next(val_loader).to(device)
        x, y = batch[:, :-1], batch[:, 1:]
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    print(f"[val] loss: {avg_loss:.4f} (over {num_batches} batches)")
    return avg_loss


if __name__ == "__main__":
    evaluate_val_loss(num_batches=10, batch_size=8, block_size=1024)
    # prompt = "The answer of 2 + 2 is "
    # output = generate(prompt)
    # print(output)
