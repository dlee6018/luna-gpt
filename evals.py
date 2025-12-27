from datasets import load_dataset
from torch.nn import functional as F
from inference import load_inference_model
import torch

@torch.no_grad()
def evaluate_hellaswag(num_samples=100):
    ds = load_dataset("Rowan/hellaswag", split="validation")
    model, tokenizer, config, device = load_inference_model(ckpt_path="best_model.pt")
    
    correct = 0
    for i in range(num_samples):
        sample = ds[i]
        ctx = sample["ctx"]
        endings = sample["endings"]
        label = int(sample["label"])
        
        # Score each ending by computing log-likelihood
        scores = []
        for ending in endings:
            text = ctx + " " + ending
            tokens = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)
            logits = model(tokens)  # [1, seq_len, vocab_size]
            # Compute loss for the ending portion only
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = tokens[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                                   shift_labels.view(-1), reduction='mean') # compare token i to i + 1
            scores.append(-loss.item())  # Higher score = lower loss = better
        
        pred = scores.index(max(scores))
        if pred == label:
            correct += 1
    
    accuracy = correct / num_samples
    print(f"HellaSwag accuracy: {accuracy:.2%}")
    return accuracy


if __name__ == "__main__":
    evaluate_hellaswag()