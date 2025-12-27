from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
import warnings
import torch
import hashlib
from threading import Thread
from queue import Queue

warnings.filterwarnings("ignore", message=".*ArrowInvalid.*")

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-30b", model_max_length=int(1e9))
tokenizer.pad_token = tokenizer.eos_token  # LLaMA doesn't have padding token

PROBABILITIES = {
    "web": 0.45,
    "code": 0.25,
    "math": 0.20,
    "academic": 0.10,
}

# ==========================================
# 1. LOADING HELPERS (STREAMING SAFE)
# ==========================================

VAL_PERCENT = 5  # 5% for validation

def _is_val_sample(text):
    """Deterministically assign sample to val set based on hash (5% val)."""
    h = int(hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest(), 16)
    return (h % 100) < VAL_PERCENT

def _hash_filter(example, train=True):
    """Filter function for train/val split based on content hash."""
    text = example.get("text") or example.get("content") or example.get("markdown") or ""
    is_val = _is_val_sample(text)
    return not is_val if train else is_val

def load_stream(name, subset=None, train=True):
    ds = load_dataset(name, subset, split="train", streaming=True)
    return ds.filter(lambda ex: _hash_filter(ex, train=train))

# ==========================================
# 2. LOAD AND MIX DATASETS
# ==========================================

def get_mixed_dataset(train=True, seed_offset=0):
    """Load and mix all datasets. Uses different seed for train vs val."""
    seed = (42 if train else 1337) + seed_offset
    
    # WEB (FineWeb-Edu)
    ds_web = load_stream("HuggingFaceFW/fineweb-edu", "sample-10BT", train=train)
    ds_web = ds_web.select_columns(["text"])
    
    # MATH (FineMath)
    ds_math = load_stream("HuggingFaceTB/finemath", "finemath-4plus", train=train)
    ds_math = ds_math.select_columns(["text"])
    
    # BOOKS (Gutenberg)
    ds_books = load_stream("swiss-ai/apertus-pretrain-gutenberg", train=train)
    ds_books = ds_books.select_columns(["text"])
    
    # ARXIV (Slimpajama-style academic)
    ds_arxiv = load_stream("neuralwork/arxiver", train=train)
    first_arxiv = next(iter(ds_arxiv))
    if "markdown" in first_arxiv:
        ds_arxiv = load_stream("neuralwork/arxiver", train=train)
        ds_arxiv = ds_arxiv.select_columns(["markdown"]).rename_column("markdown", "text")
    else:
        ds_arxiv = load_stream("neuralwork/arxiver", train=train)
        ds_arxiv = ds_arxiv.select_columns(["text"])
    
    # CODE (The Stack Smol - Python subset)
    ds_code = load_dataset(
        "bigcode/the-stack-smol",
        data_dir="data/python",
        split="train",
        streaming=True,
    )
    first_code = next(iter(ds_code))
    if "content" in first_code:
        code_col = "content"
    elif "code" in first_code:
        code_col = "code"
    else:
        raise ValueError(f"[ERROR] Unknown code dataset column. Keys found: {first_code.keys()}")
    ds_code = load_dataset(
        "bigcode/the-stack-smol",
        data_dir="data/python",
        split="train",
        streaming=True,
    )
    ds_code = ds_code.select_columns([code_col]).rename_column(code_col, "text")
    ds_code = ds_code.filter(lambda ex: _hash_filter(ex, train=train))
    
    # MIX ACADEMIC (Arxiv + Books)
    ds_academic = interleave_datasets(
        [ds_arxiv, ds_books],
        probabilities=[0.5, 0.5],
        seed=seed,
    )
    
    # MIX ALL DATASETS
    datasets_list = [ds_web, ds_code, ds_math, ds_academic]
    probs_list = [
        PROBABILITIES["web"],
        PROBABILITIES["code"],
        PROBABILITIES["math"],
        PROBABILITIES["academic"],
    ]
    
    return interleave_datasets(
        datasets_list,
        probabilities=probs_list,
        seed=seed,
    )

# ==========================================
# 6. BATCH SAMPLING WITH SEQUENCE PACKING
# ==========================================

def _batch_producer(queue, batch_size, block_size, state, train=True):
    """Background thread that tokenizes and produces batches."""
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    
    while True:
        epoch = state["epoch"]
        mixed_dataset = get_mixed_dataset(train=train, seed_offset=epoch)
        iterator = iter(mixed_dataset)
        batch = []
        current_seq = []
        
        for ex in iterator:
            text = ex.get("text")
            if not text or not (5 < len(text) < 20000):
                continue
            

            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            if current_seq:
                current_seq.append(eos_token_id)
            
            remaining_space = block_size - len(current_seq)
            
            if len(tokens) <= remaining_space:
                current_seq.extend(tokens)
            else:
                if remaining_space > 0 and current_seq:
                    current_seq.extend(tokens[:remaining_space])
                
                if len(current_seq) >= block_size:
                    batch.append(current_seq[:block_size])
                    if len(batch) >= batch_size:
                        queue.put(torch.tensor(batch, dtype=torch.long))
                        batch = []
                    current_seq = []
                
                tokens = tokens[remaining_space:] if remaining_space > 0 else tokens
                
                while len(tokens) >= block_size:
                    batch.append(tokens[:block_size])
                    if len(batch) >= batch_size:
                        queue.put(torch.tensor(batch, dtype=torch.long))
                        batch = []
                    tokens = tokens[block_size:]
                
                current_seq = tokens
        
        if current_seq:
            if len(current_seq) < block_size:
                current_seq = current_seq + [pad_token_id] * (block_size - len(current_seq))
            batch.append(current_seq[:block_size])
        
        if batch:
            while len(batch) < batch_size:
                batch.append([pad_token_id] * block_size)
            queue.put(torch.tensor(batch, dtype=torch.long))
        
        state["epoch"] += 1
        print(f"[data] epoch {epoch} complete, starting epoch {state['epoch']}", flush=True)


def get_training_corpus(batch_size=8, block_size=1024, prefetch_batches=4, train=True, start_epoch=0, is_tokenized=True):
    """
    Returns (generator, state) where generator yields pre-tokenized batch tensors.
    state["epoch"] can be read to get the current epoch.
    
    Uses a background thread to tokenize ahead, so GPU doesn't wait for CPU.
    prefetch_batches: number of batches to buffer ahead (default 4).
    start_epoch: epoch number to start from (for resuming training).
    is_tokenized: if True, return tokenized tensors; if False, return plain text strings. 1 example
    """
    state = {"epoch": start_epoch}
    
    if not is_tokenized:
        # Return plain text strings without tokenization
        def text_generator():
            while True:
                epoch = state["epoch"]
                mixed_dataset = get_mixed_dataset(train=train, seed_offset=epoch)
                for ex in mixed_dataset:
                    text = ex.get("text")
                    if text and 5 < len(text) < 20000:
                        yield text
                state["epoch"] += 1
        return text_generator(), state
    
    queue = Queue(maxsize=prefetch_batches)
    
    # Start background producer thread
    producer = Thread(
        target=_batch_producer,
        args=(queue, batch_size, block_size, state, train),
        daemon=True,
    )
    producer.start()
    
    def generator():
        while True:
            yield queue.get()
    
    return generator(), state


# ==========================================
# 7. DEBUGGING ENTRYPOINT
# ==========================================

if __name__ == "__main__":
    print("Dataset Mix:", PROBABILITIES)

    data_gen, state = get_training_corpus(batch_size=4, block_size=1024, train=True)

    sample = next(data_gen)
    
    print(f"Fetched batch shape: {sample.shape}, epoch: {state['epoch']}")
    print(f"Sample tokens (first 50): {sample[0, :50].tolist()}")
