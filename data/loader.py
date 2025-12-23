from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
import warnings
import torch
from torch.utils.data import IterableDataset, DataLoader
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

def load_stream(name, subset=None, split="train"):
    return load_dataset(name, subset, split=split, streaming=True)

# ==========================================
# 2. LOAD RAW DATASETS
# ==========================================

# WEB (FineWeb-Edu)
ds_web = load_stream("HuggingFaceFW/fineweb-edu", "sample-10BT")

# MATH (FineMath)
ds_math = load_stream("HuggingFaceTB/finemath", "finemath-4plus")

# BOOKS (Gutenberg)
ds_books = load_stream("swiss-ai/apertus-pretrain-gutenberg")

# ARXIV (Slimpajama-style academic)
ds_arxiv = load_stream("neuralwork/arxiver")

# CODE (The Stack Smol - Python subset)
ds_code = load_dataset(
    "bigcode/the-stack-smol",
    data_dir="data/python",
    split="train",
    streaming=True,
)

# ==========================================
# 3. NORMALIZE ALL TO "text"
# ==========================================

# WEB
ds_web = ds_web.select_columns(["text"])

# MATH
ds_math = ds_math.select_columns(["text"])

# BOOKS
ds_books = ds_books.select_columns(["text"])

# ARXIV: some versions use "markdown", others use "text"
first_arxiv = next(iter(ds_arxiv))
if "markdown" in first_arxiv:
    ds_arxiv = load_stream("neuralwork/arxiver")  # reload iterator
    ds_arxiv = ds_arxiv.select_columns(["markdown"]).rename_column("markdown", "text")
else:
    ds_arxiv = load_stream("neuralwork/arxiver")  # reload iterator
    ds_arxiv = ds_arxiv.select_columns(["text"])

# CODE: features=None → must inspect sample content
first_code = next(iter(ds_code))
if "content" in first_code:
    code_col = "content"
elif "code" in first_code:
    code_col = "code"
else:
    raise ValueError(f"[ERROR] Unknown code dataset column. Keys found: {first_code.keys()}")

# reload iterator to avoid skipping first element
ds_code = load_dataset(
    "bigcode/the-stack-smol",
    data_dir="data/python",
    split="train",
    streaming=True,
)
ds_code = ds_code.select_columns([code_col]).rename_column(code_col, "text")

# ==========================================
# 4. MIX ACADEMIC (Arxiv + Books)
# ==========================================

ds_academic = interleave_datasets(
    [ds_arxiv, ds_books],
    probabilities=[0.5, 0.5],
    seed=42,
)

# ==========================================
# 5. MIX ALL DATASETS
# ==========================================

datasets_list = [ds_web, ds_code, ds_math, ds_academic]
probs_list = [
    PROBABILITIES["web"],
    PROBABILITIES["code"],
    PROBABILITIES["math"],
    PROBABILITIES["academic"],
]

mixed_dataset = interleave_datasets(
    datasets_list,
    probabilities=probs_list,
    seed=42,
)

# ==========================================
# 6. BATCH SAMPLING WITH SEQUENCE PACKING
# ==========================================

def _batch_producer(queue, batch_size, block_size):
    """Background thread that tokenizes and produces batches."""
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    
    while True:
        iterator = iter(mixed_dataset)
        batch = []
        current_seq = []
        
        for ex in iterator:
            text = ex.get("text")
            if not text or not (5 < len(text) < 20000):
                continue
            
            # Tokenize document (happens in background thread)
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


def get_training_corpus(batch_size=8, block_size=1024, prefetch_batches=4):
    """
    Yields pre-tokenized, packed batches as tensors.
    
    Uses a background thread to tokenize ahead, so GPU doesn't wait for CPU.
    prefetch_batches: number of batches to buffer ahead (default 4).
    """
    queue = Queue(maxsize=prefetch_batches)
    
    # Start background producer thread
    producer = Thread(
        target=_batch_producer,
        args=(queue, batch_size, block_size),
        daemon=True,
    )
    producer.start()
    
    # Yield from queue (blocks if empty, producer fills it)
    while True:
        yield queue.get()


# ==========================================
# 7. DEBUGGING ENTRYPOINT
# ==========================================

if __name__ == "__main__":
    print("Dataset Mix:", PROBABILITIES)

    data_gen = get_training_corpus(batch_size=4, block_size=1024)

    sample = next(data_gen)
    
    print(f"Fetched batch shape: {sample.shape}")
    print(f"Sample tokens (first 50): {sample[0, :50].tolist()}")
