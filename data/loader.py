import os
# CRITICAL: Must be set before transformers imports to prevent DataLoader deadlocks
# when mixing multi-processing (DataLoader) with multi-threading (Rust Tokenizers)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
import hashlib
import warnings

# Suppress PyArrow/HuggingFace warnings for cleaner output
warnings.filterwarnings("ignore", message=".*ArrowInvalid.*")

# ==========================================
# 0. CONFIG
# ==========================================

PROBABILITIES = {
    "web": 0.45,
    "code": 0.25,
    "math": 0.20,
    "academic": 0.10,
}

VAL_PERCENT = 5 

# ==========================================
# 1. LOADING HELPERS
# ==========================================

def _is_val_sample(text):
    """Deterministically assign sample to val set based on hash."""
    h = int(hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest(), 16)
    return (h % 100) < VAL_PERCENT

def _hash_filter(example, train=True):
    # Safe text extraction
    text = example.get("text") or example.get("content") or example.get("markdown") or ""
    is_val = _is_val_sample(text)
    return not is_val if train else is_val

def load_stream(name, subset=None, train=True):
    ds = load_dataset(name, subset, split="train", streaming=True)
    return ds.filter(lambda ex: _hash_filter(ex, train=train))

# ==========================================
# 2. DATASET MIXING (WORKER AWARE)
# ==========================================

def get_mixed_dataset(train=True, seed=42, current_shard_idx=0, total_shards=1):
    """
    Load and mix datasets. 
    Accepts explicit sharding parameters to handle both Node and Worker splitting.
    """
    
    # Helper to apply the specific shard index calculated by the caller
    def shard_stream(ds):
        if total_shards > 1:
            return ds.shard(num_shards=total_shards, index=current_shard_idx)
        return ds

    # --- WEB (FineWeb-Edu) ---
    ds_web = load_stream("HuggingFaceFW/fineweb-edu", "sample-10BT", train=train)
    ds_web = shard_stream(ds_web).select_columns(["text"])
    
    # --- MATH (FineMath) ---
    ds_math = load_stream("HuggingFaceTB/finemath", "finemath-4plus", train=train)
    ds_math = shard_stream(ds_math).select_columns(["text"])
    
    # --- BOOKS (Gutenberg) ---
    ds_books = load_stream("swiss-ai/apertus-pretrain-gutenberg", train=train)
    ds_books = shard_stream(ds_books).select_columns(["text"])
    
    # --- ARXIV (Academic) ---
    ds_arxiv = load_stream("neuralwork/arxiver", train=train)
    ds_arxiv = shard_stream(ds_arxiv)
    
    def standardize_arxiv(ex):
        # Consolidate column names
        return {"text": ex.get("markdown", "") or ex.get("text", "")}
    
    ds_arxiv = ds_arxiv.map(standardize_arxiv).select_columns(["text"])
    
    # --- CODE (The Stack Smol) ---
    ds_code = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train", streaming=True)
    ds_code = shard_stream(ds_code)
    
    def standardize_code(ex):
        # Consolidate column names
        return {"text": ex.get("content", "") or ex.get("code", "")}
        
    ds_code = ds_code.map(standardize_code).select_columns(["text"])
    ds_code = ds_code.filter(lambda ex: _hash_filter(ex, train=train))

    # --- MIX ACADEMIC ---
    ds_academic = interleave_datasets([ds_arxiv, ds_books], probabilities=[0.5, 0.5], seed=seed)
    
    # --- FINAL MIX ---
    return interleave_datasets(
        [ds_web, ds_code, ds_math, ds_academic],
        probabilities=[PROBABILITIES["web"], PROBABILITIES["code"], PROBABILITIES["math"], PROBABILITIES["academic"]],
        seed=seed,
    )

# ==========================================
# 3. PACKED ITERABLE DATASET
# ==========================================

class PackedIterableDataset(IterableDataset):
    """
    Yields single 1D tensors of size `block_size`. 
    """
    def __init__(self, tokenizer, block_size, train=True, seed=42, rank=0, world_size=1):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.train = train
        self.seed = seed
        self.rank = rank            
        self.world_size = world_size 
    
    def __iter__(self):
        # 1. Calculate Global Worker ID
        worker_info = get_worker_info()
        
        if worker_info is None:
            # Single process loading (Main Process)
            total_shards = self.world_size
            shard_idx = self.rank
            effective_seed = self.seed
        else:
            # Multiprocessing loading (Worker Process)
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            
            # Calculate total global shards needed
            # If 2 GPUs with 2 workers each -> 4 total shards
            total_shards = self.world_size * num_workers
            
            # Calculate unique index for THIS specific worker
            # Rank 0 (Workers 0,1) -> indices 0, 1
            # Rank 1 (Workers 0,1) -> indices 2, 3
            shard_idx = (self.rank * num_workers) + worker_id
            
            # Offset seed to prevent identical shuffling buffers
            effective_seed = self.seed + worker_id

        # 2. Get the stream with explicit sharding
        mixed_ds = get_mixed_dataset(
            train=self.train, 
            seed=effective_seed,
            current_shard_idx=shard_idx,
            total_shards=total_shards
        )
        
        iterator = iter(mixed_ds)
        
        # 3. Packing Logic
        current_seq = []
        eos_id = self.tokenizer.eos_token_id
        
        for ex in iterator:
            text = ex.get("text")
            # Skip empty or weirdly short/long texts
            if not text or not (5 < len(text) < 500000):
                continue
            
            # Use self.tokenizer (safe because it's pickled)
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            if current_seq:
                current_seq.append(eos_id)
            
            # Optimization: If adding tokens doesn't fill the block, just extend and continue
            if len(current_seq) + len(tokens) < self.block_size:
                current_seq.extend(tokens)
                continue
                
            # If we overflow, we have at least one full block
            combined = current_seq + tokens
            
            # Yield full blocks
            for i in range(0, len(combined) - self.block_size + 1, self.block_size):
                chunk = combined[i : i + self.block_size]
                yield torch.tensor(chunk, dtype=torch.long)
            
            # Keep the remainder
            remainder_idx = ((len(combined) // self.block_size) * self.block_size)
            current_seq = combined[remainder_idx:]

# ==========================================
# 4. DATALOADER SETUP
# ==========================================

def get_dataloader(tokenizer, batch_size, block_size, train=True, num_workers=2):
    # Detect rank/world_size locally in the main process
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    # Pass everything to the dataset init
    dataset = PackedIterableDataset(
        tokenizer=tokenizer, 
        block_size=block_size, 
        train=train,
        rank=rank,
        world_size=world_size
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True 
    )

# ==========================================
# 5. ENTRYPOINT
# ==========================================

if __name__ == "__main__":
    # Simulate DDP check
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Warning: Running on {torch.cuda.device_count()} GPUs without 'torchrun'. "
              "This script will default to Rank 0 behavior if run directly with python.")

    print(f"Dataset Mix: {PROBABILITIES}")
    print("Initializing Tokenizer and DataLoader...")

    # 1. Initialize Tokenizer LOCALLY within main
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-30b", model_max_length=int(1e9))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Pass tokenizer explicitly to get_dataloader
    train_loader = get_dataloader(
        tokenizer=tokenizer,
        batch_size=4, 
        block_size=128, 
        train=True, 
        num_workers=2
    )
    
    print("Starting data fetch (this may take a moment to initialize streams)...")
    
    for i, batch in enumerate(train_loader):
        print(f"Batch {i} shape: {batch.shape}")
        
        # Just print the first batch to verify
        if i == 0:
            print(f"First sample start: {batch[0, :10].tolist()}")
            break