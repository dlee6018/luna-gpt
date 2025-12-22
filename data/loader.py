from datasets import load_dataset, interleave_datasets
import warnings

warnings.filterwarnings("ignore", message=".*ArrowInvalid.*")

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
# 6. BATCH SAMPLING
# ==========================================

def get_training_corpus(batch_size=8):
    while True:
        iterator = iter(mixed_dataset)
        batch = []
        for ex in iterator:
            text = ex.get("text")
            if text and 5 < len(text) < 20000:
                batch.append(text)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch


# ==========================================
# 7. DEBUGGING ENTRYPOINT
# ==========================================

if __name__ == "__main__":
    print("Dataset Mix:", PROBABILITIES)

    data_gen = get_training_corpus(batch_size=100)

    sample = next(data_gen)
    print(f"Fetched batch of {len(sample)} records:")
