from loader import get_training_corpus
from datasketch import MinHash, MinHashLSH
from multiprocessing import Pool
import hashlib

datagen, state = get_training_corpus(batch_size=4, block_size=1024, train=True, start_epoch=0, is_tokenized=False)

def normalize(text):
    return " ".join(text.lower().split())

def hash_text(text):
    return hashlib.sha256(text.encode()).digest()  # smaller + faster

def minhash_text(text, n=5, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for i in range(len(text) - n + 1):
        m.update(text[i:i+n].encode("utf-8"))
    return m


BATCH_SIZE = 1000
NUM_SAMPLES = 100000

seen_hashes = set()
lsh = MinHashLSH(threshold=0.9, num_perm=128)
minhash_store = {}

near_dupes = 0
global_idx = 0

with Pool(10) as p:
    for batch_start in range(0, NUM_SAMPLES, BATCH_SIZE):
        batch = [normalize(next(datagen)) for _ in range(min(BATCH_SIZE, NUM_SAMPLES - batch_start))]

        # exact hashes
        hashes = p.map(hash_text, batch)

        # MinHash only for non-exact-dupes
        unique_texts = []
        unique_ids = []

        for text, h in zip(batch, hashes):
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            unique_texts.append(text)
            unique_ids.append(global_idx)
            global_idx += 1

        minhashed = p.map(minhash_text, unique_texts)

        for doc_id, mh in zip(unique_ids, minhashed):
            matches = lsh.query(mh)
            if matches:
                near_dupes += 1
            else:
                lsh.insert(str(doc_id), mh)
                minhash_store[doc_id] = mh

print(f"Exact unique docs: {len(seen_hashes)}")
print(f"Near-duplicate docs: {near_dupes}")