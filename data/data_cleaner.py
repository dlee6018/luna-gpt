from loader import get_training_corpus
import hashlib

datagen, state = get_training_corpus(batch_size=4, block_size=1024, train=True, start_epoch=0, is_tokenized=False)

sample = next(datagen)
# print(sample)

def normalize(text):
    return " ".join(text.lower().split())

def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()

print(normalize(sample))