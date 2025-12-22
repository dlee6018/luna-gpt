from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-30b")
tokenizer.pad_token = tokenizer.eos_token # LLAMA doesn't have padding token

# text = "My name is David"
# tokens = tokenizer.encode(text)
# print(tokens)
# print(tokenizer.decode(tokens))
# print("Vocab size:", tokenizer.vocab_size)  # ~ 32,000