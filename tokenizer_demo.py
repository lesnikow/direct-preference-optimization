from transformers import AutoTokenizer

# Load default Pythia tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")

# Example message with special tokens
text = "<|im_start|>user\nHello world<|im_end|>"

# Show default tokenization
tokens = tokenizer.tokenize(text)
print("Default tokenization:")
print(tokens)
print("\nToken IDs:", tokenizer.encode(text))

# Add special tokens to vocabulary
special_tokens = {"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]}
tokenizer.add_special_tokens(special_tokens)

# Show tokenization after adding special tokens
new_tokens = tokenizer.tokenize(text)
print("\nTokenization after adding special tokens:")
print(new_tokens)
print("\nToken IDs:", tokenizer.encode(text))

# Demonstrate handling user/assistant markers
text2 = "<|im_start|>user\nHello world<|im_end|>\n<|im_start|>assistant\nHi there!<|im_end|>"
tokens2 = tokenizer.tokenize(text2)
print("\nFull conversation tokenization:")
print(tokens2)

text3 = "Assistant: Hi there! User: Hello friend!"
tokens3 = tokenizer.tokenize(text3)
print("\nShortened conversation tokenization:")
print(tokens3)
print("\nToken IDs:", tokenizer.encode(text3))
