from datasets import load_dataset


dataset = load_dataset("openai/summarize_from_feedback", "comparisons")
train_dataset = dataset["train"]

print(train_dataset)
print(train_dataset[0])
print(len(train_dataset))


dataset = load_dataset("reddit", "tldr")
print(dataset)
print(dataset.keys())
print(len(dataset))

train_dataset = dataset["train"]
print(train_dataset)
print(train_dataset[0])
print(len(train_dataset))

print("Document:", train_dataset[0]["document"])
print("Summary:", train_dataset[0]["summary"])


dataset = load_dataset("imdb")

print(dataset)

train_dataset = dataset["train"]
test_dataset = dataset["test"]

print(train_dataset[0]["text"])
print(train_dataset[0]["label"])

print(test_dataset[0]["text"])
print(test_dataset[0]["label"])




