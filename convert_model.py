import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GPTNeoXForCausalLM,
)

in_path = f"/root/policy_dcpo.pt/policy.pt"
out_path = f"/root/policy_dcpo/"

# Load the state dict
state_dict = torch.load(in_path)

print(state_dict.keys())
# print(state_dict['state'].keys())

# Load the model
model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-2.8b")

# Update the model's state dict
model.load_state_dict(state_dict["state"])

# print("Keys in the model's state_dict:")
# print(model.state_dict().keys())

# Save the model
model.save_pretrained(out_path)

# Load and save tokenizer and config
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")
config = AutoConfig.from_pretrained("EleutherAI/pythia-2.8b")

tokenizer.save_pretrained(out_path)
config.save_pretrained(out_path)


# Test
model = GPTNeoXForCausalLM.from_pretrained(out_path)
tokenizer = AutoTokenizer.from_pretrained(out_path)

inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model(**inputs)

print(outputs)
