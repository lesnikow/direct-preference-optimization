import torch
from transformers import AutoModel
from transformers import AutoTokenizer, AutoConfig
from transformers import GPTNeoXForCausalLM

# Define the path to the input model and the output file
exp_dir = ""
in_path = f"/root/policy_dcpo.pt/policy.pt"
out_path = f"/root/policy_dcpo/"

state_dict = torch.load(in_path)
model = GPTNeoXForCausalLM.from_pretrained('EleutherAI/pythia-2.8b')
model.load_state_dict(state_dict)
model.save_pretrained(out_path)

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-2.8b')
config = AutoConfig.from_pretrained('EleutherAI/pythia-2.8b')

tokenizer.save_pretrained(out_path)
config.save_pretrained(out_path)



# Test 
model = AutoModel.from_pretrained(out_path)
tokenizer = AutoTokenizer.from_pretrained(out_path)

inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model(**inputs)

print(outputs)

