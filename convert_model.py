import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GPTNeoXForCausalLM,
)

def main():
    in_path = f"/root/policy_dcpo.pt/policy.pt"
    out_path = f"/root/policy_dcpo/"

    state_dict = torch.load(in_path)
    print(state_dict.keys())

    model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-2.8b")
    model.load_state_dict(state_dict["state"])
    model.save_pretrained(out_path)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")
    config = AutoConfig.from_pretrained("EleutherAI/pythia-2.8b")

    tokenizer.save_pretrained(out_path)
    config.save_pretrained(out_path)


def test():
    model = GPTNeoXForCausalLM.from_pretrained(out_path)
    tokenizer = AutoTokenizer.from_pretrained(out_path)

    inputs = tokenizer("Hello, world!", return_tensors="pt")
    outputs = model(**inputs)
    print(outputs)

if __name__ == "__main__":
    main()
