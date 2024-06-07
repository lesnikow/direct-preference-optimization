import argparse
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GPTNeoXForCausalLM,
)


def main(in_path):
    """Main method."""

    out_path = os.path.join(os.path.dirname(in_path), "converted")
    print(f"Saving model .bin, tokenizer, config json files to {out_path}")
    os.makedirs(out_path, exist_ok=True)

    state_dict = torch.load(in_path)
    print(state_dict.keys())

    model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-2.8b")
    model.load_state_dict(state_dict["state"])
    model.save_pretrained(out_path)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")
    config = AutoConfig.from_pretrained("EleutherAI/pythia-2.8b")
    tokenizer.save_pretrained(out_path)
    config.save_pretrained(out_path)


def test(out_path):
    """Test our main method outputs."""
    model = GPTNeoXForCausalLM.from_pretrained(out_path)
    tokenizer = AutoTokenizer.from_pretrained(out_path)

    inputs = tokenizer("Hello, world!", return_tensors="pt")
    outputs = model(**inputs)
    print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load and save a model with a given state dictionary."
    )
    parser.add_argument(
        "--in_path",
        type=str,
        required=True,
        help="""Path to the input state dictionary file, e.g. 
            /root/dpo/.cache/root/hb_dataset_dpo_loss_pythia28_2024-06-07_18-19-17_935174/LATEST/policy.pt
            """,
    )

    args = parser.parse_args()
    main(args.in_path)
