#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert a model from a state dictionary to a model, tokenizer, and config."""

import argparse
import logging
import os
import sys
import time

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
    logging.info(f"Starting conversion of model from {in_path} to {out_path}...")
    os.makedirs(out_path, exist_ok=True)

    logging.info(f"Loading state dictionary from {in_path}")
    state_dict = torch.load(in_path)

    logging.info("Loading tokenizer and config")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b")
    config = AutoConfig.from_pretrained("EleutherAI/pythia-6.9b")

    logging.info("Loading model from state dictionary")
    model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-6.9b")
    model.load_state_dict(state_dict["state"])

    logging.info("Saving tokenizer and config")
    tokenizer.save_pretrained(out_path)
    config.save_pretrained(out_path)

    logging.info("Saving model")
    model.save_pretrained(out_path)


def test_main(out_path):
    """Test our main method outputs."""

    model = GPTNeoXForCausalLM.from_pretrained(out_path)
    tokenizer = AutoTokenizer.from_pretrained(out_path)

    inputs = tokenizer("Hello, world!", return_tensors="pt")
    outputs = model(**inputs)
    print(outputs)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"logs/log_time{time.time()}.log"),
        ],
    )

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
