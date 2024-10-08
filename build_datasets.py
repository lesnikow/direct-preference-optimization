#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for making datasets for the project
"""

import argparse
import logging
import os
import sys
import time


def build_maj_data():
    """Build the majority datasets."""

    with open(fp_maj_all, "r") as f:
        data_maj = f.readlines()
        logging.info(f"Loaded {len(data_maj)} maj comments")


def main():
    """Main function"""

    logging.info("Building datasets")

    fp_maj_all = "/home/adam/data/reddit_data_v2/reddit_maj_data_for_DCPO_v2_all.json"
    fp_sc_all = "/home/adam/data/reddit_data_v2/reddit_sc_data_for_DCPO_v2_all.json"

    # Load the data, fp is actually a python lis of dictionaries.

    with open(fp_sc_all, "r") as f:
        data_sc = f.readlines()
        logging.info(f"Loaded {len(data_sc)} sc comments")

    # Build dict of completions, grouped by same prompt.
    # Sample dictionary is form:
    # {"prompt": "Prompt text ... ", "chosen": "Chosen completion text... ", "rejected":
    # "rejected completion text..."}

    sc_completions_dict = {}
    cnt = 0
    cnt_limit = 4
    for line in data_sc:
        # logging.info(f"line is {line}")
        if line.strip() == "[" or line.strip() == "]":
            logging.info("Skipping line")
            continue
        line_dict = eval(line)[0]
        # logging.info(f"line_dict is {line_dict}")
        prompt = line_dict["prompt"]
        chosen = line_dict["chosen"]
        rejected = line_dict["rejected"]
        if prompt not in sc_completions_dict:
            sc_completions_dict[prompt] = {"chosen": [], "rejected": []}
        sc_completions_dict[prompt]["chosen"].append(chosen)
        sc_completions_dict[prompt]["rejected"].append(rejected)
        cnt += 1
        if cnt >= cnt_limit:
            break

    logging.info(f"sc_completions_dict is {sc_completions_dict}")
    logging.info("Datasets built")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"log_time{time.time()}.log"),
        ],
    )
    logging.info("Starting main block.")

    parser = argparse.ArgumentParser(description="Build datasets for the project")

    args = parser.parse_args()
    logging.info(args)

    main()
