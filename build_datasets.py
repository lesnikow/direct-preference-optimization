#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for making datasets for the project.
"""

import argparse
import logging
import os
import sys
import time


def remove_im_start_end_tags(data):
    """Remove the start and end tags from the data"""

    data = str.replace(data, "<|im_start|>", "")
    data = str.replace(data, "<|im_end|>", "")
    return data


def build_completions(fp_all, replace_im_start_end_tags=True):
    """Build the completions dictionary."""

    logging.info(f"Building completions from {fp_all}...")
    with open(fp_all, "r") as f:
        data = f.readlines()
        logging.info(f"Loaded {len(data)}, comments")

    completions = {}
    cnt = 0
    cnt_limit = 4
    for line in data:
        if line.strip() == "[" or line.strip() == "]":
            logging.info("Skipping line")
            continue
        line_dict = eval(line)[0]

        prompt, chosen, rejected = (
            line_dict["prompt"],
            line_dict["chosen"],
            line_dict["rejected"],
        )

        if replace_im_start_end_tags:
            prompt, chosen, rejected = (
                remove_im_start_end_tags(prompt),
                remove_im_start_end_tags(chosen),
                remove_im_start_end_tags(rejected),
            )

        if prompt not in completions:
            completions[prompt] = {"chosen": [], "rejected": []}
        completions[prompt]["chosen"].append(chosen)
        completions[prompt]["rejected"].append(rejected)

        cnt += 1
        if cnt >= cnt_limit:
            break

    for prompt in completions.keys():
        logging.info(f"Prompt is:\n{prompt}")
        logging.info(f"Chosen completions are:\n{completions[prompt]['chosen']}")
        logging.info(f"Rejected completions are:\n{completions[prompt]['rejected']}")

    return completions


def write_out_dataset():
    pass


def main():
    """Main function"""

    logging.info("Building datasets")

    fp_sc_all = "/home/adam/data/reddit_data_v2/reddit_sc_data_for_DCPO_v2_all.json"
    fp_maj_all = "/home/adam/data/reddit_data_v2/reddit_maj_data_for_DCPO_v2_all.json"

    build_completions(fp_sc_all)
    build_completions(fp_maj_all)

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
