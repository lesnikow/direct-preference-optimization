#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for making datasets for the project.
"""

import argparse
import json
import logging
import os
import sys
import time

import tqdm


def remove_im_start_end_tags(data):
    """Remove the start and end tags from the data"""

    data = str.replace(data, "<|im_start|>", "")
    data = str.replace(data, "<|im_end|>", "")
    return data


def build_completions(fp_all, replace_im_start_end_tags=False, verbose=False):
    """
    Build the completions dictionary from the fp_all filepath. This completion
    dictionary is of the form:
    {
        "prompt": {
            "chosen": [chosen1, chosen2, ...],
            "rejected": [rejected1, rejected2, ...]
        },
        ...
    }

    Args:
        fp_all: filepath to the source data file
        replace_im_start_end_tags: whether to replace the start and end tags
        verbose: whether to be verbose

    Returns:
        completions: dictionary of the form:
            {
                "prompt": {
                    "chosen": [chosen1, chosen2, ...],
                    "rejected": [rejected1, rejected2, ...]
                },
                ...
    """

    logging.info(f"Building completions from {fp_all}...")
    with open(fp_all, "r") as f:
        data = f.readlines()
        logging.info(f"Loaded {len(data)}, comments")

    completions = {}
    cnt = 0
    cnt_limit = 2**10
    prompts = 0
    for line in tqdm.tqdm(data):

        if line.strip() == "[" or line.strip() == "]":
            logging.info(f"Skipping line: {line.strip()}")
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
            prompts += 1
        completions[prompt]["chosen"].append(chosen)
        completions[prompt]["rejected"].append(rejected)

        cnt += 1
        if cnt >= cnt_limit:
            break

    assert len(completions) == prompts

    if verbose:
        for prompt in completions.keys():
            logging.info(f"Prompt is:\n{prompt}")
            logging.info(f"Chosen completions are:\n{completions[prompt]['chosen']}")
            logging.info(
                f"Rejected completions are:\n{completions[prompt]['rejected']}"
            )

    logging.info(f"Built completions from {fp_all}")
    logging.info(f"Number of prompts: {prompts}")
    logging.info(f"Number of completions: {cnt}")

    return completions


def write_out_dataset(
    data,
    out_name="dataset.txt",
    out_base_fp="/home/adam/llm-sct/data/reddit/raw/gpt-3.5-turbo-0125/",
    cnt_limit=2**20,
):
    """
    Write out the dataset as a text file containing a Python list of dictionaries of the
    form: [
        {"prompt": prompt, "chosen": chosen, "rejected": rejected},
        ...
    ]

    The data is written out in utf-16 encoding to avoid UnicodeEncodeErrors.

    Args:
        data: dictionary of the form:
            {
                "prompt": {
                    "chosen": [chosen1, chosen2, ...],
                    "rejected": [rejected1, rejected2, ...]
                },
                ...
            }
        out_name: name of the output file
        out_base_fp: base filepath to write the output file
        cnt_limit: limit on the number of items to write out

    Returns:
        None

    Side effects:
        Writes out the data to the output file.
    """
    cnt = 0
    unicode_error_cnt = 0
    with open(os.path.join(out_base_fp, out_name), "w", encoding="utf-16") as f:
        if isinstance(data, dict):
            f.write("[\n")
            for prompt in data.keys():
                for chosen, rejected in zip(
                    data[prompt]["chosen"], data[prompt]["rejected"]
                ):
                    item = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
                    json_str = json.dumps(item, ensure_ascii=False)
                    out_line = f"{json_str},\n"

                    try:
                        f.write(
                            out_line.encode("utf-16", "surrogatepass").decode(
                                encoding="utf-16"
                            )
                        )
                    except UnicodeEncodeError as e:
                        logging.error(f"UnicodeEncodeError on line number {cnt}")
                        logging.error(f"Error: {e}")
                        unicode_error_cnt += 1
                        continue

                    cnt += 1
                    if cnt >= cnt_limit:
                        break
            f.write("]")
            logging.info(f"Wrote out {cnt} items to {out_name}")
            logging.info(f"UnicodeEncodeError count: {unicode_error_cnt}")

        else:
            raise NotImplementedError("Data must be a dictionary")


def main():
    """Main function"""

    logging.info("Building datasets")

    fp_sc_all = "/home/adam/data/reddit_data_v2/reddit_sc_data_for_DCPO_v2_all.json"
    fp_maj_all = "/home/adam/data/reddit_data_v2/reddit_maj_data_for_DCPO_v2_all.json"

    completions_sc_all = build_completions(fp_sc_all)
    write_out_dataset(completions_sc_all, "sc_dataset.json")
    completions_maj_all = build_completions(fp_maj_all)
    write_out_dataset(completions_maj_all, "maj_dataset.json")

    logging.info("Datasets built")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"logs/log_time{time.time()}.log"),
        ],
    )
    logging.info("Starting main block.")

    parser = argparse.ArgumentParser(description="Build datasets for the project")

    args = parser.parse_args()
    logging.info(args)

    main()
