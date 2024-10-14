#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for making datasets for the project.
"""

import argparse
import json
import logging
import os
import random
import sys
import time

import tqdm

CNT_LIMIT_DEFAULT = 2**22


def set_randomness_seed(seed=0):
    """Set the randomness seed for reproducibility"""
    random.seed(seed)


def remove_im_start_end_tags(data):
    """Remove the start and end tags from the data"""

    data = str.replace(data, "<|im_start|>", "")
    data = str.replace(data, "<|im_end|>", "")
    return data


def build_completions(
    fp_all, cnt_limit=CNT_LIMIT_DEFAULT, replace_im_start_end_tags=False, verbose=False
):
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

    logging.info("Building completions from %s", fp_all)
    with open(fp_all, "r", encoding="utf-8") as f:
        data = f.readlines()
        logging.info("Loaded %d comments", len(data))

    completions = {}
    cnt = 0
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


def sample_dataset_from_completions(
    completions, max_prompts=CNT_LIMIT_DEFAULT, max_completions=CNT_LIMIT_DEFAULT
):
    """
    Build a dataset from the completions dictionary, sampling
    prompts without replacement, then adding each item for that prompt.

    Args:
        completions: dictionary of the form:
            {
                "prompt": {
                    "chosen": [chosen1, chosen2, ...],
                    "rejected": [rejected1, rejected2, ...]
                },
                ...
            }
        max_prompts: maximum number of prompts to sample
        max_completions: maximum number of chosen, rejected pairs to sample

    Returns:
        dataset: sampled dictionary of the form:
            {
                "prompt": {
                    "chosen": [chosen1, chosen2, ...],
                    "rejected": [rejected1, rejected2, ...]
                },
                ...
            }
    """
    dataset = {}
    prompts = list(completions.keys())
    random.shuffle(prompts)
    logging.info("Shuffled prompts")

    logging.info(f"Sampling prompts until {max_completions} completions hit.")
    cnt = 0
    while True:
        prompt = prompts.pop()
        dataset[prompt] = {
            "chosen": completions[prompt]["chosen"],
            "rejected": completions[prompt]["rejected"],
        }
        cnt += len(completions[prompt]["chosen"])
        if len(dataset) > max_prompts:
            break
        if cnt > max_completions:
            logging.info("Breaking due to max completions hit.")
            break

    logging.info(f"Sampled prompts until {max_completions} completions hit.")
    logging.info(f"Number of prompts: {len(dataset.keys())}")
    logging.info(f"Number of completions: {cnt}")

    return dataset


def sample_datasets_from_completions(
    completions_maj,
    completions_sc,
    max_prompts=CNT_LIMIT_DEFAULT,
    max_completions_maj=CNT_LIMIT_DEFAULT,
):
    """
    Build datasets from the completions dictionary, sampling without replacement.

    Sample same prompts from both completions_maj and completions_sc.
    """

    assert completions_maj.keys() <= completions_sc.keys()

    dataset_maj, dataset_sc = {}, {}
    prompts = list(completions_maj.keys())
    random.shuffle(prompts)
    logging.info("Shuffled prompts")

    logging.info(f"Sampling prompts until {max_completions_maj} completions hit.")
    cnt_maj, cnt_sc = 0, 0
    for prompt in prompts:
        dataset_maj[prompt] = {
            "chosen": completions_maj[prompt]["chosen"],
            "rejected": completions_maj[prompt]["rejected"],
        }
        dataset_sc[prompt] = {
            "chosen": completions_sc[prompt]["chosen"],
            "rejected": completions_sc[prompt]["rejected"],
        }
        cnt_maj += len(completions_maj[prompt]["chosen"])
        cnt_sc += len(completions_sc[prompt]["chosen"])
        if len(dataset_maj) > max_prompts:
            break
        if cnt_maj > max_completions_maj:
            logging.info("Breaking due to max completions hit.")
            break

    logging.info(f"Sampled prompts until {max_completions_maj} completions hit.")
    logging.info(f"Number of prompts: {len(dataset_maj.keys())}")
    logging.info(f"Number of completions_maj: {cnt_maj}")
    logging.info(f"Number of completions_sc: {cnt_sc}")

    return dataset_maj, dataset_sc


def write_out_dataset(
    data,
    out_name="dataset.txt",
    out_base_fp="/home/adam/llm-sct/data/reddit/raw/gpt-3.5-turbo-0125/",
    cnt_limit=CNT_LIMIT_DEFAULT,
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
            items = []
            for prompt in data.keys():
                for chosen, rejected in zip(
                    data[prompt]["chosen"], data[prompt]["rejected"]
                ):
                    item = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
                    if len(items) >= cnt_limit:
                        break
                    items.append(item)

            for i, item in enumerate(items):
                json_str = json.dumps(item, ensure_ascii=False)
                out_line = f"{json_str}"
                if i < len(items) - 1:
                    out_line += ","
                out_line += "\n"
                try:
                    f.write(
                        out_line.encode("utf-16", "surrogatepass").decode(
                            encoding="utf-16"
                        )
                    )
                except UnicodeEncodeError as e:
                    logging.error(f"UnicodeEncodeError on item number {i}")
                    logging.error(f"Error: {e}")
                    unicode_error_cnt += 1
                    continue
                cnt += 1
            f.write("]")
            logging.info("Wrote out %d cnt items to %s", cnt, out_name)
            logging.info("UnicodeEncodeError count: %d", unicode_error_cnt)
        else:
            raise NotImplementedError("Data must be a dictionary")


def main(max_completions_list=None):
    """Main function"""

    logging.info("Starting main method.")
    set_randomness_seed()
    logging.info("Building datasets")

    fp_sc_all = "/home/adam/data/reddit_data_v2/reddit_sc_data_for_DCPO_v2_all.json"
    fp_maj_all = "/home/adam/data/reddit_data_v2/reddit_maj_data_for_DCPO_v2_all.json"

    completions_sc_all = build_completions(fp_sc_all, cnt_limit=2**30)
    completions_maj_all = build_completions(fp_maj_all, cnt_limit=2**30)

    for max_completions in max_completions_list:
        completions_maj_sampled, completions_sc_sampled = (
            sample_datasets_from_completions(
                completions_maj_all,
                completions_sc_all,
                max_completions_maj=max_completions,
            )
        )

        write_out_dataset(
            completions_maj_sampled,
            f"maj_dataset_sampled_{max_completions}.json",
            cnt_limit=max_completions,
        )
        write_out_dataset(
            completions_sc_sampled,
            f"sc_dataset_sampled_{max_completions}.json",
            cnt_limit=max_completions,
        )

    logging.info("Finished main method.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"logs/log_time{time.time()}.log"),
        ],
    )

    parser = argparse.ArgumentParser(description="Build datasets for the project")

    parser.add_argument(
        "--max-completions-list",
        "-mcl",
        nargs="+",
        type=int,
        default=[2400],
        help="List of maximum number of completions to sample",
    )

    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )

    args = parser.parse_args()
    logging.info(args)

    main(max_completions_list=args.max_completions_list)
