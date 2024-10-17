#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert models, generate model answers, make judgements and show results."""

import logging
import os
import subprocess
import sys
import time

dpo_exp_dirs = [
    "shp_maj_data_v2_dataset_dpo_loss_pythia69_model_8_batch_size_2024-09-25_16-57-28_215286",
    "shp_sc_data_v2_dataset_dpo_loss_pythia69_model_8_batch_size_2024-09-25_16-59-16_258857",
]

def get_recent_exp_dirs(seconds):
    """Get the experiment directories that were modified in the last seconds seconds.

    We ignore the wandb directory and any dirs without the LATEST directory.
    """

    exp_dirs = []
    for exp_dir in os.listdir(
        os.path.expanduser("~/direct-preference-optimization/.cache/adamlesnikowski/")
    ):
        exp_dir_fullpath = os.path.expanduser(
            f"~/direct-preference-optimization/.cache/adamlesnikowski/{exp_dir}"
        )
        if os.path.getmtime(exp_dir_fullpath) > time.time() - seconds:
            if os.path.exists(f"{exp_dir_fullpath}/LATEST/policy.pt"):
                if exp_dir not in ["wandb"]:
                    exp_dirs.append(exp_dir)
    return exp_dirs


def test_get_recent_exp_dirs():
    """Test get_recent_exp_dirs function."""

    seconds = 60 * 60 * 6
    exp_dirs = get_recent_exp_dirs(seconds)
    assert len(exp_dirs) > 0
    assert ["wandb"] not in exp_dirs

    for exp_dir in exp_dirs:
        logging.info(f"Checking if {exp_dir} exists")
        assert os.path.exists(
            os.path.expanduser(
                f"~/direct-preference-optimization/.cache/adamlesnikowski/{exp_dir}"
            )
        )


def convert_models(dpo_exp_dirs, overwrite=False):
    """Convert models to format suitable for fastchat."""

    for exp_dir in dpo_exp_dirs:
        in_path = os.path.expanduser(
            f"~/direct-preference-optimization/.cache/adamlesnikowski/{exp_dir}/LATEST/policy.pt"
        )

        if (
            os.path.exists(
                os.path.expanduser(
                    f"~/direct-preference-optimization/.cache/adamlesnikowski/{exp_dir}/LATEST/converted/"
                )
            )
            and not overwrite
        ):
            logging.info(f"Converted model already exists for {exp_dir}")
            continue

        subprocess.run(f"du -sh {in_path}", shell=True)
        subprocess.run(f"python3 convert_model.py --in_path {in_path}", shell=True)


def test_convert_models():
    """Test convert_models function."""

    seconds = 60 * 60 * 24
    dpo_exp_dirs = get_recent_exp_dirs(seconds)

    convert_models(dpo_exp_dirs)


def fastchat_setup():
    """Setup fastchat environment."""

    subprocess.run(
        f"source {os.path.expanduser('~/env-fastchat/bin/activate')}",
        shell=True,
        executable="/bin/bash",
    )

    os.chdir(os.path.expanduser("~/fast-chat/fastchat/llm_judge/"))


def generate_model_answers(exp_dir, max_new_tokens):
    num_gpus = 1
    model_path = os.path.expanduser(
        f"~/direct-preference-optimization/.cache/adamlesnikowski/{exp_dir}/LATEST/converted/"
    )
    print(f"Model path: {model_path}")
    subprocess.run(
        [
            "python3",
            "gen_model_answer.py",
            "--model-path",
            model_path,
            "--model-id",
            exp_dir,
            "--num-gpus-total",
            str(num_gpus),
            "--max-new-token",
            str(max_new_tokens),
        ]
    )


def make_fastchat_llm_judge_model_answers():
    max_new_tokens = 128
    for exp_dir in dpo_exp_dirs:
        print(f"Generating model answers for {exp_dir}")
        generate_model_answers(exp_dir, max_new_tokens)


def make_fastchat_llm_judge_model_judgements():
    subprocess.run(
        [
            "python3",
            "gen_judgment.py",
            "--mode",
            "single",
            "--judge-model",
            "gpt-4-turbo",
            "--model-list",
            *dpo_exp_dirs,
            "--parallel",
            "256",
        ]
    )
    subprocess.run(
        [
            "python3",
            "gen_judgment.py",
            "--mode",
            "pairwise-all",
            "--judge-model",
            "gpt-4-turbo",
            "--model-list",
            *dpo_exp_dirs,
            "--parallel",
            "256",
        ]
    )


def show_results():
    with open("out_single.txt", "w") as f:
        subprocess.run(
            [
                "python3",
                "show_result.py",
                "--mode",
                "single",
                "--judge-model",
                "gpt-4-turbo",
                "--model-list",
                *dpo_exp_dirs,
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    with open("out_pw.txt", "w") as f:
        subprocess.run(
            [
                "python3",
                "show_result.py",
                "--mode",
                "pairwise-all",
                "--judge-model",
                "gpt-4-turbo",
                "--model-list",
                *dpo_exp_dirs,
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
        )


def main():
    # convert_models()
    fastchat_setup()
    # make_fastchat_llm_judge_model_answers()
    # make_fastchat_llm_judge_model_judgements()
    show_results()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"logs/log_time{time.time()}.log"),
        ],
    )
    main()
