#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert models, generate model answers, make judgements and show results."""

import logging
import os
import subprocess
import sys
import time


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

        if os.path.exists(
            os.path.expanduser(
                f"~/direct-preference-optimization/.cache/adamlesnikowski/{exp_dir}/LATEST/converted/"
            )
        ):
            if not overwrite:
                logging.info(f"Model already converted for {exp_dir}")
                logging.info(f"Skipping")
                continue
            elif overwrite:
                logging.info(f"Overwriting converted model for {exp_dir}")

        venv_python = os.path.join(os.path.expanduser("~/env"), "bin", "python3")

        subprocess.run(
            [
                venv_python,
                "convert_model.py",
                "--in_path",
                in_path,
            ]
        )


def test_convert_models():
    """Test convert_models function."""

    seconds = 60 * 60 * 12
    dpo_exp_dirs = get_recent_exp_dirs(seconds)
    dpo_exp_dirs = [
        "maj_shp_data_v3_matched_prompts_16_dataset_dpo_loss_pythia69_model_8_batch_size_2024-10-17_01-17-47_748858",
        "maj_shp_data_v3_matched_prompts_1000_dataset_dpo_loss_pythia69_model_8_batch_size_2024-10-17_01-05-51_987880",
        "maj_shp_data_v3_matched_prompts_32000_dataset_dpo_loss_pythia69_model_8_batch_size_2024-10-16_06-28-57_000634",
    ]

    convert_models(dpo_exp_dirs, overwrite=True)


def fastchat_setup():
    """Setup fastchat environment."""

    os.chdir(os.path.expanduser("~/fast-chat/fastchat/llm_judge/"))


def generate_model_answers(exp_dir, max_new_tokens, overwrite=False):
    """Generate model answers for a given model."""

    output_path = os.path.expanduser(
        f"~/fast-chat/fastchat/llm_judge/data/mt_bench/model_answer/{exp_dir}.jsonl"
    )
    logging.info(f"Output path for model answers: {output_path}")

    if os.path.exists(output_path):
        if not overwrite:
            logging.info(f"Model answers already exist for {exp_dir}")
            logging.info(f"Skipping")
            return
        else:
            logging.info(
                f"Model answers exist, but overwriting model answers for {exp_dir}"
            )

    logging.info(f"Generating model answers")
    num_gpus = 1
    model_path = os.path.expanduser(
        f"~/direct-preference-optimization/.cache/adamlesnikowski/{exp_dir}/LATEST/converted/"
    )

    venv_python = os.path.join(os.path.expanduser("~/env-fastchat"), "bin", "python3")
    subprocess.run(
        [
            venv_python,
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


def make_fastchat_llm_judge_model_answers(dpo_exp_dirs):
    """Make fastchat llm judge model answers."""

    max_new_tokens = 128
    for exp_dir in dpo_exp_dirs:
        print(f"Generating model answers for {exp_dir}")
        generate_model_answers(exp_dir, max_new_tokens)


def test_make_fastchat_llm_judge_model_answers():
    """Test make_fastchat_llm_judge_model_answers function."""

    fastchat_setup()

    dpo_exp_dirs = [
        "maj_shp_data_v3_matched_prompts_1000_dataset_dpo_loss_pythia69_model_8_batch_size_2024-10-17_01-05-51_987880",
        "shp_sc_data_v2_40k_rump_cut_dataset_dpo_loss_pythia69_model_8_batch_size_2024-10-02_12-13-11_651914",
        "vicuna-7b-v1.3",
    ]

    make_fastchat_llm_judge_model_answers(dpo_exp_dirs)


def make_fastchat_llm_judge_model_judgements():
    """Make fastchat llm judge model judgements."""

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


def show_results(dpo_exp_dirs):
    """Show results."""

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
    """Main method."""

    dpo_exp_dirs = get_recent_exp_dirs(60 * 60 * 24 * 5)

    show_results(dpo_exp_dirs)
    # convert_models(dpo_exp_dirs)
    # fastchat_setup()
    # make_fastchat_llm_judge_model_answers()
    # make_fastchat_llm_judge_model_judgements()


def test_main():
    """Test main method."""

    result = main()
    assert result is None


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
