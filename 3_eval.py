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
    """Setup for fastchat modules."""

    os.chdir(os.path.expanduser("~/fast-chat/fastchat/llm_judge/"))


def make_answers_single_model(model_path, max_new_tokens, overwrite=False):
    """Generate model answers for a given model."""

    if "LATEST" not in model_path:
        model_id = "_".join(model_path.split("/")[-2:])
    else:
        model_id = "_".join(model_path.split("/")[-3:])

    output_path = os.path.expanduser(
        f"~/fast-chat/fastchat/llm_judge/data/mt_bench/model_answer/{model_id}.jsonl"
    )
    logging.info(f"Output path for model answers: {output_path}")

    if os.path.exists(output_path):
        if not overwrite:
            logging.info(f"Model answers already exist for {model_id}, skipping")
            return
        else:
            logging.info(
                f"Model answers exist, but overwriting model answers for {model_id}"
            )

    logging.info(f"Generating model answers for {model_id}")
    num_gpus = 1
    venv_python = os.path.join(os.path.expanduser("~/env-fastchat"), "bin", "python3")
    subprocess.run(
        [
            venv_python,
            "gen_model_answer.py",
            "--model-path",
            model_path,
            "--model-id",
            model_id,
            "--num-gpus-total",
            str(num_gpus),
            "--max-new-token",
            str(max_new_tokens),
        ]
    )


def make_answers(dpo_exp_dirs, run_intermediate_models=False, reverse=False):
    """Make fastchat llm judge model answers."""

    model_paths = []
    max_new_tokens = 128
    base_path = os.path.expanduser(
        "~/direct-preference-optimization/.cache/adamlesnikowski"
    )
    for exp_dir in dpo_exp_dirs:
        if run_intermediate_models:
            exp_path = os.path.join(base_path, exp_dir)
            intermediate_models = os.listdir(exp_path)
            intermediate_models.remove("config.yaml")
            intermediate_models.remove("LATEST")

            model_paths_to_add = [
                os.path.join(exp_path, model) for model in intermediate_models
            ]
            model_paths.extend(model_paths_to_add)
        else:
            model_paths.append(os.path.join(base_path, exp_dir, "LATEST/converted"))

        if reverse:
            model_paths = reversed(model_paths)
        logging.info(f"Model paths: {model_paths}")
        for model_path in model_paths:
            logging.info(f"Model path: {model_path}")
            if not os.path.exists(model_path):
                logging.info(f"Model path does not exist: {model_path}")
                continue
            make_answers_single_model(model_path, max_new_tokens)


def test_make_answers():
    """Test make_answers function."""

    fastchat_setup()

    dpo_exp_dirs = [
        "maj_shp_data_v3_matched_prompts_1000_dataset_dpo_loss_pythia69_model_8_batch_size_2024-10-17_01-05-51_987880",
        "shp_sc_data_v2_40k_rump_cut_dataset_dpo_loss_pythia69_model_8_batch_size_2024-10-02_12-13-11_651914",
        "vicuna-7b-v1.3",
    ]

    make_answers(dpo_exp_dirs)


def make_judgements_for_mode(mode, dpo_exp_dirs, baseline_model="pythia-2.8b"):
    """Make fastchat llm judge model judgements for the given mode.

    Available modes: single, pairwise-all, pairwise-baseline.

    baseline_model is the model to compare against in pairwise-baseline mode.
    """

    venv_python = os.path.join(os.path.expanduser("~/env-fastchat"), "bin", "python3")
    logging.info("Making judgements for mode: %s", mode)

    if mode in ["single", "pairwise-all"]:
        subprocess.run(
            [
                venv_python,
                "gen_judgment.py",
                "--mode",
                mode,
                "--judge-model",
                "gpt-4-turbo",
                "--model-list",
                *dpo_exp_dirs,
                "--parallel",
                "256",
            ]
        )
    elif mode == "pairwise-baseline":
        subprocess.run(
            [
                venv_python,
                "gen_judgment.py",
                "--mode",
                mode,
                "--judge-model",
                "gpt-4-turbo",
                "--model-list",
                *dpo_exp_dirs,
                "--parallel",
                "256",
                "--baseline-model",
                baseline_model,
            ]
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")


def show_results_for_mode(mode, dpo_exp_dirs):
    """Show results for a given mode and dpo_exp_dirs."""

    venv_python = os.path.join(os.path.expanduser("~/env-fastchat"), "bin", "python3")
    os.chdir(os.path.expanduser("~/fast-chat/fastchat/llm_judge/"))

    if mode in ["single", "pairwise-all"]:
        subprocess.run(
            [
                venv_python,
                os.path.expanduser("~/fast-chat/fastchat/llm_judge/show_result.py"),
                "--mode",
                mode,
                "--judge-model",
                "gpt-4-turbo",
                "--model-list",
                *dpo_exp_dirs,
            ]
        )
    elif mode == "pairwise-baseline":
        subprocess.run(
            [
                venv_python,
                os.path.expanduser("~/fast-chat/fastchat/llm_judge/show_result.py"),
                "--mode",
                mode,
                "--judge-model",
                "gpt-4-turbo",
                "--model-list",
                *dpo_exp_dirs,
                "--baseline-model",
                "pythia-6.9b",
            ]
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")


def show_results(dpo_exp_dirs):
    """Show results."""

    for mode in ["single", "pairwise-all"]:
        show_results_for_mode(mode, dpo_exp_dirs)


def main():
    """Main method."""

    logging.info("Starting main method")
    # dpo_exp_dirs = get_recent_exp_dirs(60 * 60 * 24 * 14)
    dpo_exp_dirs = [
        "sft_condorcet_all_dataset_sft_loss_pythia28_model_32_batch_size_2024-11-26_20-38-29_179432"
    ]
    logging.info(f"Recent experiment directories: {dpo_exp_dirs}")

    # convert_models(dpo_exp_dirs)
    fastchat_setup()
    make_answers(dpo_exp_dirs, run_intermediate_models=True)

    make_judgements_mode = "pairwise-baseline"
    # make_judgements_for_mode(
    #    make_judgements_mode, dpo_exp_dirs, baseline_model="pythia-2.8b"
    # )

    show_results_mode = "pairwise-baseline"
    # show_results_for_mode(show_results_mode, dpo_exp_dirs)


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
