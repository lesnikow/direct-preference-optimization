#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script is used to run direct preference optimization (DPO) experiments for the
a-arm and b-arm datasets.
"""

import subprocess
import sys

# Direct preference optimization (DPO) experiments
a_arm_dataset = ""
b_arm_dataset = ""
n_arm_dataset = ""
model = "pythia69"
model_fsdp_policy_mp = "bfloat16"
loss = "dpo"
loss_beta = 0.1
batch_size = 8
gradient_accumulation_steps = 1
trainer = "FSDPTrainer"
n_epochs = 1
n_examples = None
eval_batch_size = 4
eval_every = 20000
n_eval_examples = 16
ulimit_value = 32000
voters_model = "gpt35"


def run_dpo(dataset, sft_exp_dir, exp_name):
    command = [
        "python3",
        "-u",
        "train.py",
        f"model={model}",
        f"model.archive=.cache/adamlesnikowski/{sft_exp_dir}/LATEST/policy.pt",
        f"datasets=[{dataset}]",
        f"loss={loss}",
        f"loss.beta={loss_beta}",
        f"exp_name={exp_name}",
        f"gradient_accumulation_steps={gradient_accumulation_steps}",
        f"batch_size={batch_size}",
        f"eval_batch_size={eval_batch_size}",
        f"trainer={trainer}",
        "sample_during_eval=false",
        f"model.fsdp_policy_mp={model_fsdp_policy_mp}",
        f"eval_every={eval_every}",
        f"n_eval_examples={n_eval_examples}",
        f"n_epochs={n_epochs}",
        f"n_examples={n_examples if n_examples is not None else 'null'}",
    ]
    subprocess.run(command)


def run_a_arm(sft_exp_dir):
    exp_name = (
        f"{a_arm_dataset}_dataset_{loss}_loss_{model}_model_{batch_size}_batch_size"
    )
    run_dpo(a_arm_dataset, sft_exp_dir, exp_name)


def run_b_arm(sft_exp_dir):
    exp_name = (
        f"{b_arm_dataset}_dataset_{loss}_loss_{model}_model_{batch_size}_batch_size"
    )
    run_dpo(b_arm_dataset, sft_exp_dir, exp_name)


def run_n_arm(sft_exp_dir):
    exp_name = f"no_train_{a_arm_dataset}_dataset_{loss}_loss_{model}_model_{batch_size}_batch_size"
    run_dpo(n_arm_dataset, sft_exp_dir, exp_name)


def main():
    print(
        f"Starting DPO experiments for a-arm dataset: {a_arm_dataset} and b-arm dataset: {b_arm_dataset}..."
    )
    print(
        f"batch_size: {batch_size}, gradient_accumulation_steps: {gradient_accumulation_steps}; "
        f"effective_batch_size: {batch_size * gradient_accumulation_steps}"
    )

    # Set ulimit (Note: This might not work on all systems and might require additional permissions)
    import resource

    resource.setrlimit(resource.RLIMIT_NOFILE, (ulimit_value, ulimit_value))

    if len(sys.argv) < 3:
        print("Usage: python script.py <arm> <sft_exp_dir>")
        sys.exit(1)

    arm = sys.argv[1]
    sft_exp_dir = sys.argv[2]

    if arm == "a":
        run_a_arm(sft_exp_dir)
    elif arm == "b":
        run_b_arm(sft_exp_dir)
    elif arm == "n":
        run_n_arm(sft_exp_dir)
    else:
        print(
            "Invalid argument, use e.g. 'a' or 'b' for running a or b arm respectively."
        )


if __name__ == "__main__":
    main()
