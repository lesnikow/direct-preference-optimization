#!/bin/bash
# Direct preference optimization (DPO) experiments

a_arm_dataset='maj_shp_data_v3_topic_matched_2400'
b_arm_dataset='sc_shp_data_v3_topic_matched_2400'
n_arm_dataset='null_data'
model='pythia69'
model_fsdp_policy_mp="bfloat16"
loss='dpo'
loss_beta=0.1
batch_size=8
gradient_accumulation_steps=1
trainer='FSDPTrainer'
n_epochs=1
n_examples=null

eval_batch_size=4
eval_every=20000
n_eval_examples=16

ulimit_value=32000
voters_model='gpt35'


function run_dpo {
    dataset=$1
    sft_exp_dir=$2
    exp_name=$3

    python3 -u train.py \
      model="$model" \
      model.archive=".cache/adamlesnikowski/${sft_exp_dir}/LATEST/policy.pt" \
      datasets="[$dataset]" \
      loss="$loss" \
      loss.beta="$loss_beta" \
      exp_name="$exp_name" \
      gradient_accumulation_steps="$gradient_accumulation_steps" \
      batch_size="$batch_size" \
      eval_batch_size="$eval_batch_size" \
      trainer="$trainer" \
      sample_during_eval=false \
      model.fsdp_policy_mp="$model_fsdp_policy_mp" \
      eval_every="$eval_every" \
      n_eval_examples="$n_eval_examples" \
      n_epochs="$n_epochs" \
      n_examples="$n_examples"
}


function run_a_arm {
    sft_exp_dir=$2
    exp_name="${a_arm_dataset}_dataset_${loss}_loss_${model}_model_${batch_size}_batch_size"
    run_dpo "$a_arm_dataset" "$sft_exp_dir" "$exp_name" 
}


function run_b_arm {
    sft_exp_dir=$2
    exp_name="${b_arm_dataset}_dataset_${loss}_loss_${model}_model_${batch_size}_batch_size"
    run_dpo "$b_arm_dataset" "$sft_exp_dir" "$exp_name"
}


function run_n_arm {
    sft_exp_dir=$2
    exp_name="no_train_${a_arm_dataset}_dataset_${loss}_loss_${model}_model_${batch_size}_batch_size"
    run_dpo "$n_arm_dataset" "$sft_exp_dir" "$exp_name"
}


function main {
    echo "Starting DPO experiments for a-arm dataset: $a_arm_dataset and b-arm dataset: $b_arm_dataset..."
    echo "batch_size: $batch_size, gradient_accumulation_steps: $gradient_accumulation_steps; effective_batch_size: $((batch_size * gradient_accumulation_steps))"

    # cd "$HOME/direct-preference-optimization" || { echo "Directory not found!"; exit 1; }
    ulimit -n "$ulimit_value"

    if [ "$1" == "a" ]; then
        run_a_arm "$@"
    elif [ "$1" == "b" ]; then
        run_b_arm "$@"
    elif [ "$1" == "n" ]; then
        run_n_arm "$@"
    else
        echo "Invalid argument, use e.g. 'a' or 'b' for running a or b arm respectively."
    fi
}


main "$1" "$2"
