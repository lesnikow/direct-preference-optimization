#!/bin/bash
# Soft fine-tuning (SFT) experiments

gradient_accumulation_steps=4
batch_size=32
trainer='FSDPTrainer'
# trainer='BasicTrainer'
voters_model='gpt35'
eval_batch_size=4
eval_every=40000
ulimit_value=32000

function run_sft {
    dataset=$1
    exp_name=$2
    python3 -u train.py \
      model=pythia28 \
      datasets="[$dataset]" \
      loss=sft \
      exp_name="$exp_name" \
      gradient_accumulation_steps="$gradient_accumulation_steps" \
      batch_size="$batch_size" \
      eval_batch_size="$eval_batch_size" \
      trainer="$trainer" \
      sample_during_eval=false \
      model.fsdp_policy_mp=bfloat16 \
      eval_every="$eval_every"
}

function run_a_arm {
    dataset='shp_maj_data'
    exp_name="${dataset}_dataset_sft_loss_pythia28_${batch_size}_batch_size"
    run_sft "$dataset" "$exp_name"
}

function run_b_arm {
    dataset='shp_sc_data'
    exp_name="${dataset}_dataset_sft_loss_pythia28_${batch_size}_batch_size"
    run_sft "$dataset" "$exp_name"
}


function main {
    cd "$HOME/direct-preference-optimization" || { echo "Directory not found!"; exit 1; }
    ulimit -n "$ulimit_value"
    if [ "$1" == "a" ]; then
        run_a_arm
    elif [ "$1" == "b" ]; then
        run_b_arm
    else
        echo "Invalid argument, use e.g. 'a' or 'b' for running a or b arm respectively."
    fi
}


main "$1"

