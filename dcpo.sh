#!/bin/bash
#SBATCH --job-name=dcpo
#SBATCH --output=slurm/%j.out
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --gpus=A100-PCI-80GB:1
##SBATCH --gres=shard:24
#SBATCH --time=1-00:00:00
#SBATCH --qos=high

## Status
hostname
date
gpustat

## Setup
source /nas/ucb/adamlesnikowski/env/bin/activate
cd /nas/ucb/adamlesnikowski/dpo

## Setup wandb
source .env
wandb login $WANDB_API_KEY

## SFT
ulimit -n 64000
gradient_accumulation_steps=128
batch_size=1
trainer='BasicTrainer'
voters_model='gpt35'
cd /nas/ucb/adamlesnikowski/dpo
eval_batch_size=2

function run_sft {
  dataset=$1
  exp_name=$2
  python -u train.py \
    model=pythia28 \
    datasets=[$dataset] \
    loss=sft \
    exp_name=$exp_name \
    gradient_accumulation_steps=$gradient_accumulation_steps \
    batch_size=$batch_size \
    eval_batch_size=$eval_batch_size \
    trainer=$trainer \
    sample_during_eval=false \
    model.fsdp_policy_mp=bfloat16
}

### Run
dataset=$1
exp_name="${dataset}_dataset_sft_loss_pythia28_${batch_size}_batch_size"
run_sft $dataset $exp_name

