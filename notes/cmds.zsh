# notes/cmds.zsh for dpo repo

## Setup cmds for sft, dpo

sudo apt install -y neovim htop atop bmon tree python3.10-venv zsh unzip

mkdir -p .config/nvim
touch .config/nvim/init.vim
sh -c 'curl -fLo "${XDG_DATA_HOME:-$HOME/.local/share}"/nvim/site/autoload/plug.vim --create-dirs \
       https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim'
# nvim :PlugUpdate

python3 -m venv env
source env/bin/activate
pip install --upgrade pip
echo "source env/bin/activate" >> ~/.bashrc 
echo "clear" >> ~/.bashrc

git config --global user.email "adam.lesnikowski@gmail.com"
git config --global user.name "Adam Lesnikowski"
git config --global credential.helper cache
git config --global credential.helper 'cache --timeout=36000'


### Data setup
git clone https://github.com/lesnikow/llm-sct.git


### DPO, SFT setup
cd ~
git clone https://github.com/lesnikow/direct-preference-optimization.git dpo
cd dpo
pip install -r requirements.txt

# vim .env
source .env
wandb login $WANDB_API_KEY


### Evals via fast-chat setup
cd /nas/ucb/adamlesnikowski
python3 -m venv env-fastchat
source env-fastchat/bin/activate
pip install --upgrade pip

git clone https://github.com/lesnikow/fast-chat.git fastchat
cd fastchat

pip3 install -e ".[model_worker,webui]"
pip install anthropic openai==0.28




## CHAI slurm cmds


# Basic slurm cmds
srun --mem=100mb --time=0:01:00 cat /etc/hostname
srun --mem=1GB --time=0:01:00 --gpus=1 nvidia-smi
srun --mem=1GB --time=0:01:00 --gpus=A6000:1 nvidia-smi
srun --mem=1GB --time=0:01:00 --gpus=A4000:1 --job-name=adam_test nvidia-smi 

# Interactive jobs
srun --mem=1GB --time=0:01:00 --gpus=A4000:1 --job-name=adam_test --pty bash
srun --pty --mem 1gb --time=01:00:00 bash

# QoS cmds
srun --mem=1GB --time=0:01:00 --gpus=A6000:1 --job-name=adam_test --qos=high nvidia-smi 
srun --mem=1GB --time=0:01:00 --gpus=A6000:4 --job-name=adam_test --qos=high nvidia-smi 

# Running from slurm.sh
srun --mem=1GB --time=0:01:00 --gpus=A4000:2 --job-name=adam_test --qos=high /nas/ucb/adamlesnikowski/slurm.sh
srun --mem=1GB --time=0:01:00 --gpus=A4000:4 --job-name=adam_test --qos=high /nas/ucb/adamlesnikowski/slurm.sh
srun --mem=1GB --time=0:01:00 --gpus=A6000:4 --job-name=adam_test --qos=high /nas/ucb/adamlesnikowski/slurm.sh

# A100 gpus cmds
srun --mem=1GB --time=0:01:00 --gpus=A100-SXM4-80GB:1 --job-name=adam_test --qos=high /nas/ucb/adamlesnikowski/slurm.sh
srun --mem=1GB --time=0:01:00 --gpus=A100-PCI-80GB:1 --job-name=adam_test --qos=high /nas/ucb/adamlesnikowski/slurm.sh
srun --mem=1GB --time=0:01:00 --gpus=A100-SXM4-80GB:4 --job-name=adam_test --qos=high /nas/ucb/adamlesnikowski/slurm.sh
srun --mem=1GB --time=0:01:00 --gpus=A100-PCI-80GB:4 --job-name=adam_test --qos=high /nas/ucb/adamlesnikowski/slurm.sh



# Interactive job to debug my train run sft, dpo, wanbd
srun --pty --mem=8gb -c 4 --gres=shard:8 --time=01:00:00 bash
srun --pty --mem=32gb -c 4 --gpus=A100-SXM4-80GB:1 --time=04:00:00 bash
srun --pty --mem=32gb -c 4 --gpus=A100-SXM4-80GB:2 --qos=high --time=04:00:00 bash
srun --pty --mem=32gb -c 4 --gpus=A100-SXM4-80GB:2 --qos=high --time=04:00:00 "bash"

srun --pty --mem=32gb -c 4 --gpus=A100-PCI-80GB:4 --qos=high --time=04:00:00 "bash"
srun --pty --mem=32gb -c 64 --gpus=A100-PCI-80GB:4 --qos=high --time=04:00:00 "bash"

srun --pty --mem=512G -c 128 --gpus=A100-PCI-80GB:4 --qos=high --time=04:00:00 "bash"
srun --pty --mem=512G -c 128 --gpus=A100-PCI-80GB:2 --qos=high --time=04:00:00 "bash"
srun --pty --mem=512G -c 128 --gpus=A100-PCI-80GB:1 --qos=high --time=04:00:00 "bash"

srun --pty --mem=512G -c 128 --gpus=A100-SXM4-80GB:1 --qos=high --time=04:00:00 "bash"
srun --pty --mem=512G -c 128 --gpus=A6000:1 --qos=high --time=04:00:00 "bash"

srun --pty --mem=512G -c 128 --gpus=A100-SXM4-80GB:1 --qos=high --time=12:00:00 "bash"
srun --pty --mem=512G -c 128 --gpus=A100-PCI-80GB:1 --qos=high --time=12:00:00 "bash"

srun --pty --mem=256G -c 128 --gpus=A100-PCI-80GB:1 --qos=high --time=12:00:00 "bash"

# Slurm cluster info
sinfo -N -O "NodeList:4,CPUsState:.15,Memory:.9 ,FreeMem:.9 ,StateCompact:6,Gres:30,GresUsed:50" | grep A100
sinfo -N -O "NodeList:4,CPUsState:.15,Memory:.9 ,FreeMem:.9 ,StateCompact:6,Gres:30,GresUsed:50"


## Two arm trial, using different preference models, on helpful base dataset

# A-arm: Random voter, using thirty three voters, three judgement models,
# llama-3-8b?, oai gpt-3.5 01-25, anthropic claude 3 haiku
# B-arm: Majority prefernce, using thirty three voters, three judgement models,
# llama-3-8b?, oai gpt-3.5 01-25, anthropic claude 3 haiku

## SFT
ulimit -n 64000
gradient_accumulation_steps=8
batch_size=64
eval_batch_size=$batch_size

### A-arm, random voter, 33 voters
dataset="rv_33_voters"
exp_name="${dataset}_dataset_sft_loss_pythia28_${batch_size}_batch_size"
python -u train.py \
    model=pythia28 \
    datasets=[${dataset}] \
    loss=sft \
    exp_name=${exp_name} \
    gradient_accumulation_steps=$gradient_accumulation_steps \
    batch_size=$batch_size \
    eval_batch_size=$eval_batch_size \
    trainer=BasicTrainer \
    sample_during_eval=false \
    model.fsdp_policy_mp=bfloat16

### B-arm, majority preferences, 33 voters
dataset="mp_33_voters"
exp_name="${dataset}_dataset_sft_loss_pythia28_${batch_size}_batch_size"
python -u train.py \
    model=pythia28 \
    datasets=[${dataset}] \
    loss=sft \
    exp_name=${exp_name} \
    gradient_accumulation_steps=$gradient_accumulation_steps \
    batch_size=$batch_size \
    eval_batch_size=$eval_batch_size \
    trainer=BasicTrainer \
    sample_during_eval=false \
    model.fsdp_policy_mp=bfloat16


## DPO
### Parameters
loss_beta=0.1
ulimit -n 64000
gradient_accumulation_steps=8
batch_size=32
eval_batch_size=8

### Define function to call python -u train.py with args dataset, exp_name, sft_exp_dir
function run_dpo {
  dataset=$1
  exp_name=$2
  sft_exp_dir=$3

  python -u train.py \
    model=pythia28 \
    datasets=[$dataset] \
    loss=dpo \
    loss.beta=$loss_beta \
    exp_name=$exp_name \
    gradient_accumulation_steps=$gradient_accumulation_steps \
    batch_size=$batch_size \
    eval_batch_size=$eval_batch_size \
    trainer=BasicTrainer \
    sample_during_eval=false \
    model.fsdp_policy_mp=bfloat16 \
    model.archive=".cache/adamlesnikowski/${sft_exp_dir}/LATEST/policy.pt"
}


### A-arm, random voter, 33 voters
dataset="rv_33_voters"
exp_name="${dataset}_dataset_dpo_loss_pythia28_${batch_size}_batch_size"
sft_exp_dir="rv_33_voters_dataset_sfo_loss_pythia28_64_batch_size_2024-06-21_14-59-56_224105"
run_dpo $dataset $exp_name $sft_exp_dir


### B-arm, majority preference, 33 voters
dataset="mp_33_voters"
exp_name="${dataset}_dataset_dpo_loss_pythia28_${batch_size}_batch_size"
sft_exp_dir="mp_33_voters_dataset_sfo_loss_pythia28_64_batch_size_2024-06-21_15-15-36_010543"
run_dpo $dataset $exp_name $sft_exp_dir






## Five arm trial

# A-arm: Anthropic helpful-base

# B-arm: Random voter, "DPO" in two arm trial
 
# C-arm: Majority preferences, "DCPO" in two arm trial
 
# D-arm: All voters
 
# E-arm: Majority preferences x n




## SFT

ulimit -n 64000
gradient_accumulation_steps=2
batch_size=64
eval_batch_size=$batch_size

### A-arm, helpful-base hb dataset, sft loss
python -u train.py \
    model=pythia28 \
    datasets=[hb] \
    loss=sft \
    exp_name=hb_dataset_sft_loss_pythia28 \
    gradient_accumulation_steps=$gradient_accumulation_steps \
    batch_size=$batch_size \
    eval_batch_size=$eval_batch_size \
    trainer=FSDPTrainer \
    sample_during_eval=false \
    model.fsdp_policy_mp=bfloat16


### B-arm random voter rv dataset sft loss
python -u train.py \
    model=pythia28 \
    datasets=[rv] \
    loss=sft \
    exp_name=rv_dataset_sft_loss_pythia28 \
    gradient_accumulation_steps=$gradient_accumulation_steps \
    batch_size=$batch_size \
    eval_batch_size=$eval_batch_size \
    trainer=FSDPTrainer \
    sample_during_eval=false \
    model.fsdp_policy_mp=bfloat16


### C-arm majority pref mp dataset sft loss
python -u train.py \
    model=pythia28 \
    datasets=[mp] \
    loss=sft \
    exp_name=mp_dataset_sft_loss_pythia28 \
    gradient_accumulation_steps=$gradient_accumulation_steps \
    batch_size=$batch_size \
    eval_batch_size=$eval_batch_size \
    trainer=FSDPTrainer \
    sample_during_eval=false \
    model.fsdp_policy_mp=bfloat16


### D-arm all voters av dataset sft loss
python -u train.py \
    model=pythia28 \
    datasets=[av] \
    loss=sft \
    exp_name=av_dataset_sft_loss_pythia28 \
    gradient_accumulation_steps=$gradient_accumulation_steps \
    batch_size=$batch_size \
    eval_batch_size=$eval_batch_size \
    trainer=FSDPTrainer \
    sample_during_eval=false \
    model.fsdp_policy_mp=bfloat16


### E-arm repeated majority pref n times mpn dataset sft loss
python -u train.py \
    model=pythia28 \
    datasets=[rmp] \
    loss=sft \
    exp_name=mpn_dataset_sft_loss_pythia28 \
    gradient_accumulation_steps=$gradient_accumulation_steps \
    batch_size=$batch_size \
    eval_batch_size=$eval_batch_size \
    trainer=FSDPTrainer \
    sample_during_eval=false \
    model.fsdp_policy_mp=bfloat16


## DPO

ulimit -n 64000
loss_beta=0.1
gradient_accumulation_steps=2
batch_size=64
eval_batch_size=$batch_size


### A-arm, helpful-base hb dataset, dpo loss
dataset="hb"
exp_name="${dataset}_dataset_dpo_loss_pythia28"
sft_exp_dir="hb_dataset_sft_loss_pythia28_2024-06-06_23-35-42_315586/"

python -u train.py \
  model=pythia28 \
  datasets=[$dataset] \
  loss=dpo \
  loss.beta=$loss_beta \
  exp_name=$exp_name \
  gradient_accumulation_steps=$gradient_accumulation_steps \
  batch_size=$batch_size \
  eval_batch_size=$eval_batch_size \
  trainer=FSDPTrainer \
  sample_during_eval=false \
  model.fsdp_policy_mp=bfloat16 \
  model.archive=".cache/root/${sft_exp_dir}/LATEST/policy.pt"


### B-arm random voter rv dataset, dpo loss
dataset="rv"
exp_name="${dataset}_dataset_dpo_loss_pythia28"
sft_exp_dir="rv_dataset_sft_loss_pythia28_2024-06-07_18-00-46_380909/"

python -u train.py \
  model=pythia28 \
  datasets=[$dataset] \
  loss=dpo \
  loss.beta=$loss_beta \
  exp_name=$exp_name \
  gradient_accumulation_steps=$gradient_accumulation_steps \
  batch_size=$batch_size \
  eval_batch_size=$eval_batch_size \
  trainer=FSDPTrainer \
  sample_during_eval=false \
  model.fsdp_policy_mp=bfloat16 \
  model.archive=".cache/root/${sft_exp_dir}/LATEST/policy.pt"


### C-arm majority pref mp dataset, dpo loss
dataset="mp"
exp_name="${dataset}_dataset_dpo_loss_pythia28"
sft_exp_dir="mp_dataset_sft_loss_pythia28_2024-06-07_18-09-45_573945"

python -u train.py \
  model=pythia28 \
  datasets=[$dataset] \
  loss=dpo \
  loss.beta=$loss_beta \
  exp_name=$exp_name \
  gradient_accumulation_steps=$gradient_accumulation_steps \
  batch_size=$batch_size \
  eval_batch_size=$eval_batch_size \
  trainer=FSDPTrainer \
  sample_during_eval=false \
  model.fsdp_policy_mp=bfloat16 \
  model.archive=".cache/root/${sft_exp_dir}/LATEST/policy.pt"


### D-arm all voters av dataset, dpo loss
dataset="av"
exp_name="${dataset}_dataset_dpo_loss_pythia28"
sft_exp_dir="av_dataset_sft_loss_pythia28_2024-06-06_20-27-23_235653"

python -u train.py \
  model=pythia28 \
  datasets=[$dataset] \
  loss=dpo \
  loss.beta=$loss_beta \
  exp_name=$exp_name \
  gradient_accumulation_steps=$gradient_accumulation_steps \
  batch_size=$batch_size \
  eval_batch_size=$eval_batch_size \
  trainer=FSDPTrainer \
  sample_during_eval=false \
  model.fsdp_policy_mp=bfloat16 \
  model.archive=".cache/root/${sft_exp_dir}/LATEST/policy.pt"


### E-arm repeated majority pref n times mpn dataset, dpo loss
dataset="rmp"
exp_name="${dataset}_dataset_dpo_loss_pythia28"
sft_exp_dir="mpn_dataset_sft_loss_pythia28_2024-06-06_22-00-50_990593"

python -u train.py \
  model=pythia28 \
  datasets=[$dataset] \
  loss=dpo \
  loss.beta=$loss_beta \
  exp_name=$exp_name \
  gradient_accumulation_steps=$gradient_accumulation_steps \
  batch_size=$batch_size \
  eval_batch_size=$eval_batch_size \
  trainer=FSDPTrainer \
  sample_during_eval=false \
  model.fsdp_policy_mp=bfloat16 \
  model.archive=".cache/root/${sft_exp_dir}/LATEST/policy.pt"





## Evals
### Run model adapter code models

dpo_exp_dirs=(
  "rv_33_voters_dataset_dpo_loss_pythia28_32_batch_size_2024-06-21_19-42-41_738922"
  "mp_33_voters_dataset_dpo_loss_pythia28_32_batch_size_2024-06-21_19-50-12_125630"
)
for exp_dir in ${dpo_exp_dirs[@]}; do
  dataset=$(echo $exp_dir | cut -d'_' -f1)
  in_path="/nas/ucb/adamlesnikowski/dpo/.cache/adamlesnikowski/${exp_dir}/LATEST/policy.pt"
  du -sh ${in_path}
  python3 convert_model.py --in_path ${in_path}
done



### Make fast-chat llm-judge answers
max_new_tokens=512
function gen_model_answers {
  dpo_exp_dirs=$1
  max_new_tokens=$2
  for exp_dir in ${dpo_exp_dirs[@]}; do
    dataset=$(echo $exp_dir | cut -d'_' -f1)
    model_path="/nas/ucb/adamlesnikowski/dpo/.cache/adamlesnikowski/${exp_dir}/LATEST/converted/"
    echo "Model path: ${model_path}"
    python3 gen_model_answer.py \
      --model-path ${model_path} \
      --model-id ${exp_dir} \
      --num-gpus-total 1 \
      --max-new-token ${max_new_tokens}
  done
}

dpo_exp_dirs=(
  "rv_33_voters_dataset_dpo_loss_pythia28_32_batch_size_2024-06-21_19-42-41_738922"
)
gen_model_answers "${dpo_exp_dirs}" "${max_new_tokens}"


dpo_exp_dirs=(
  "mp_33_voters_dataset_dpo_loss_pythia28_32_batch_size_2024-06-21_19-50-12_125630"
)
gen_model_answers "${dpo_exp_dirs}" "${max_new_tokens}"


### Make fast-chat llm-judge judgements
#### Single mode 
source /nas/ucb/adamlesnikowski/dpo/.env
export OPENAI_API_KEY

model_answers=(
  "rv_3_x_11_voters_dpo_loss_pythia28_32_batch_size_2024-06-21_19-42-41_738922"
  "mp_3_x_11_voters_dpo_loss_pythia28_32_batch_size_2024-06-21_19-50-12_125630"
)
python3 gen_judgment.py \
  --model-list "${model_answers[@]}" \
  --parallel 16 \
  --mode single \
  --judge-model "gpt-4-turbo"


python3 show_result.py \
  --mode "single" \
  --judge-model "gpt-4-turbo"


#### Pairwise-all mode
source /root/fast-chat/.env
export OPENAI_API_KEY

python3 gen_judgment.py \
  --model-list "${model_answers[@]}" \
  --parallel 16 \
  --mode pairwise-all \
  --judge-model "gpt-4-turbo"

python3 show_result.py \
  --mode "pairwise-all" \
  --judge-model "gpt-4-turbo" \
  --model-list "${model_answers[@]}"

















### Copy over files in between instances

src_id=11087135
dest_id=11119184
src_fp="/root/dpo/outgoing/"
dest_fp="/root/incoming/"

vastai execute ${src_id} "ls -l ${src_fp}"
vastai execute ${src_id} "du -sh ${src_fp}"

vastai execute ${dest_id} "ls -l ${dest_fp}"
vastai execute ${dest_id} "du -sh ${dest_fp}"

vastai copy ${src_id}:${src_fp} ${dest_id}:${dest_fp}




### Copy over lm_eval results dir None to local machine

rsync -avP -e 'ssh -p 25805' root@45.135.56.11:/root/dpo/None ~/dpo/results/


## Test direct-preference-optimization repo cmds

python -m pytest -vv --durations=10 preference_datasets.py 















