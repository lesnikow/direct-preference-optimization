# notes/cmds.zsh for dpo repo

## Commands for sft, dpo

sudo apt install -y neovim htop atop bmon tree python3.10-venv zsh

python3 -m venv env --system-site-packages
source env/bin/activate
pip install --upgrade pip

git config --global user.email "adam.lesnikowski@gmail.com"
git config --global user.name "Adam Lesnikowski"
git config --global credential.helper cache
git config --global credential.helper 'cache --timeout=36000'


### Data
git clone https://github.com/lesnikow/llm-sct.git


### DPO, SFT
cd ~
git clone https://github.com/lesnikow/direct-preference-optimization.git dpo
cd dpo
pip install -r requirements-pytorch-container.txt

source .env
wandb login WANDB_API_KEY


### Evals via fast-chat
cd ~
git clone https://github.com/lesnikow/fast-chat.git
cd fast-chat
pip3 install -e ".[model_worker,webui]"
pip install anthropic openai








## Five arm trial

# A-arm: Anthropic helpful-base

# B-arm: Random voter, "DPO" in two arm trial
 
# C-arm: Majority preferences, "DCPO" in two arm trial
 
# D-arm: All voters
 
# E-arm: Majority preferences x n




## Basic experiment setup, replicated dpo codebase

[X] Eval data DCPO, loss DPO

[X] Eval data DPO, loss DPO

[X] Dataset DCPO, loss DPO

[X] Dataset DPO,  loss DPO

[X] Dataset DCPO, loss SFT

[X] Dataset DPO,  loss SFT



## SFT


### B-arm 4 x H100 80 GB, ulimit, more freq evals, DCPO datset, sft loss
ulimit -n 64000; python -u train.py model=pythia28 datasets=[dcpo] loss=sft exp_name=dataset_DCPO_loss_sft_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=10000

### A-arm 4 x H100 80 GB, ulimit, more freq evals, DPO datset, sft loss
ulimit -n 64000; python -u train.py model=pythia28 datasets=[dpo] loss=sft exp_name=dataset_DPO_loss_sft_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=10000


### 4 x H100 80 GB, more freq evals
ulimit -n 64000; python -u train.py model=pythia28 datasets=[hh] loss=sft exp_name=anthropic_dpo_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=10000

### 4 x H100 80 GB 
ulimit -n 64000; python -u train.py model=pythia28 datasets=[hh] loss=sft exp_name=anthropic_dpo_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16

### 2 x H100 80 GB 
ulimit -n 64000; python -u train.py model=pythia28 datasets=[hh] loss=sft exp_name=anthropic_dpo_pythia28 gradient_accumulation_steps=4 batch_size=32 eval_batch_size=16 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16

### 1 x H100 80 GB, stable at close to 65 GB memory used
ulimit -n 64000; python -u train.py model=pythia28 datasets=[hh] loss=sft exp_name=anthropic_dpo_pythia28 gradient_accumulation_steps=8 batch_size=16 eval_batch_size=8 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16


### OG command SFT
ulimit -n 64000; python -u train.py model=pythia28 datasets=[hh] loss=sft exp_name=anthropic_dpo_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16



## DPO


### B-arm continued; 4 x H100 80 GB, ulimit, more freq evals, DCPO dataset, dpo loss 
ulimit -n 64000; python -u train.py model=pythia28 datasets=[dcpo] loss=dpo loss.beta=0.1 exp_name=dataset_DCPO_loss_dpo_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=10000 model.archive=.cache/root/dataset_DCPO_loss_sft_pythia28_2024-05-22_21-13-30_959011/LATEST/policy.pt


### A-arm continued; 4 x H100 80 GB, ulimit, more freq evals, DPO dataset, dpo loss 
ulimit -n 64000; python -u train.py model=pythia28 datasets=[dpo] loss=dpo loss.beta=0.1 exp_name=dataset_DPO_loss_dpo_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=10000 model.archive=.cache/root/dataset_dpo_loss_sft_pythia_28_2024-05-22_21-00-06_950890/LATEST/policy.pt


### 4 x H100 80 GB, ulimit, more freq evals
ulimit -n 64000; python -u train.py model=pythia28 datasets=[hh] loss=dpo loss.beta=0.1 exp_name=anthropic_dpo_pythia28_dpo gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=10000 model.archive=.cache/root/.../LATEST/policy.pt

### OG command DPO
python -u train.py model=pythia28 datasets=[hh] loss=dpo loss.beta=0.1 exp_name=anthropic_dpo_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=/path/to/archive/from/sft/LATEST/policy.pt


## Evals


### Eleuther model harness


#### A-arm eval
lm_eval --model hf \
    --model_args pretrained=.cache/root/dataset_DPO_loss_dpo_pythia28_2024-05-22_21-25-05_438706/LATEST/ \
    --tasks truthfulqa \
    --device cuda:0 \
    --batch_size auto:4

#### B-arm eval
lm_eval --model hf \
    --model_args pretrained=.cache/root/dataset_DCPO_loss_dpo_pythia28_2024-05-22_21-45-42_170746/LATEST/ \
    --tasks truthfulqa \
    --device cuda:0 \
    --batch_size auto:4



### Copy over lm_eval results dir Non to local machine

rsync -avP -e 'ssh -p 25805' root@45.135.56.11:/root/dpo/None ~/dpo/results/


## Test direct-preference-optimization repo cmds

python -m pytest -vv --durations=10 preference_datasets.py 















