# notes/cmds.zsh for dpo repo

## Setup cmds for sft, dpo

sudo apt install -y neovim htop atop bmon tree python3.10-venv zsh

mkdir -p .config/nvim
sh -c 'curl -fLo "${XDG_DATA_HOME:-$HOME/.local/share}"/nvim/site/autoload/plug.vim --create-dirs \
       https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim'
#nvim :PlugUpdate

python3 -m venv env --system-site-packages
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
pip install -r requirements-pytorch-container.txt

source .env
wandb login $WANDB_API_KEY


### Evals via fast-chat setup
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
## Run model adapter code models
dpo_exp_dirs=(
  "hb_dataset_dpo_loss_pythia28_2024-06-07_18-19-17_935174"
  "rv_dataset_dpo_loss_pythia28_2024-06-07_18-38-18_587808"
  "mp_dataset_dpo_loss_pythia28_2024-06-07_18-59-42_906288"
)

in_paths=()
for exp_dir in ${dpo_exp_dirs[@]};
do
  in_paths+=("/root/dpo/.cache/root/${exp_dir}/LATEST/policy.pt")
done

# run du -sh on each in_path
for i in {1..2};
do
  du -sh ${in_paths[i]}
done


for i in {1..2};
do
  python3 convert_model.py --in_path ${in_paths[i]} 
done




python3 convert_model.py --in_path [] --out_path []


### fast-chat llm-judge




python gen_model_answer.py --model-path [MODEL-PATH] --model-id [MODEL-ID]



































### Copy over lm_eval results dir None to local machine

rsync -avP -e 'ssh -p 25805' root@45.135.56.11:/root/dpo/None ~/dpo/results/


## Test direct-preference-optimization repo cmds

python -m pytest -vv --durations=10 preference_datasets.py 















