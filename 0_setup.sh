#!/bin/bash
# This script is intended to setup a new machine for our DCPO project.



function install_packages() {
    sudo apt install -y neovim htop atop bmon tree python3.10-venv zsh unzip
}

function install_omz() {
    sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
    chsh -s $(which zsh)
}

function install_vim_plug() {
    mkdir -p .config/nvim
    touch .config/nvim/init.vim
    sh -c 'curl -fLo "${XDG_DATA_HOME:-$HOME/.local/share}"/nvim/site/autoload/plug.vim --create-dirs \
          https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim'
    bash -c 'nvim +PlugUpdate +qall'
}






python3.11 -m venv env
source env/bin/activate
pip install --upgrade pip
echo "source env/bin/activate" >> ~/.bashrc 
echo "clear" >> ~/.bashrc

git config --global user.email "adam.lesnikowski@gmail.com"
git config --global user.name "Adam Lesnikowski"
git config --global credential.helper cache
git config --global credential.helper 'cache --timeout=360000'


### Data setup
git clone https://github.com/lesnikow/llm-sct.git


### DPO, SFT setup
cd ~
git clone https://github.com/lesnikow/direct-preference-optimization.git 
cd direct-preference-optimization
pip install -r requirements.txt

# vim .env
source .env
wandb login $WANDB_API_KEY


### Evals via fast-chat setup
cd $HOME
python3 -m venv env-fastchat
deactivate && source $HOME/env-fastchat/bin/activate
pip install --upgrade pip

cd fast-chat
pip3 install -e ".[model_worker,webui]"
pip3 install -e ".[model_worker,webui,llm_judge]"

pip install anthropic openai==0.28

