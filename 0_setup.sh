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


function setup_env_dpo() {
    python3 -m venv env-dpo
    source env-dpo/bin/activate
    pip install --upgrade pip
    echo "source env-dpo/bin/activate" >> ~/.bashrc 
}


function setup_env_fastchat() {
    cd $HOME
    python3 -m venv env-fastchat
    deactivate && source env-fastchat/bin/activate
    pip install --upgrade pip
    deactivate
}


function config_git() {
    git config --global user.email "adam.lesnikowski@gmail.com"
    git config --global user.name "Adam Lesnikowski"
    git config --global credential.helper cache
    git config --global credential.helper 'cache --timeout=360000'
}


function setup_llm_sct_repo() {
    cd $HOME
    git clone https://github.com/lesnikow/llm-sct.git
}


function setup_direct_preference_optimization_repo() {
    cd $HOME
    git clone https://github.com/lesnikow/direct-preference-optimization.git 
    cd direct-preference-optimization
    source env-dpo/bin/activate
    pip install -r requirements.txt
    deactivate && cd $HOME
}


function_setup_fastchat_repo() {
    cd $HOME
    git clone https://github.com/lesnikow/fast-chat.git
    cd fast-chat
    source env-fastchat/bin/activate
    pip install -e ".[model_worker,webui,llm_judge]"
    pip install anthropic openai==0.28
    deactivate && cd $HOME
}


function wanddb_login() {
    source $HOME/direct-preference-optimization/.env
    wandb login $WANDB_API_KEY
}


function main() {
    echo "Starting 0_setup.sh script..."

    echo "Finished 0_setup.sh script."
}

main

