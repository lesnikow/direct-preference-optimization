#!/bin/bash
# Eval of models



dpo_exp_dirs=(
    "shp_maj_data_dataset_dpo_loss_pythia69_model_8_batch_size_2024-09-20_20-06-14_700176"
    "shp_sc_data_dataset_dpo_loss_pythia69_model_8_batch_size_2024-09-20_23-17-00_932340"
    "no_train_shp_maj_data_dataset_sft_loss_pythia69_model_1_batch_size_2024-09-21_21-09-45_286675"
)


function convert_models {

    deactivate && source $HOME/env/bin/activate
    cd $HOME/direct-preference-optimization/

    for exp_dir in "${dpo_exp_dirs[@]}"; do
      in_path="$HOME/direct-preference-optimization/.cache/adamlesnikowski/${exp_dir}/LATEST/policy.pt"
      du -sh "${in_path}"
      python3 convert_model.py --in_path ${in_path}
    done
}


function make_fastchat_llm_judge_model_answers {
    date


}



function make_fastchat_llm_judge_model_judgements {
    date


}


function show_results {
    date


}



function main {
    convert_models
    make_fastchat_llm_judge_model_answers
    make_fastchat_llm_judge_model_judgements
    show_results
}


main
