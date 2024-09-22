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


function generate_model_answers {
    num_gpus=4
    exp_dir=$1
    max_new_tokens=$2
    model_path="$HOME/direct-preference-optimization/.cache/adamlesnikowski/${exp_dir}/LATEST/converted/"
    echo "Model path: ${model_path}"
    python3 gen_model_answer.py \
    --model-path ${model_path} \
    --model-id ${exp_dir} \
    --num-gpus-total ${num_gpus} \
    --max-new-token ${max_new_tokens}
}


function make_fastchat_llm_judge_model_answers {
    source $HOME/env-fastchat/bin/activate
    cd $HOME/fast-chat/fastchat/llm_judge/

    source $HOME/direct-preference-optimization/.env
    export OPENAI_API_KEY
    max_new_tokens=128

    for exp_dir in "${dpo_exp_dirs[@]}"; do
        echo "Generating model answers for ${exp_dir}"
        generate_model_answers "${exp_dir}" "${max_new_tokens}"
    done
}



function make_fastchat_llm_judge_model_judgements {
    python3 gen_judgment.py \
      --mode "single" \
      --judge-model "gpt-4-turbo" \
      --model-list "${dpo_exp_dirs[@]}" \
      --parallel 256

    python3 gen_judgment.py \
      --mode "pairwise-all" \
      --judge-model "gpt-4-turbo" \
      --model-list "${dpo_exp_dirs[@]}" \
      --parallel 256
}


function show_results {
    python3 show_result.py \
      --mode "single" \
      --judge-model "gpt-4-turbo" \
      --model-list "${dpo_exp_dirs[@]}"

    python3 show_result.py \
      --mode "pairwise-all" \
      --judge-model "gpt-4-turbo" \
      --model-list "${dpo_exp_dirs[@]}"
}



function main {
    # convert_models
    # make_fastchat_llm_judge_model_answers
    make_fastchat_llm_judge_model_judgements
    # show_results
}


main















































