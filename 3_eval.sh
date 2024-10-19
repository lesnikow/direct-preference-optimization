#!/bin/bash
# Eval of models


dpo_exp_dirs=(
    "sc_shp_data_v3_matched_prompts_1000_dataset_dpo_loss_pythia69_model_8_batch_size_2024-10-16_00-13-22_492768"
    "maj_shp_data_v3_matched_prompts_1000_dataset_dpo_loss_pythia69_model_8_batch_size_2024-10-16_00-13-36_789292"   
    "sc_shp_data_v3_matched_prompts_2000_dataset_dpo_loss_pythia69_model_8_batch_size_2024-10-16_00-23-55_905499"
    "maj_shp_data_v3_matched_prompts_2000_dataset_dpo_loss_pythia69_model_8_batch_size_2024-10-16_00-28-17_553502"
    "sc_shp_data_v3_matched_prompts_4000_dataset_dpo_loss_pythia69_model_8_batch_size_2024-10-16_00-41-42_307862"
    "maj_shp_data_v3_matched_prompts_4000_dataset_dpo_loss_pythia69_model_8_batch_size_2024-10-16_00-54-30_060280"
    "sc_shp_data_v3_matched_prompts_8000_dataset_dpo_loss_pythia69_model_8_batch_size_2024-10-16_01-12-20_844513"
    "maj_shp_data_v3_matched_prompts_8000_dataset_dpo_loss_pythia69_model_8_batch_size_2024-10-16_01-44-32_662714"
    "sc_shp_data_v3_matched_prompts_16000_dataset_dpo_loss_pythia69_model_8_batch_size_2024-10-16_02-10-07_721487"
    "maj_shp_data_v3_matched_prompts_16000_dataset_dpo_loss_pythia69_model_8_batch_size_2024-10-16_03-20-47_141077"
    "sc_shp_data_v3_matched_prompts_32000_dataset_dpo_loss_pythia69_model_8_batch_size_2024-10-16_04-11-40_585501"
    "maj_shp_data_v3_matched_prompts_32000_dataset_dpo_loss_pythia69_model_8_batch_size_2024-10-16_06-28-57_000634"
    "sc_shp_data_v3_matched_prompts_64000_dataset_dpo_loss_pythia69_model_8_batch_size_2024-10-16_07-39-57_040672"
    "maj_shp_data_v3_matched_prompts_64000_dataset_dpo_loss_pythia69_model_8_batch_size_2024-10-16_12-41-23_885774"
)

function convert_models() {

    source $HOME/env/bin/activate
    cd $HOME/direct-preference-optimization/

    for exp_dir in "${dpo_exp_dirs[@]}"; do
        in_path="$HOME/direct-preference-optimization/.cache/adamlesnikowski/${exp_dir}/LATEST/policy.pt"
        du -sh "${in_path}"
        python3 convert_model.py --in_path ${in_path}
    done
}


function fastchat_setup() {
    source $HOME/direct-preference-optimization/.env
    export OPENAI_API_KEY
    source $HOME/env-fastchat/bin/activate
    cd $HOME/fast-chat/fastchat/llm_judge/
}


function generate_model_answers() {
    # Helper function to generate model answers.

    num_gpus=1
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


function make_fastchat_llm_judge_model_answers() {

    export max_new_tokens=128
    for exp_dir in "${dpo_exp_dirs[@]}"; do
        echo "Generating model answers for ${exp_dir}"
        generate_model_answers "${exp_dir}" "${max_new_tokens}"
    done
}


function make_fastchat_llm_judge_model_judgements() {

    python3 gen_judgment.py \
      --mode "single" \
      --judge-model "gpt-4-turbo" \
      --model-list "${dpo_exp_dirs[@]}" \
      --parallel 256

#    python3 gen_judgment.py \
#      --mode "pairwise-all" \
#      --judge-model "gpt-4-turbo" \
#      --model-list "${dpo_exp_dirs[@]}" \
#      --parallel 256
}


function show_results() {

    python3 show_result.py \
      --mode "single" \
      --judge-model "gpt-4-turbo" \
      --model-list "${dpo_exp_dirs[@]}" | tee out_single.txt

#    python3 show_result.py \
#      --mode "pairwise-all" \
#      --judge-model "gpt-4-turbo" \
#      --model-list "${dpo_exp_dirs[@]}" | tee out_pw.txt
}



function main() {
    #convert_models
    fastchat_setup
    #make_fastchat_llm_judge_model_answers
    #make_fastchat_llm_judge_model_judgements
    show_results
}


main















































