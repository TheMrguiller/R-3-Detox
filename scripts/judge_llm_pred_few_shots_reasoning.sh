#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=7-00:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1

source src/evaluation/JudgeLM/judgelm/bin/activate
# pip install flash-attn==2.0.4 --no-build-isolation
data_path="data/interim/judgellm_few_shots_pair_reasoning/"
save_path="data/processed/judgellm_few_shots_pair_reasoning/"
# Ensure the save directory exists
if [ ! -d "$save_path" ]; then
    echo "Directory $save_path does not exist. Creating it..."
    mkdir -p "$save_path"
fi
file_list=($(find "$data_path" -maxdepth 1 -type f -name "*.json" -exec realpath {} \;))
# Debugging: Check file list
echo "File list: ${file_list[@]}"
echo "Number of files: ${#file_list[@]}"
params=(33)
# Iterate over params
for param in "${params[@]}"; do
    # Define directories for the current parameter
    save_path_name="${save_path}params${param}/"
    # save_path_name_reversed="${save_path}params${param}_reversed/"

    # Ensure directories exist
    if [ ! -d "$save_path_name" ]; then
        echo "Directory $save_path_name does not exist. Creating it..."
        mkdir -p "$save_path_name"
    fi
    # if [ ! -d "$save_path_name_reversed" ]; then
    #     echo "Directory $save_path_name_reversed does not exist. Creating it..."
    #     mkdir -p "$save_path_name_reversed"
    # fi

    # Iterate over files
    for file in "${file_list[@]}"; do
        # Debugging: Paths and files
        base_name=$(basename "${file}")
        echo "Processing parameter: ${param}"
        echo "Processing file: ${base_name}"
        if [ "${base_name}" = "judgellm_qwq_preview-judgellm_marco-o1.json" ]; then
            continue
        fi
        if [ "${base_name}" = "judgellm_openO1-judgellm_marco-o1.json" ]; then
            continue
        fi
        
        # echo "Question file: ${data_path}${base_name}"
        # echo "Answer file: ${save_path_name}${base_name}"
        # echo "Answer file reversed: ${save_path_name_reversed}${base_name}"

        # Run the command (replace with actual execution logic)
        srun --ntasks=1 --cpus-per-task=32 python src/evaluation/JudgeLM/JudgeLM/judgelm/llm_judge/gen_model_judgement_multi.py \
            --model-path "BAAI/JudgeLM-${param}B-v1.0" \
            --model-id "${param}b-JudgeLM" \
            --question-file "${data_path}${base_name}" \
            --answer-file "${save_path_name}${base_name}" \
            --num-gpus-per-model 1 \
            --num-gpus-total 1 \
            --temperature 0 \
            --if-fast-eval 1 > logs/system/generating_judge_pred_${SLURM_JOB_ID}.txt
        break
        # srun --ntasks=1 --cpus-per-task=32 python src/evaluation/JudgeLM/JudgeLM/judgelm/llm_judge/gen_model_judgement_multi.py \
        #     --model-path "BAAI/JudgeLM-${param}B-v1.0" \
        #     --model-id "${param}b-JudgeLM" \
        #     --question-file "${data_path}${base_name}" \
        #     --answer-file "${save_path_name_reversed}${base_name}" \
        #     --num-gpus-per-model 1 \
        #     --num-gpus-total 1 \
        #     --temperature 0 \
        #     --if-fast-eval 1 \
        #     --if-reverse-answers 1 > logs/system/generating_judge_pred_reversed_${SLURM_JOB_ID}.txt
    done
done

echo "All tasks completed."