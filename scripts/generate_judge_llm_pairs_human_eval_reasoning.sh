#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --mem=48GB
##SBATCH --gres=gpu:1

source src/evaluation/JudgeLM/judgelm/bin/activate
script_dir="data/interim/judgellm_reasoning_human_eval/"
save_path="data/interim/judgellm_humman_eval_pairs_reasoning/"
# Ensure the save directory exists
if [ ! -d "$save_path" ]; then
    echo "Directory $save_path does not exist. Creating it..."
    mkdir -p "$save_path"
fi
echo "Script dir: $script_dir"
# Get a list of .json files in the directory
# Populate the file list
file_list=($(find "$script_dir" -maxdepth 1 -type f -name "*.json" -exec realpath {} \;))

# Debugging: Check file list
echo "File list: ${file_list[@]}"
echo "Number of files: ${#file_list[@]}"

# Pairwise processing
for ((i=0; i<${#file_list[@]}; i++)); do
    for ((j=i+1; j<${#file_list[@]}; j++)); do
        
        # Get base names without extensions
        base_name_i=$(basename "${file_list[i]}" .json)
        base_name_j=$(basename "${file_list[j]}" .json)
        
        # Construct new file name
        name="${save_path}${base_name_i}-${base_name_j}.json"

        # Debugging: Print what is being processed
        echo "Processing pair: ${base_name_i} and ${base_name_j}"
        echo "Output file: $name"
        
        # Run the command
        srun python src/evaluation/JudgeLM/JudgeLM/judgelm/data/JudgeLM/judgelm_preprocess.py \
            --ans1_file_path "${file_list[i]}" \
            --ans2_file_path "${file_list[j]}" \
            --save_path "$name"
        
        # Debugging: Confirm completion
        echo "Finished processing $name"
    done
done
