#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=7-00:00:00
#SBATCH --mem=72GB
#SBATCH --gres=gpu:1

source env/bin/activate

# "marco-o1","openO1","qwq_preview", "llama3_1_8B", "qwen2_5_7B"
# srun --ntasks=1 --cpus-per-task=32  python src/training/icl_main_method_with_error_fixing.py --folder llama3_1_8B > logs/system/icl_main_method_with_error_fixing_llama3_1_8B_${SLURM_JOB_ID}.txt

# srun --ntasks=1 --cpus-per-task=32  python src/training/icl_main_method_with_error_fixing.py --folder qwen2_5_7B > logs/system/icl_main_method_with_error_fixing_qwen2_5_7B_${SLURM_JOB_ID}.txt

# srun --ntasks=1 --cpus-per-task=32  python src/training/icl_main_method_with_error_fixing.py --folder openO1 > logs/system/icl_main_method_with_error_fixing_openO1_${SLURM_JOB_ID}.txt

# srun --ntasks=1 --cpus-per-task=32  python src/training/icl_main_method_with_error_fixing.py --folder marco-o1 > logs/system/icl_main_method_with_error_fixing_marco-o1_${SLURM_JOB_ID}.txt

srun --ntasks=1 --cpus-per-task=32  python src/training/icl_main_method_with_error_fixing.py --folder qwq_preview > logs/system/icl_main_method_with_error_fixing_qwq_preview_${SLURM_JOB_ID}.txt