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
srun --ntasks=1 --cpus-per-task=32  python src/training/icl_main_method.py --model_name $1> logs/system/icl_main_method_${SLURM_JOB_ID}.txt