#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=7-00:00:00
#SBATCH --mem=32GB
##SBATCH --gres=gpu:1
source env/bin/activate

# srun --ntasks=1 --cpus-per-task=32  python src/data/translate_chinese_final_results.py --folder $1 > logs/system/translate_chinese_final_results_${SLURM_JOB_ID}.txt

srun --ntasks=1 --cpus-per-task=32  python src/data/translate_chinese_final_results.py --folder qwq_preview > logs/system/translate_chinese_final_results_${SLURM_JOB_ID}.txt

srun --ntasks=1 --cpus-per-task=32  python src/data/translate_chinese_final_results.py --folder qwq_preview > logs/system/translate_chinese_final_results_${SLURM_JOB_ID}.txt


