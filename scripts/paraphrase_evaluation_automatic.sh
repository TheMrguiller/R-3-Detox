#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=7-00:00:00
#SBATCH --mem=32GB
##SBATCH --gres=gpu:1

source env/bin/activate

srun --ntasks=1 --cpus-per-task=32  python src/evaluation/launch_paraphrase_evaluation.py --folder $1 > logs/system/launch_paraphrase_evaluation_${SLURM_JOB_ID}.txt


# srun --ntasks=1 --cpus-per-task=32 python src/evaluation/toxicity.py > results/reports/paraphrase_automatic_metrics/final/paraphrase_base_toxicity.txt