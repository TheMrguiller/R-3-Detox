#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
#SBATCH --mem=24GB
##SBATCH --gres=gpu:1
source env/bin/activate


# srun --ntasks=1 --cpus-per-task=32  python src/evaluation/toxicity.py > results/reports/paraphrase_automatic_metrics/final/paraphrase_base_toxicity.txt
srun --ntasks=1 --cpus-per-task=32  python src/evaluation/metrics.py > logs/system/metrics_${SLURM_JOB_ID}.txt

# srun --ntasks=1 --cpus-per-task=32  python src/data/dataset_clean_shap.py > logs/system/dataset_clean_shap_${SLURM_JOB_ID}.txt

