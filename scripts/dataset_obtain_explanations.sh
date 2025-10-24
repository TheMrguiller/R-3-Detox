#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
source env/bin/activate

## tomh/toxigen_hatebert
## tomh/toxigen_roberta 
## unitary/toxic-bert
## unitary/unbiased-toxic-roberta
## Xuhui/ToxDect-roberta-large

srun --ntasks=1 --cpus-per-task=32  python src/data/data_explanability.py --model_name $1 > logs/system/data_explanability_${SLURM_JOB_ID}.txt
# srun --ntasks=1 --cpus-per-task=32  python src/data/dataset_clean_shap.py > logs/system/dataset_clean_shap_${SLURM_JOB_ID}.txt

