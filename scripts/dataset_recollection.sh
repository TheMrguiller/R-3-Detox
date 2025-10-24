#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --mem=24GB
##SBATCH --mail-type=ALL

source env/bin/activate
source scripts/my_custom.env
# srun --ntasks=1 --cpus-per-task=16  python src/data/dataset_download.py > logs/system/dataset_download_${SLURM_JOB_ID}.txt
srun --ntasks=1 --cpus-per-task=16  python src/data/dataset_preprocess.py > logs/system/dataset_preprocess_${SLURM_JOB_ID}.txt
