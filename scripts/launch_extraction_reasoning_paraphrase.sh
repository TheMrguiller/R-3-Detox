#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=7-00:00:00
#SBATCH --mem=72GB
#SBATCH --gres=gpu:1
source env/bin/activate

# srun --ntasks=1 --cpus-per-task=16  python src/training/extract_reasoning_paraphrase_pipeline.py > logs/system/extract_reasoning_paraphrase_pipeline_${SLURM_JOB_ID}.txt
srun --ntasks=1 --cpus-per-task=16  python src/training/extract_reasoning_paraphrase_pipeline_new.py > logs/system/extract_reasoning_paraphrase_pipeline_new_${SLURM_JOB_ID}.txt