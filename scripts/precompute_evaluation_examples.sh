#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
#SBATCH --mem=72GB
##SBATCH --gres=gpu:1

source env/bin/activate

srun --ntasks=1 --cpus-per-task=32  python src/data/precompute_evaluation_examples.py --chunk_index $1 --all_chunks False > logs/system/precompute_evaluation_examples_${SLURM_JOB_ID}.txt


