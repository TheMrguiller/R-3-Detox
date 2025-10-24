#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=7-00:00:00
#SBATCH --mem=72GB
##SBATCH --gres=gpu:1
source env/bin/activate

# "marco-o1","openO1","qwq_preview","skywork-o1"
srun --ntasks=1 --cpus-per-task=32  python src/external_approaches/paradetox.py > logs/system/paradetox_${SLURM_JOB_ID}.txt