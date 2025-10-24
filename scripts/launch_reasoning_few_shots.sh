#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=7-00:00:00
#SBATCH --mem=72GB
#SBATCH --gres=gpu:1
source env/bin/activate

# "marco-o1","openO1","qwq_preview","skywork-o1"
# srun --ntasks=1 --cpus-per-task=32  python src/training/reasoning_few_shots_generation_pipeline.py > logs/system/reasoning_few_shots_generation_pipeline_${SLURM_JOB_ID}.txt
# srun --ntasks=1 --cpus-per-task=16  python src/training/reasoning_few_shots_generation_pipeline_new.py --config_name marco-o1 > logs/system/reasoning_few_shots_generation_pipeline_new_${SLURM_JOB_ID}.txt
# srun --ntasks=1 --cpus-per-task=16  python src/training/reasoning_few_shots_generation_pipeline_new.py --config_name openO1 > logs/system/reasoning_few_shots_generation_pipeline_new_${SLURM_JOB_ID}.txt
srun --ntasks=1 --cpus-per-task=16  python src/training/reasoning_few_shots_generation_pipeline_new.py --config_name $1 > logs/system/reasoning_few_shots_generation_pipeline_new_${SLURM_JOB_ID}.txt