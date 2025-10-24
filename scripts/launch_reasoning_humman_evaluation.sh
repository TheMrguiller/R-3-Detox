#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
#SBATCH --mem=72GB
#SBATCH --gres=gpu:1
source env/bin/activate

srun --ntasks=1 --cpus-per-task=32  python src/training/reasoning_generation_pipeline_for_human_evaluation.py --config_name marco-o1 --proxy_type huggingface > logs/system/reasoning_generation_pipeline_for_human_evaluation_${SLURM_JOB_ID}.txt
# srun --ntasks=1 --cpus-per-task=32  python src/training/reasoning_generation_pipeline_for_human_evaluation.py --config_name openO1 --proxy_type huggingface > logs/system/reasoning_generation_pipeline_for_human_evaluation_${SLURM_JOB_ID}.txt
# srun --ntasks=1 --cpus-per-task=32  python src/training/reasoning_generation_pipeline_for_human_evaluation.py --config_name qwq_preview --proxy_type huggingface > logs/system/reasoning_generation_pipeline_for_human_evaluation_${SLURM_JOB_ID}.txt
# srun --ntasks=1 --cpus-per-task=32  python src/training/reasoning_generation_pipeline_for_human_evaluation.py --config_name skywork-o1 --proxy_type huggingface > logs/system/reasoning_generation_pipeline_for_human_evaluation_${SLURM_JOB_ID}.txt