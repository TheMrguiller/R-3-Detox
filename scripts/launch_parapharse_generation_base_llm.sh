#!/bin/bash
#SBATCH --partition=gpu-fast
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=7-00:00:00
#SBATCH --mem=72GB
#SBATCH --gres=gpu:1
source env/bin/activate

# "marco-o1","openO1","qwq_preview","skywork-o1"
srun --ntasks=1 --cpus-per-task=32  python src/external_approaches/base_llm.py --model_name llama3_1_8B > logs/system/base_llm_llama3_1_8B_${SLURM_JOB_ID}.txt

srun --ntasks=1 --cpus-per-task=32  python src/external_approaches/base_llm.py --model_name marco-o1 > logs/system/base_llm_marco-o1_${SLURM_JOB_ID}.txt

srun --ntasks=1 --cpus-per-task=32  python src/external_approaches/base_llm.py --model_name qwen2_5_7B > logs/system/base_llm_qwen2_5_7B_${SLURM_JOB_ID}.txt

srun --ntasks=1 --cpus-per-task=32  python src/external_approaches/base_llm.py --model_name openO1 > logs/system/base_llm_openO1_${SLURM_JOB_ID}.txt

srun --ntasks=1 --cpus-per-task=32  python src/external_approaches/base_llm.py --model_name qwq_preview > logs/system/base_llm_qwq_preview_${SLURM_JOB_ID}.txt