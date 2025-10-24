#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
#SBATCH --mem=24GB
##SBATCH --gres=gpu:1

source env/bin/activate
#llama3_1_8B,openO1,marco-o1,qwen2_5_7B
# srun --ntasks=1 --cpus-per-task=32  python src/data/extract_paraphrase_from_experiment_v2.py --folder marco-o1 > logs/system/extract_paraphrase_from_experiment_v2_marco_o1_${SLURM_JOB_ID}.txt

# srun --ntasks=1 --cpus-per-task=32  python src/data/extract_paraphrase_from_experiment_v2.py --folder openO1 > logs/system/extract_paraphrase_from_experiment_v2_openO1_${SLURM_JOB_ID}.txt

# srun --ntasks=1 --cpus-per-task=32  python src/data/extract_paraphrase_from_experiment_v2.py --folder llama3_1_8B > logs/system/extract_paraphrase_from_experiment_v2_llama3_1_8B_${SLURM_JOB_ID}.txt

# srun --ntasks=1 --cpus-per-task=32  python src/data/extract_paraphrase_from_experiment_v2.py --folder qwen2_5_7B > logs/system/extract_paraphrase_from_experiment_v2_qwen2_5_7B_${SLURM_JOB_ID}.txt


srun --ntasks=1 --cpus-per-task=32  python src/data/extract_paraphrase_from_experiment_v2.py --folder qwq_preview > logs/system/extract_paraphrase_from_experiment_v2_qwen2_5_7B_${SLURM_JOB_ID}.txt
