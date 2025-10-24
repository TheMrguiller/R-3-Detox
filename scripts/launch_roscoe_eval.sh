#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --mem=48GB
##SBATCH --gres=gpu:1


cd src/evaluation/parlai-app

source parlai_env/bin/activate
# bash projects/roscoe/roscoe_data/download_annotated.sh
# python -c "import nltk; nltk.download('punkt_tab')"
# python -c "import nltk; nltk.download('stopwords')"
# python projects/roscoe/roscoe.py -t sim_sce -m facebook/roscoe-512-roberta-base --dataset-path projects/roscoe/roscoe_data/human_eval/ --output-directory projects/roscoe/roscoe_data/human_eval_output/ --datasets reasoning_openO1 reasoning_skywork-o1 > logs_reasoning_marco-o1_${SLURM_JOB_ID}.txt
# python projects/roscoe/roscoe.py -t sim_sce -m facebook/roscoe-512-roberta-base --dataset-path projects/roscoe/roscoe_data/human_eval/ --output-directory projects/roscoe/roscoe_data/human_eval_output/ --datasets reasoning_marco-o1 reasoning_openai_o1 reasoning_openO1 reasoning_qwq_preview reasoning_skywork-o1 > logs_reasoning_marco-o1_${SLURM_JOB_ID}.txt
