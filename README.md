# Reflect, Reason, Rephrase ($\textup{R}^3$-Detox) : An In-Context Learning Approach to Text Detoxification
<div align="center">

[![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)](https://github.com/TheMrguiller/Collaborative-Content-Moderation)
[<img src="https://img.shields.io/badge/Framework-PyTorch-red.svg"/>](https://pytorch.org/)
[![GitHub license](https://img.shields.io/github/license/eureka-research/Eureka)](LICENSE)
______________________________________________________________________
<img src="https://raw.githubusercontent.com/TheMrguiller/R-3-Detox/main/resources/methodology.svg" width="800">
<p><em>Figure 1: Overall diagram of the proposed $\textup{R}^3$-Detox framework. Initially, we preprocess the data presented in Section \ref{dataset} by extracting SHAP values from a set of toxicity detectors. Next, we generate guided reasoning using a set of self-reflection models. We then process their output reasoning to ensure that it contains no code-switching or data leakage, meaning the final non-toxic paraphrase is not explicitly present in the reasoning prior to the detoxification phase. We then evaluate the models and select the best reasoning for each comment. Finally, we validate the few-shot learning examples generated through our $\textup{R}^3$-Detox framework by comparing them to state-of-the-art detoxification techniques using ICL. </em></p>
</div>

## Abstract
Traditional content moderation censors harmful content, but often limits user participation. Text detoxification offers a better alternative, promoting civility without silencing voices. However, prior approaches oversimplify the task by treating detoxification as a one-step process, neglecting the deep contextual analysis needed to remove toxicity while preserving meaning. In this paper, we introduce $\textup{R}^3$-Detox—a Reflect, Reason, and Rephrase framework that enhances detoxification through a structured three-step process, all executed within a single prompt. First, we instruct the LLM to analyze potential toxic words or phrases, guided by Shapley values from toxicity detectors, to counteract potential hallucinations. Next, the model assesses the overall toxicity of the sentence based on these identified elements. Finally, leveraging this prior analysis, the model reasons about necessary modifications to eliminate toxicity while maintaining meaning. We apply this framework and Self-Reflection models to enrich offensive content paraphrasing datasets—ParaDetox, Parallel Detoxification, and APPDIA—by adding explicit detoxification reasoning to each instance, which originally contained only input sentences and their paraphrases. We evaluate our methodology using In-Context Learning, comparing $\textup{R}^3$-Detox against state-of-the-art methods on the same datasets. Experimental results show that our approach outperforms existing methodologies, even in instruction-following models.

## To-Do List ✅

- [ ] Upload the Data 📂
- [ ] Upload the Code 📝
## Instalation guide
The installation is dual as we have two repositories embedded: JudgeLLM repository and R-3-Detxo. The base python version is 3.9.18.
To install R-3-Detox, we need to use the following command:
```
virtualenv venv env
source env/bin/activate
pip install -r requirements.txt
```
To install the JudgeLLM, we recommend generating a new environment as they have different library requirements. The requirements is located at src/evaluation/JudgeLM/requirements.txt. So the next command is needed:
```
pip install -r src/evaluation/JudgeLM/requirements.txt
```
## Data and data preprocessing

This document explains how to obtain the raw datasets used in the experiments, how to preprocess them to create the combined dataset.

1) Obtaining the datasets

From the project root you can run the downloader entrypoint which fetches public datasets and stores them under `data/raw/` and `data/external/`:
Remember activating or setting your HuggingFace Token in your environment variables.
```bash
export HUGGINGFACE_TOKEN=
source env/bin/activate
python src/data/dataset_download.py
```

Main datasets downloaded by the script:

- `data/raw/APPDIA/` — APPDIA original splits (train/dev/test)
- `data/raw/paradetox/` — ParaDetox tsv files
- `data/raw/parallel_detoxification_dataset/` — Parallel Detoxification tsv
- `data/raw/toxicity_dataset/` — HuggingFace snapshot (toxicity dataset)
- `data/external/CAPP_article/` — paraphrase CSVs generated externally and used in the experiments

If you prefer or need to run the download inside a SLURM job, use the SLURM helper scripts that activate the environment.

2) Preprocessing / creating the combined dataset

After the raw files are available, create the cleaned and merged dataset used in our experiments by running:

```bash
# run locally
python src/data/dataset_preprocess.py

# or via SLURM helper (activates env and runs the preprocessing)
sbatch scripts/dataset_recollection.sh
```

Artifacts produced:

- `data/interim/dataset/` — intermediate per-source CSVs
- `data/processed/dataset/dataset.csv` — combined, cleaned dataset used in experiments
3) Data explainability

To generate the SHAP values using pre-trained toxicity detection models, we provide a script that processes the dataset and computes importance values for each token in the input texts. The following toxicity detection models are supported:

- tomh/toxigen_hatebert
- tomh/toxigen_roberta
- unitary/toxic-bert 
- unitary/unbiased-toxic-roberta
- Xuhui/ToxDect-roberta-large

To run the SHAP value generation:

```bash
# Run locally (replace model_name with one from the list above)
python src/data/data_explanability.py --model_name "unitary/toxic-bert"

# Or using SLURM (recommended for large datasets)
sbatch scripts/dataset_obtain_explanations.sh "unitary/toxic-bert"
```

The script will:
1. Load the preprocessed dataset from `data/processed/dataset/dataset.csv`
2. Use the specified toxicity detection model to analyze each text
3. Generate SHAP values showing which tokens contribute most to toxicity detection
4. Save the results in `data/interim/shap_values/dataset_{model_name}.csv`

Each generated CSV will contain:
- Original text
- Tokenized text
- SHAP values per token (indicating importance for toxicity prediction)
- Model's toxicity prediction score

Once all the shap values are obtain, we recommend cleaning the SHAP values in order to get a more contextualized results where words are generated from tokens.
```
python src/data/dataset_preprocess.py
```
To obtain the final dataset with the SHAP values, we provide with a function to aggregate the values.
```
python src/data/dataset_preprocess.py
```
python src/data/data_shap_aggregation.py
```
The aggregated SHAP values are stored in `data/processed/shap_values_aggregated processed_shap_values.csv` and are used to guide the detoxification process by identifying which parts of the text most strongly indicate toxicity.

4) Final paraphrasing and reasoning values

We have provided the final results of the research to facilitate easy replication of the experiments. The dataset containing the few shot examples is located in the data/processed/final_few_shot_reasoning folder. This directory includes data such as the non toxic and toxic datapoints, along with their corresponding reasoning, explaining why a sample is considered non toxic and how it can be transformed into its paraphrased version.

The final paraphrases used in the experiments, from which we derived the quality metrics of the generated paraphrases, are stored in the data/processed/final_paraphrases folder. This directory contains only the toxic inputs that were converted into non toxic paraphrases, together with their associated reasoning. In this case, we provide the results for each model and experiment discussed throughout the article.

Notes: the preprocessing performs profanity detection, cleaning and deduplication. See `src/data/dataset_preprocess.py` for implementation details and configurable options.

## Few-shot obtainance Precompute few-shot & evaluation examples

To precompute few-shot examples and evaluation slices (these scripts are chunkable and SLURM-ready):

```bash
# Precompute few-shot examples (use chunk index for parallel runs)
sbatch scripts/precompute_few_shots_examples.sh <chunk_index>

# Precompute evaluation examples (use chunk index for parallel runs)
sbatch scripts/precompute_evaluation_examples.sh <chunk_index>
```

Outputs are typically stored in `data/precomputed_few_shots_examples/` and `data/precomputed_evaluation_examples/`.

4) Running paraphrase generation

Generation launchers live in the `scripts/` directory. Examples include:

- `scripts/launch_parapharse_generation.sh` — generic launcher
- `scripts/launch_parapharse_generation_base_llm.sh` — base LLM launcher
- `scripts/launch_parapharse_generation_detoxllm.sh` — detox-focused LLM launcher

Open the launcher you want and adjust the model identifier, output folder and environment variables. They typically activate `env/bin/activate` and call the appropriate `src/` inference script.

5) Automatic evaluation of paraphrases

Use the evaluation launcher to compute reference-based and reference-free metrics on generated paraphrases.

```bash
# run evaluation via SLURM helper
sbatch scripts/paraphrase_evaluation_automatic.sh <folder_name>

# or run locally against a folder of generated outputs
python src/evaluation/launch_paraphrase_evaluation.py --folder llama3_1_8B
```

Inputs and outputs:

- Input: generated paraphrase CSVs placed in `data/processed/final_paraphrases/<folder_name>/` (one file per model/seed).
- Output: metric CSVs and JSON summaries under `results/metrics/paraphrase_automatic_metrics/`.

Evaluation details
- Reference-based metrics: BLEU, ROUGE, BERTScore, and toxicity checks.
- Reference-free metrics: content-similarity, fluency, style-transfer metrics and custom metrics implemented in `src/evaluation/metrics.py`.
- Advanced: JudgeLM-based evaluation and ParlAI/Roscoe utilities are available under `src/evaluation/JudgeLM/` and `src/evaluation/parlai-app/projects/roscoe/` respectively. See their READMEs for installation and usage.

6) Troubleshooting & tips

- Most helper scripts assume SLURM. To run locally, remove `sbatch`/`srun` wrappers and call the Python entrypoints directly.
- If a download fails, inspect `src/data/dataset_download.py` for the source URL and try a manual download.
- Ensure `env/` exists and required packages are installed; create a venv via `python -m venv env && source env/bin/activate && pip install -r requirements.txt` if you add a consolidated `requirements.txt`.

If you'd like, I can add a minimal consolidated `requirements.txt` and a short example showing how to run the full pipeline locally without SLURM.
