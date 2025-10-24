#!/usr/bin/env python3
import pandas as pd
import os
import numpy as np
import multiprocessing
from pandarallel import pandarallel
from better_profanity import ProfanityExtractor
from clean_text import TextToxicityCleaner
from utils import parallel_process_dataframe_with_progress_multi_function,text_dedup_parallel
from transformers import AutoTokenizer
num_cores=multiprocessing.cpu_count() if os.getenv("SLURM_CPUS_PER_TASK") is None else int(os.getenv("SLURM_CPUS_PER_TASK"))
pandarallel.initialize(progress_bar=True,nb_workers=num_cores)
project_path=os.path.abspath(__file__).split('src')[0]


def process_profanity_list(chunk,text_column):
    profanity_checker = ProfanityExtractor()
    
    chunk["explicit_vocab"]=chunk[text_column].apply(lambda x: profanity_checker.detect_censor_words(x))
    return chunk

def process_dataset_parallel_detoxification(output_path):
    
    df_train_no_aggre = pd.read_csv(project_path+"data/raw/parallel_detoxification_dataset/parallel_detoxification_dataset_small.tsv", sep='\t')
    df_train_no_aggre.rename(columns={"toxic_comment":"sentence"}, inplace=True)
    df_train = df_train_no_aggre.groupby("sentence").agg(
        sentence = ("sentence", "first"),
        paraphrase=("civil_comment", lambda x: list(set(x)))
    )
    if not os.path.exists(output_path):
        os.makedirs(output_path)
   
    df_train = parallel_process_dataframe_with_progress_multi_function(df_train, num_cores,"sentence", process_profanity_list,desc="Processing profanity")
    df_final = pd.DataFrame({
            'source': "parallel_detoxification",
            'split': "train",
            'sentence': df_train['sentence'],
            'paraphrase': df_train['paraphrase'],
            'label':np.ones(len(df_train)).tolist(),
            'explicit_vocab': df_train['explicit_vocab']
        })
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df_final.to_csv(f"{output_path}/parallel_detoxification_dataset.csv", index=False)
    return df_final

def process_dataset_toxicity_ParaDetox(output_path):
   
    df_train = pd.read_csv(project_path+"data/raw/paradetox/paradetox.tsv", sep='\t')
    df_train.rename(columns={"toxic":"sentence"}, inplace=True)
    df_train["paraphrase"] = df_train.parallel_apply(lambda row: list({val for val in [row["neutral1"], row["neutral2"], row["neutral3"]] if pd.notna(val)}), axis=1)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df_train = parallel_process_dataframe_with_progress_multi_function(df_train, num_cores,"sentence", process_profanity_list,desc="Processing profanity")
    df_final = pd.DataFrame({
            'source': "paradetox",
            'split': "train",
            'sentence': df_train['sentence'],
            'paraphrase': df_train['paraphrase'],
            'label':np.ones(len(df_train)).tolist(),
            'explicit_vocab': df_train['explicit_vocab']
        })
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df_final.to_csv(f"{output_path}/ParaDetox.csv", index=False)
    return df_final

def process_dataset_APPDIA(output_path):
    
    df_train = pd.read_csv(project_path+"data/raw/APPDIA/original-train.tsv", sep='\t')
    df_valid = pd.read_csv(project_path+"data/raw/APPDIA/original-dev.tsv", sep='\t')
    df_test = pd.read_csv(project_path+"data/raw/APPDIA/original-test.tsv", sep='\t')
    df_train.rename(columns={"offensive-text":"sentence","style-transferred-text":"paraphrase"}, inplace=True)
    df_valid.rename(columns={"offensive-text":"sentence","style-transferred-text":"paraphrase"}, inplace=True)
    df_test.rename(columns={"offensive-text":"sentence","style-transferred-text":"paraphrase"}, inplace=True)
    df_train["split"] = "train"
    df_valid["split"] = "val"
    df_test["split"] = "test"
    df_train
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df_intermediate = pd.concat([df_train,df_valid,df_test])
    df_intermediate.reset_index(drop=True, inplace=True)
    df_intermediate = parallel_process_dataframe_with_progress_multi_function(df_intermediate, num_cores, "sentence" ,process_profanity_list,desc="Processing profanity")
    text_cleaner=TextToxicityCleaner()
    df_intermediate=text_cleaner.clean(df_intermediate,"sentence",profane_word_column="explicit_vocab")
    df_intermediate=text_dedup_parallel(df_intermediate, 'sentence')
    df_intermediate.dropna(subset=['sentence'],inplace=True)
    df_intermediate.reset_index(drop=True, inplace=True)
    df_intermediate=text_cleaner.clean(df_intermediate,"paraphrase",profane_word_column="explicit_vocab")
    df_intermediate["paraphrase"]=df_intermediate["paraphrase"].apply(lambda x: [x])
    df_intermediate.reset_index(drop=True, inplace=True)
    df_final = pd.DataFrame({
            'source': "APPDIA",
            'split': df_intermediate['split'],
            'sentence': df_intermediate['sentence'],
            'paraphrase': df_intermediate['paraphrase'],
            'label':np.ones(len(df_intermediate)).tolist(),
            'explicit_vocab': df_intermediate['explicit_vocab']
        })
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df_final.to_csv(f"{output_path}/APPDIA.csv", index=False)
    return df_final

def process_dataset__non_toxic(output_path):
    df = pd.read_csv(project_path+"data/raw/toxicity_dataset/TheMrguiller/toxicity_big_bird_default_train.csv")
    df = df[(df["label"] == 0) & (df["article_name"] == "Jigsaw unintended bias in toxicity classification") & (df["number_of_annotators"] >= 10)]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    df.rename(columns={"text":"sentence"}, inplace=True)
    df["token_lenght"] = df["sentence"].parallel_apply(lambda x: len(tokenizer(x)["input_ids"]))
    df = df[df["token_lenght"]<=100]
    df.drop(columns=["token_lenght"], inplace=True)
    df.drop(columns=["article_name","data_source","toxicity_score","number_of_annotators"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["paraphrase"] = [""]*len(df)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df_final = pd.DataFrame({
            'source': "non_toxic",
            'split': "train",
            'sentence': df['sentence'],
            'paraphrase': df['paraphrase'],
            'label':df["label"],
            'explicit_vocab': df["explicit_vocab"]
        })
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df_final.to_csv(f"{output_path}/non_toxic.csv", index=False)
    return df_final

# def process_dataset_CAPP(output_path):
#     df_appdia = pd.read_csv(project_path+"data/external/CAPP_article/APPDIA_Generated-Paraphrases.csv")
#     df_paradetox = pd.read_csv(project_path+"data/external/CAPP_article/ParaDetox_Generated-Paraphrases.csv")
#     df_final = pd.concat([df_paradetox,df_appdia])
#     df_final.reset_index(drop=True, inplace=True)
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#     df_final.to_csv(f"{output_path}/experiment_test_dataset.csv", index=False)


if __name__ == "__main__":
    
    df_appdia=process_dataset_APPDIA(project_path+"data/interim/dataset/")
    df_paradetox=process_dataset_toxicity_ParaDetox(project_path+"data/interim/dataset/")
    df_parallel=process_dataset_parallel_detoxification(project_path+"data/interim/dataset/")
    df_non_toxic=process_dataset__non_toxic(project_path+"data/interim/dataset/")
    df_final = pd.concat([df_appdia,df_paradetox,df_parallel])
    df_final = pd.concat([df_final,df_non_toxic.sample(n=len(df_final),replace=False,random_state=42)])
    df_final.reset_index(drop=True, inplace=True)
    df_final.to_csv(project_path+"data/processed/dataset/dataset.csv", index=False)
    
