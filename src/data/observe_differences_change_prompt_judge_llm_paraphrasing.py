import pandas as pd
import os
import glob
import argparse
project_path=os.path.abspath(__file__).split('src')[0]
import sys
sys.path.append(project_path)
from typing import List
import random
from tqdm import tqdm


def generate_ranking_per_evaluation(result:str):
    result_base = result.split(" ")
    result= []
    for i in range(len(result_base)):
        result.append(float(result_base[i]))
    return result


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_prediction_path", default=project_path + 'data/processed/human_eval_paraphrasing_judgellm_round_1/')
    args = parser.parse_args()
    data_dir = args.judge_prediction_path
    
    models_new = ["params7","params13","params33"]
    models_base =["params7_copy","params13_copy","params33_copy"]
    for model_new,model_base in zip(models_new,models_base):
        files_new = glob.glob(data_dir+model_new+"/*.json")
        files_base = glob.glob(data_dir+model_base+"/*.json")
        # df_ranking_new= pd.DataFrame(columns=["idx","result"])
        # df_ranking_base = pd.DataFrame(columns=["idx","result"])
        dfs_new = [pd.read_json(file, lines=True) for file in files_new]
        dfs_base = [pd.read_json(file, lines=True) for file in files_base]
        df_ranking_bases = [pd.DataFrame(columns=["idx","result"]) for i in range(len(dfs_base))]
        df_ranking_news = [pd.DataFrame(columns=["idx","result"]) for i in range(len(dfs_new))]
        for idx in tqdm(range(len(dfs_new[0]))):
            for i in range(len(files_base)):
                df_ranking_bases[i].loc[len(df_ranking_bases[i])] = [idx,generate_ranking_per_evaluation(dfs_base[i].loc[idx]["pred_text"])]
                df_ranking_news[i].loc[len(df_ranking_news[i])] = [idx,generate_ranking_per_evaluation(dfs_new[i].loc[idx]["pred_text"])]
        number_of_elements_different = [0,0,0]
        for idx in tqdm(range(len(df_ranking_news[0]))):
            for i in range(len(files_new)):
                if df_ranking_news[i].loc[idx]["result"] != df_ranking_bases[i].loc[idx]["result"]:
                    number_of_elements_different[i]+=1
        print(f"Number of elements different for {model_new} compared to {model_base}: {number_of_elements_different}")
    
                
            
