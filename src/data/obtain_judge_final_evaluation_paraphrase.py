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
from src.data.obtain_automatic_metric_ranking import rank_models_fractional

def generate_ranking_per_evaluation(result:str,model_names:List[str]):
    final_result = {}
    result_base = result.split(" ")
    result= []
    for i in range(len(result_base)):
        result.append(float(result_base[i]))
    for i in range(len(result)):
        final_result[model_names[i]] = result[i]
    sorted_result = sorted(final_result.items(), key=lambda x: x, reverse=False)
    ranked_result = {}
    max_value = sorted_result[0][1]
    rank=1
    for i in range(len(sorted_result)):
        if sorted_result[i][1] == max_value:
            ranked_result[sorted_result[i][0]] = rank
        else:
            rank+=1
            max_value = sorted_result[i][1]
            ranked_result[sorted_result[i][0]] = rank
    return ranked_result

def obtain_ranking_results(rankings:List[dict],models:List[str]):
    ranking_points = {
        "1":4,
        "2":3,
        "3":2,
        "4":1,
        "5":0
    }
    models_dicts = {}
    for model in models:
        models_dicts[model] = 0
    for idx,ranking in enumerate(rankings):
        if len(ranking)!=len(models):
            print("Error in ranking")
            continue
        for model in models:
            models_dicts[model]+= ranking_points[str(ranking[model])]
    return models_dicts

def ranking_per_source(df_ranking,source):
    df_ranking = df_ranking[df_ranking["source"]==source]
    df_ranking.reset_index(drop=True,inplace=True)
    ranking_result=obtain_ranking_results(df_ranking["ranking"].tolist(),list(df_ranking["ranking"].loc[0].keys()))
    ranking_result = {k: v for k, v in sorted(ranking_result.items(), key=lambda item: item[1],reverse=True)}
    with open(ranking_result_path+model+f"{source}_ranking.json", 'w') as f:
        f.write(str(ranking_result))
        f.close()
if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_prediction_path", default=project_path + 'data/processed/human_eval_paraphrasing_judgellm_round2/')
    args = parser.parse_args()
    data_dir = args.judge_prediction_path
    first_round_path = project_path + 'data/interim/judgellm_paraphrasing_ranking/'
    ranking_result_path = project_path + 'data/processed/judgellm_paraphrasing_ranking/'
    models = ["params7","params13","params33"]
    for model in models:
        files_second_round = glob.glob(data_dir+model+"/*.json")
        files_first_round = glob.glob(first_round_path+model+"/*.json")
        df_ranking = pd.DataFrame(columns=['idx',"ranking","source"])
        dfs_first_round = pd.read_json(files_first_round[0], lines=True)
        dfs_second_round = pd.read_json(files_second_round[0], lines=True)
        for idx in tqdm(range(len(dfs_second_round))):
            
            results = generate_ranking_per_evaluation(dfs_second_round.loc[idx]["pred_text"],[dfs_second_round.loc[idx]["answer1_model_id"],dfs_second_round.loc[idx]["answer2_model_id"],dfs_second_round.loc[idx]["answer3_model_id"]])
            prior_results = dfs_first_round.loc[idx]["result"]
            for result in results:
                for prior_result in prior_results:
                    keys = list(prior_result.keys())
                    if result in keys:
                        prior_rank = prior_result[result]
                        new_rank = results[result]
                        for key in keys:
                            if prior_result[key]==prior_rank:
                                prior_result[key] = new_rank
            final_rank = {}
            for prior_result in prior_results:
                for key in prior_result.keys():
                    final_rank[key] = prior_result[key]
            # final_rank = rank_models_fractional(final_rank,reversed=False)
            df_ranking.loc[len(df_ranking)] = [idx,final_rank,""]
        if not os.path.exists(ranking_result_path):
            os.makedirs(ranking_result_path)
        df = pd.read_csv(project_path+'data/interim/paraphrasing_eval_dataset/paraphrasing_evaluation_dataset.csv')
        for index,row in df_ranking.iterrows():
            if row["idx"] in df["idx"].tolist():
                df_ranking.at[index,"source"] = df[df["idx"]==row["idx"]]["source"].values[0]
        
        df_ranking.to_json(ranking_result_path+model+".json",orient="records",lines=True)
        ranking_result=obtain_ranking_results(df_ranking["ranking"].tolist(),list(df_ranking["ranking"].loc[0].keys()))
        ranking_result = {k: v for k, v in sorted(ranking_result.items(), key=lambda item: item[1],reverse=True)}
        with open(ranking_result_path+model+"general_ranking.json", 'w') as f:
            f.write(str(ranking_result))
            f.close()
        ranking_per_source(df_ranking,"APPDIA")
        ranking_per_source(df_ranking,"paradetox")
        ranking_per_source(df_ranking,"parallel_detoxification")
        
        
    
                
            
