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
from datasets import load_dataset
import numpy as np
import json
from statsmodels.stats import inter_rater as irr
from sklearn.metrics import cohen_kappa_score

def process_dataframe_ranking_second_round(df):
    df_rank= pd.DataFrame(columns=["idx","triplet_result"])
    for idx,single_annotation in tqdm(df.iterrows()):
        idx = single_annotation["metadata.idx"]
        modelA = single_annotation["metadata.ModelA_name"][0]
        modelB = single_annotation["metadata.ModelB_name"][0]
        modelC = single_annotation["metadata.ModelC_name"][0]
        ranks = single_annotation["paraphrase_response_rating.responses"]
        for index_ranks,rank in enumerate(ranks):
            for index_rank,element in enumerate(rank):
                if element["value"]=="response-1":
                    ranks[index_ranks][index_rank]["value"] = modelA
                elif element["value"]=="response-2":
                    ranks[index_ranks][index_rank]["value"] = modelB
                elif element["value"]=="response-3":
                    ranks[index_ranks][index_rank]["value"] = modelC
        final_rank = {}
        for rank in ranks:
            for element in rank:
                final_rank[element["value"]] = element["rank"]
        df_rank.loc[len(df_rank)] = [idx,final_rank]
    return df_rank

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

def change_ranking(rankings):
    final_ranking = {}
    for ranking in rankings:
        if ranking["rank"]==1:
            final_ranking[ranking["value"]] = 3
        elif ranking["rank"]==2:
            final_ranking[ranking["value"]] = 4
        elif ranking["rank"]==3:
            final_ranking[ranking["value"]] = 5
    return final_ranking
def reorganize_ranking_first_round(df_ranking):
    df_ranking["result_triplet0"] = df_ranking["result_triplet0"].apply(lambda x: change_ranking(x))
    df_ranking["result_triplet1"] = df_ranking["result_triplet1"].apply(lambda x: change_ranking(x))
    df_ranking["result_triplet2"] = df_ranking["result_triplet2"].apply(lambda x: change_ranking(x))
    return df_ranking

def update_first_round_ranking(base_dict,update_dict):
    model_name = ""
    for key in update_dict.keys():
        if key in base_dict.keys():
            model_name = key
    old_value = base_dict[model_name]
    new_value = update_dict[model_name]
    for key in base_dict.keys():
        if base_dict[key] == old_value:
            base_dict[key] = new_value
    return base_dict

def unify_ranking(df_ranking_first_round,df_ranking_second_round):
    df_final = pd.DataFrame(columns=["idx","ranking"])
    for index in range(len(df_ranking_first_round)):
        idx = df_ranking_first_round["idx"].loc[index]
        triplet0 = df_ranking_first_round[df_ranking_first_round["idx"]==idx]["result_triplet0"].values[0]
        triplet1 = df_ranking_first_round[df_ranking_first_round["idx"]==idx]["result_triplet1"].values[0]
        triplet2 = df_ranking_first_round[df_ranking_first_round["idx"]==idx]["result_triplet2"].values[0]
        triplet_result = df_ranking_second_round[df_ranking_second_round["idx"]==idx]["triplet_result"].values[0]
        triplet0=update_first_round_ranking(triplet0,triplet_result)
        triplet1=update_first_round_ranking(triplet1,triplet_result)
        triplet2=update_first_round_ranking(triplet2,triplet_result)
        triplet0.update(triplet1)
        triplet0.update(triplet2)
        triplet0 = {k: v for k, v in sorted(triplet0.items(), key=lambda item: item[1],reverse=False)}
        df_final.loc[len(df_final)] = [idx,triplet0]
    return df_final
        
def borda_count(ranks):
    rank_borda_values = {
        1:5,
        2:4,
        3:3,
        4:2,
        5:1
    }
    model_dict = {}
    for key in ranks.keys():
        model_dict[key] = rank_borda_values[ranks[key]]
    return model_dict

def rank_models(data:dict):
    # Sorting the dictionary by values in descending order
    # Assigning rank positions
    # Initialize a dictionary to store the ranks
    ranked_data = {}
    rank = 1
    max_value = max(data.values())
    for key, value in data.items():
        if value == max_value:
            ranked_data[key] = rank
        else:
            max_value = value
            rank += 1
            ranked_data[key] = rank
    return ranked_data

if __name__ =="__main__":
    df_rank_user1_round1 = pd.read_csv(project_path+"data/interim/human_eval_paraphrasing_round1/paraphrasing_human_evaluation_user_1.csv")
    df_rank_user2_round1 = pd.read_csv(project_path+"data/interim/human_eval_paraphrasing_round1/paraphrasing_human_evaluation_user_2.csv")
    df_rank_user3_round1 = pd.read_csv(project_path+"data/interim/human_eval_paraphrasing_round1/paraphrasing_human_evaluation_user_3.csv")
    
    df_rank_user1_round1["result_triplet0"] = df_rank_user1_round1["result_triplet0"].apply(lambda x: eval(x))
    df_rank_user1_round1["result_triplet1"] = df_rank_user1_round1["result_triplet1"].apply(lambda x: eval(x))
    df_rank_user1_round1["result_triplet2"] = df_rank_user1_round1["result_triplet2"].apply(lambda x: eval(x))
    df_rank_user2_round1["result_triplet0"] = df_rank_user2_round1["result_triplet0"].apply(lambda x: eval(x))
    df_rank_user2_round1["result_triplet1"] = df_rank_user2_round1["result_triplet1"].apply(lambda x: eval(x))
    df_rank_user2_round1["result_triplet2"] = df_rank_user2_round1["result_triplet2"].apply(lambda x: eval(x))
    df_rank_user3_round1["result_triplet0"] = df_rank_user3_round1["result_triplet0"].apply(lambda x: eval(x))
    df_rank_user3_round1["result_triplet1"] = df_rank_user3_round1["result_triplet1"].apply(lambda x: eval(x))
    df_rank_user3_round1["result_triplet2"] = df_rank_user3_round1["result_triplet2"].apply(lambda x: eval(x))

    df_rank_user1_round2_path = project_path+"data/interim/human_eval_paraphrasing_round2/paraphrasing_human_evaluation_user_1.csv"
    df_rank_user2_round2_path = project_path+"data/interim/human_eval_paraphrasing_round2/paraphrasing_human_evaluation_user_2.csv"
    df_rank_user3_round2_path = project_path+"data/interim/human_eval_paraphrasing_round2/paraphrasing_human_evaluation_user_3.csv"
    if not os.path.exists(project_path+"data/interim/human_eval_paraphrasing_round2/"):
            os.makedirs(project_path+"data/interim/human_eval_paraphrasing_round2/")
    if os.path.exists(df_rank_user1_round2_path):
        df_rank_user1_round2 = pd.read_csv(project_path+"data/interim/human_eval_paraphrasing_round2/paraphrasing_human_evaluation_user_1.csv")
        df_rank_user1_round2["triplet_result"] = df_rank_user1_round2["triplet_result"].apply(lambda x: eval(x))
    else:
        df_rank_user1_round2 = load_dataset("TheMrguiller/Paraphrase_Evaluation_Round_2_U0")["train"].to_pandas()
        df_rank_user1_round2 = process_dataframe_ranking_second_round(df_rank_user1_round2)
        
        df_rank_user1_round2.to_csv(df_rank_user1_round2_path)
    if os.path.exists(df_rank_user2_round2_path):
        df_rank_user2_round2 = pd.read_csv(project_path+"data/interim/human_eval_paraphrasing_round2/paraphrasing_human_evaluation_user_2.csv")
        df_rank_user2_round2["triplet_result"] = df_rank_user2_round2["triplet_result"].apply(lambda x: eval(x))
    else:
        df_rank_user2_round2 = load_dataset("TheMrguiller/Paraphrase_Evaluation_Round_2_U1")["train"].to_pandas()
        df_rank_user2_round2 = process_dataframe_ranking_second_round(df_rank_user2_round2)
        df_rank_user2_round2.to_csv(df_rank_user2_round2_path)
    if os.path.exists(df_rank_user3_round2_path):
        df_rank_user3_round2 = pd.read_csv(project_path+"data/interim/human_eval_paraphrasing_round2/paraphrasing_human_evaluation_user_3.csv")
        df_rank_user3_round2["triplet_result"] = df_rank_user3_round2["triplet_result"].apply(lambda x: eval(x))
    else:
        df_rank_user3_round2 = load_dataset("TheMrguiller/Paraphrase_Evaluation_Round_2_U2")["train"].to_pandas()
        df_rank_user3_round2 = process_dataframe_ranking_second_round(df_rank_user3_round2)
        df_rank_user3_round2.to_csv(df_rank_user3_round2_path)
    
    df_rank_user1_round1 = reorganize_ranking_first_round(df_rank_user1_round1)
    df_rank_user2_round1 = reorganize_ranking_first_round(df_rank_user2_round1)
    df_rank_user3_round1 = reorganize_ranking_first_round(df_rank_user3_round1)

    df_rank_user1 = unify_ranking(df_rank_user1_round1,df_rank_user1_round2)
    df_rank_user2 = unify_ranking(df_rank_user2_round1,df_rank_user2_round2)
    df_rank_user3 = unify_ranking(df_rank_user3_round1,df_rank_user3_round2)
    if not os.path.exists(project_path+"data/processed/human_eval_paraphrasing/"):
            os.makedirs(project_path+"data/processed/human_eval_paraphrasing/")
    df_rank_user1.to_csv(project_path+"data/processed/human_eval_paraphrasing/paraphrasing_human_evaluation_user_1.csv")
    df_rank_user2.to_csv(project_path+"data/processed/human_eval_paraphrasing/paraphrasing_human_evaluation_user_2.csv")
    df_rank_user3.to_csv(project_path+"data/processed/human_eval_paraphrasing/paraphrasing_human_evaluation_user_3.csv")
    annotator_1 = []
    annotator_2 = []
    annotator_3 = []
    for index in range(len(df_rank_user1)):
        rank_user1_values = df_rank_user1["ranking"].loc[index]
        rank_user2_values = df_rank_user2["ranking"].loc[index]
        rank_user3_values = df_rank_user3["ranking"].loc[index]
        for key in rank_user1_values.keys():
            annotator_1.append(rank_user1_values[key])
            annotator_2.append(rank_user2_values[key])
            annotator_3.append(rank_user3_values[key])
        
    annotations = list(zip(annotator_1,annotator_2,annotator_3))
    annotations = np.array(annotations)
    fleish_kappa = irr.fleiss_kappa(irr.aggregate_raters(annotations)[0], method='fleiss')
    print(f"Fleiss Kappa base annotations style: {fleish_kappa}")
    cohan_kappa = cohen_kappa_score(annotator_1,annotator_2)
    print(f"Cohen Kappa annotator 1 and annotator 2: {cohan_kappa}")
    cohan_kappa = cohen_kappa_score(annotator_1,annotator_3)
    print(f"Cohen Kappa annotator 1 and annotator 3: {cohan_kappa}")
    cohan_kappa = cohen_kappa_score(annotator_2,annotator_3)
    print(f"Cohen Kappa annotator 2 and annotator 3: {cohan_kappa}")
    # https://stats.stackexchange.com/questions/641742/what-is-the-best-way-of-combining-multiple-rankings-into-one
    df_rank_user1["borda_count"] = df_rank_user1["ranking"].apply(lambda x: borda_count(x))
    df_rank_user2["borda_count"] = df_rank_user2["ranking"].apply(lambda x: borda_count(x))
    df_rank_user3["borda_count"] = df_rank_user3["ranking"].apply(lambda x: borda_count(x))
    df_majority = pd.DataFrame(columns=["idx","ranking"])
    for index in range(len(df_rank_user1)):
        idx = df_rank_user1["idx"].loc[index]
        borda_count_values_user1 = df_rank_user1[df_rank_user1["idx"]==idx]["borda_count"].values[0]
        borda_count_values_user2 = df_rank_user2[df_rank_user2["idx"]==idx]["borda_count"].values[0]
        borda_count_values_user3 = df_rank_user3[df_rank_user3["idx"]==idx]["borda_count"].values[0]
        borda_count_values = {}
        for key in borda_count_values_user1.keys():
            borda_count_values[key] = borda_count_values_user1[key]+borda_count_values_user2[key]+borda_count_values_user3[key]
        borda_count_values = {k: v for k, v in sorted(borda_count_values.items(), key=lambda item: item[1],reverse=True)}
        ranked_borda_count = rank_models(borda_count_values)
        df_majority.loc[len(df_majority)] = [idx,ranked_borda_count]
    df_majority.to_csv(project_path+"data/processed/human_eval_paraphrasing/paraphrasing_human_evaluation_majority_voting.csv",index=False)

        # ranking_per_source(df_ranking,"APPDIA")
        # ranking_per_source(df_ranking,"paradetox")
        # ranking_per_source(df_ranking,"parallel_detoxification")
        
        
    
                
            
