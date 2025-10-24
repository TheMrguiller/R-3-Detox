from datasets import load_dataset
import pandas as pd
import os
import sys
project_path = os.path.abspath(__file__).split('src')[0]
sys.path.append(project_path)
from src.data.aggregate_judge_llm_predictions import obtain_best_models
import numpy as np
from collections import Counter
from statsmodels.stats import inter_rater as irr
import itertools
import random
from tqdm import tqdm

#Borda count 
def compute_majority_vot(ranks,model_list):
    model_dict = {}
    for model in model_list:
        model_dict[model] = 0
    for rating in ranks:
        for individual_rating in rating:
            points = 0
            if individual_rating["rank"]==1:
                points = 3
            elif individual_rating["rank"]==2:
                points = 2
            elif individual_rating["rank"]==3:
                points = 1
            model_dict[individual_rating["value"]]+=points
    if model_dict[model_list[0]]==model_dict[model_list[1]] and model_dict[model_list[1]]==model_dict[model_list[2]]:
        return model_dict
    return model_dict
    
def extract_unique_users_dataframes(df_grouped,users):
    dfs = []
    rank_to_model_reference = {
        "response-1": "modelA",
        "response-2": "modelB",
        "response-3": "modelC"
    }
    df_rank_user1= pd.DataFrame(columns=["idx","result_triplet0","result_triplet1","result_triplet2"])
    df_rank_user2= pd.DataFrame(columns=["idx","result_triplet0","result_triplet1","result_triplet2"])
    df_rank_user3= pd.DataFrame(columns=["idx","result_triplet0","result_triplet1","result_triplet2"])
    df_second_round = pd.DataFrame(columns=["idx","prompt","modelAparaphrase","modelBparaphrase","modelCparaphrase","modelA","modelB","modelC","source","user"])
    for idx,group in tqdm(df_grouped):
        user_1_triplets = []
        user_2_triplets = []
        user_3_triplets = []
        winners_user1 = []
        winners_user2 = []
        winners_user3 = []
        for i,(_,single_annotation) in enumerate(group.iterrows()):
            idx = single_annotation["metadata.idx"]
            modelA_paraphrase = single_annotation["ModelA"]
            modelB_paraphrase = single_annotation["ModelB"]
            modelC_paraphrase = single_annotation["ModelC"]
            modelA = single_annotation["metadata.ModelA_name"][0]
            modelB = single_annotation["metadata.ModelB_name"][0]
            modelC = single_annotation["metadata.ModelC_name"][0]
            annotator_list = np.array(single_annotation["paraphrase_response_rating.responses.users"])
            ranks = single_annotation["paraphrase_response_rating.responses"]
            for index_ranks,rank in enumerate(ranks):
                for index_rank,element in enumerate(rank):
                    if element["value"]=="response-1":
                        ranks[index_ranks][index_rank]["value"] = modelA
                    elif element["value"]=="response-2":
                        ranks[index_ranks][index_rank]["value"] = modelB
                    else:
                        ranks[index_ranks][index_rank]["value"] = modelC
            
            
            for annotations_index,rank in enumerate(ranks):
                    value_index=users.index(annotator_list[annotations_index])
                    ordered_rank = sorted(rank,key=lambda x: x["rank"],reverse=True)
                    if value_index==0:
                        user_1_triplets.append(ordered_rank)
                        winner_partial = []
                        for element in ordered_rank:
                            if element["rank"]==1:
                                winner_partial.append(element)
                        winner = random.choice(winner_partial)
                        winners_user1.append({
                            "paraphrase":modelA_paraphrase if winner["value"]==modelA else modelB_paraphrase if winner["value"]==modelB else modelC_paraphrase,
                            "model_name":winner["value"]
                        })
                    elif value_index==1:
                        user_2_triplets.append(ordered_rank)
                        winner_partial = []
                        for element in ordered_rank:
                            if element["rank"]==1:
                                winner_partial.append(element)
                        winner = random.choice(winner_partial)
                        winners_user2.append({
                            "paraphrase":modelA_paraphrase if winner["value"]==modelA else modelB_paraphrase if winner["value"]==modelB else modelC_paraphrase,
                            "model_name":winner["value"]
                        })
                    elif value_index==2:
                        user_3_triplets.append(ordered_rank)
                        winner_partial = []
                        for element in ordered_rank:
                            if element["rank"]==1:
                                winner_partial.append(element)
                        winner = random.choice(winner_partial)
                        winners_user3.append({
                            "paraphrase":modelA_paraphrase if winner["value"]==modelA else modelB_paraphrase if winner["value"]==modelB else modelC_paraphrase,
                            "model_name":winner["value"]
                        })
                        pass
        df_rank_user1.loc[len(df_rank_user1)] = [idx,user_1_triplets[0],user_1_triplets[1],user_1_triplets[2]]
        df_rank_user2.loc[len(df_rank_user2)] = [idx,user_2_triplets[0],user_2_triplets[1],user_2_triplets[2]]
        df_rank_user3.loc[len(df_rank_user3)] = [idx,user_3_triplets[0],user_3_triplets[1],user_3_triplets[2]]
        for user, winners in zip(users,[winners_user1,winners_user2,winners_user3]):
            winners = random.sample(winners,3)
            df_second_round.loc[len(df_second_round)] = [idx,single_annotation["prompt"],winners[0]["paraphrase"],winners[1]["paraphrase"],winners[2]["paraphrase"],winners[0]["model_name"],winners[1]["model_name"],winners[2]["model_name"],single_annotation["metadata.source"][0],user]
        
    return df_rank_user1,df_rank_user2,df_rank_user3,df_second_round
                

if __name__ == "__main__":

    annotations_path = project_path+"data/interim/humman_annotations_paraphrase_round1/Paraphrase_Evaluation_Round_1.json"
    if os.path.exists(annotations_path):
        df = pd.read_json(annotations_path)
    else:
        ds =load_dataset("TheMrguiller/Paraphrase_Evaluation_Round_1")
        df=ds["train"].to_pandas()
        if not os.path.exists(project_path+"data/interim/humman_annotations_paraphrase_round1/"):
            os.makedirs(project_path+"data/interim/humman_annotations_paraphrase_round1/")
        df.to_json(annotations_path)
    users = [df["paraphrase_response_rating.responses.users"].iloc[0][0],
        df["paraphrase_response_rating.responses.users"].iloc[0][1],
        df["paraphrase_response_rating.responses.users"].iloc[0][2]
    ]
    df_grouped = df.groupby("metadata.idx")
    #TODO: Per user store their unique annotations
    df_rank_user1,df_rank_user2,df_rank_user3,df_second_round = extract_unique_users_dataframes(df_grouped,users)
    # annotator_1 = df_triplets_user_1["rank"].tolist()
    # annotator_2 = df_triplets_user_2["rank"].tolist()
    # annotator_3 = df_triplets_user_3["rank"].tolist()
    # annotations = list(zip(annotator_1,annotator_2,annotator_3))
    # annotations = np.array(annotations)
    # fleish_kappa = irr.fleiss_kappa(irr.aggregate_raters(annotations)[0], method='fleiss')
    # print(f"Fleiss Kappa base annotations style: {fleish_kappa}")
    
    # df_majority.to_csv(project_path+"data/interim/human_eval_paraphrasing/paraphrasing_human_evaluation_majority_voting.csv",index=False)
    df_second_round.to_csv(project_path+"data/interim/human_eval_paraphrasing/paraphrasing_human_evaluation_dataset_2.csv",index=False)
    if not os.path.exists(project_path+"data/interim/human_eval_paraphrasing_round1/"):
        os.makedirs(project_path+"data/interim/human_eval_paraphrasing_round1/")
    df_rank_user1.to_csv(project_path+"data/interim/human_eval_paraphrasing_round1/paraphrasing_human_evaluation_user_1.csv",index=False)
    df_rank_user2.to_csv(project_path+"data/interim/human_eval_paraphrasing_round1/paraphrasing_human_evaluation_user_2.csv",index=False)
    df_rank_user3.to_csv(project_path+"data/interim/human_eval_paraphrasing_round1/paraphrasing_human_evaluation_user_3.csv",index=False)
    