import pandas as pd
import os
import glob
import argparse
project_path=os.path.abspath(__file__).split('src')[0]
import sys
import numpy as np

def borda_count(triplet):
    model_dict = {}
    for i in range(len(triplet)):

        if triplet[i]["rank"]==1:
            model_dict[triplet[i]["value"]] = 3
        elif triplet[i]["rank"]==2:
            model_dict[triplet[i]["value"]] = 2
        elif triplet[i]["rank"]==3:
            model_dict[triplet[i]["value"]] = 1
    model_dict = dict(sorted(model_dict.items(), key=lambda item: item[1],reverse=True))
    return model_dict

def get_metric(path_to_reference_free_metrics:str,path_to_reference:str):
    df_reference_free = pd.read_csv(path_to_reference_free_metrics)
    df_reference = pd.read_csv(path_to_reference)
    df_reference.drop(columns=['bert_scores',"rouge_scores","bleu_scores","source"], inplace=True)
    df = pd.merge(df_reference_free, df_reference, on='idx')
    columns = df.columns.tolist()
    columns.remove("idx")
    columns.remove('source')
    for column in columns:
        df[column] = df[column].apply(lambda x: float(x))
    return df

if __name__ == "__main__":

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

    df_dict = {
        "paradetox": get_metric(project_path+'results/metrics/paraphrase_automatic_metrics/paradetox/bart/reference_free_metrics.csv',project_path+'results/metrics/paraphrase_automatic_metrics/paradetox/bart/reference_metrics.csv'),
        "detoxllm":get_metric(project_path+'results/metrics/paraphrase_automatic_metrics/detoxllm/detoxllm/reference_free_metrics.csv',project_path+'results/metrics/paraphrase_automatic_metrics/detoxllm/detoxllm/reference_metrics.csv'),
        "llama3_1_8B":get_metric(project_path+'results/metrics/paraphrase_automatic_metrics/llama3_1_8B/experiment_few_shot_7_results/reference_free_metrics.csv',project_path+'results/metrics/paraphrase_automatic_metrics/llama3_1_8B/experiment_few_shot_7_results/reference_metrics.csv'),
        "marco-o1": get_metric(project_path+'results/metrics/paraphrase_automatic_metrics/marco-o1/experiment_few_shot_10_results/reference_free_metrics.csv',project_path+'results/metrics/paraphrase_automatic_metrics/marco-o1/experiment_few_shot_10_results/reference_metrics.csv'),
        "openO1":get_metric(project_path+'results/metrics/paraphrase_automatic_metrics/openO1/experiment_few_shot_10_results/reference_free_metrics.csv',project_path+'results/metrics/paraphrase_automatic_metrics/openO1/experiment_few_shot_10_results/reference_metrics.csv'),
        "pseudoparadetox":get_metric(project_path+'results/metrics/paraphrase_automatic_metrics/pseudoparadetox/pseudoparadetox/reference_free_metrics.csv',project_path+'results/metrics/paraphrase_automatic_metrics/pseudoparadetox/pseudoparadetox/reference_metrics.csv'),
        "qwen2_5_7B":get_metric(project_path+'results/metrics/paraphrase_automatic_metrics/qwen2_5_7B/experiment_few_shot_7_results/reference_free_metrics.csv',project_path+'results/metrics/paraphrase_automatic_metrics/qwen2_5_7B/experiment_few_shot_7_results/reference_metrics.csv'),
        "qwq_preview":get_metric(project_path+'results/metrics/paraphrase_automatic_metrics/qwq_preview/experiment_few_shot_10_results/reference_free_metrics.csv',project_path+'results/metrics/paraphrase_automatic_metrics/qwq_preview/experiment_few_shot_10_results/reference_metrics.csv'),
        "human":pd.read_csv(project_path+'results/metrics/paraphrase_automatic_metrics/human/human/metrics.csv')
    }

    for key in df_dict.keys():
        idxs = df_rank_user1_round1["idx"].tolist()
        df_dict[key] = df_dict[key][df_dict[key]["idx"].isin(idxs)]
    
    disagreement__triplets= []
    content_similarity_whole_dataset = []
    bert_score_whole_dataset = []
    toxic_score_whole_dataset = []
    style_transfer_whole_dataset = []
    for i in range(len(df_rank_user1_round1)):
        idx = df_rank_user1_round1["idx"].iloc[i]
        triplet0_user1 = df_rank_user1_round1["result_triplet0"].iloc[i]
        triplet0_user1_borda = borda_count(triplet0_user1)
        triplet0_user2 = df_rank_user2_round1["result_triplet0"].iloc[i]
        triplet0_user2_borda = borda_count(triplet0_user2)
        triplet0_user3 = df_rank_user3_round1["result_triplet0"].iloc[i]
        triplet0_user3_borda = borda_count(triplet0_user3)
        triplet1_user1 = df_rank_user1_round1["result_triplet1"].iloc[i]
        triplet1_user1_borda = borda_count(triplet1_user1)
        triplet1_user2 = df_rank_user2_round1["result_triplet1"].iloc[i]
        triplet1_user2_borda = borda_count(triplet1_user2)
        triplet1_user3 = df_rank_user3_round1["result_triplet1"].iloc[i]
        triplet1_user3_borda = borda_count(triplet1_user3)
        triplet2_user1 = df_rank_user1_round1["result_triplet2"].iloc[i]
        triplet2_user1_borda = borda_count(triplet2_user1)
        triplet2_user2 = df_rank_user2_round1["result_triplet2"].iloc[i]
        triplet2_user2_borda = borda_count(triplet2_user2)
        triplet2_user3 = df_rank_user3_round1["result_triplet2"].iloc[i]
        triplet2_user3_borda = borda_count(triplet2_user3)
        sum_models_0 = []
        for key in triplet0_user1_borda.keys():
            sum_models_0.append(triplet0_user1_borda[key]+triplet0_user2_borda[key]+triplet0_user3_borda[key])
        sum_models_1 = []
        for key in triplet1_user1_borda.keys():
            sum_models_1.append(triplet1_user1_borda[key]+triplet1_user2_borda[key]+triplet1_user3_borda[key])
        sum_models_2 = []
        for key in triplet2_user1_borda.keys():
            sum_models_2.append(triplet2_user1_borda[key]+triplet2_user2_borda[key]+triplet2_user3_borda[key])
        if sum_models_0[0] == sum_models_0[1] and sum_models_0[1] == sum_models_0[2]:
             disagreement__triplets.append((idx,0,list(triplet0_user1_borda.keys()),[triplet0_user1_borda,triplet0_user2_borda,triplet0_user3_borda]))
        else:
            triplet0_user1_models = list(triplet0_user1_borda.keys())
            for key in triplet0_user1_models:
                content_similarity_whole_dataset.append((idx,key,df_dict[key]["content_similarities"].loc[idx]))
                bert_score_whole_dataset.append((idx,key,df_dict[key]["bert_scores"].loc[idx]))
                toxic_score_whole_dataset.append((idx,key,df_dict[key]["toxic_scores"].loc[idx]))
            style_transfer_whole_dataset.append((idx,key,df_dict[key]["style_transfer_scores"].loc[idx]))
        if sum_models_1[0] == sum_models_1[1] and sum_models_1[1] == sum_models_1[2]:
            disagreement__triplets.append((idx,1,list(triplet1_user1_borda.keys()),[triplet1_user1_borda,triplet1_user2_borda,triplet1_user3_borda]))
        else:
            triplet1_user1_models = list(triplet1_user1_borda.keys())
            for key in triplet1_user1_models:
                content_similarity_whole_dataset.append((idx,key,df_dict[key]["content_similarities"].loc[idx]))
                bert_score_whole_dataset.append((idx,key,df_dict[key]["bert_scores"].loc[idx]))
                toxic_score_whole_dataset.append((idx,key,df_dict[key]["toxic_scores"].loc[idx]))
                style_transfer_whole_dataset.append((idx,key,df_dict[key]["style_transfer_scores"].loc[idx]))
        if sum_models_2[0] == sum_models_2[1] and sum_models_2[1] == sum_models_2[2]:
            disagreement__triplets.append((idx,2,list(triplet2_user1_borda.keys()),[triplet2_user1_borda,triplet2_user2_borda,triplet2_user3_borda]))
        else:
            triplet2_user1_models = list(triplet2_user1_borda.keys())
            for key in triplet2_user1_models:
                content_similarity_whole_dataset.append((idx,key,df_dict[key]["content_similarities"].loc[idx]))
                bert_score_whole_dataset.append((idx,key,df_dict[key]["bert_scores"].loc[idx]))
                toxic_score_whole_dataset.append((idx,key,df_dict[key]["toxic_scores"].loc[idx]))
                style_transfer_whole_dataset.append((idx,key,df_dict[key]["style_transfer_scores"].loc[idx]))

    content_similarity = []
    bert_score = []
    toxic_score = []
    style_transfer = []
    for disagreement in disagreement__triplets:
        idx = disagreement[0]
        triplet = disagreement[1]
        keys = disagreement[2]
        for key in keys:
            content_similarity.append((idx,key,df_dict[key]["content_similarities"].loc[idx]))
            bert_score.append((idx,key,df_dict[key]["bert_scores"].loc[idx]))
            toxic_score.append((idx,key,df_dict[key]["toxic_scores"].loc[idx]))
            style_transfer.append((idx,key,df_dict[key]["style_transfer_scores"].loc[idx]))
    
    print("Content similarity discrepancy")
    print(np.median([float(x[2]) for x in content_similarity]))
    print(np.std([float(x[2]) for x in content_similarity]))
    print("Bert score discrepancy")
    print(np.mean([float(x[2]) for x in bert_score]))
    print(np.std([float(x[2]) for x in bert_score]))
    print("Toxic score discrepancy")
    print(np.mean([float(x[2]) for x in toxic_score]))
    print(np.std([float(x[2]) for x in toxic_score]))
    print("Style transfer discrepancy")
    print(np.mean([float(x[2]) for x in style_transfer]))
    print(np.std([float(x[2]) for x in style_transfer]))
    print("Content similarity whole dataset")
    print(np.mean([float(x[2]) for x in content_similarity_whole_dataset]))
    print(np.std([float(x[2]) for x in content_similarity_whole_dataset]))
    print("Bert score whole dataset")
    print(np.mean([float(x[2]) for x in bert_score_whole_dataset]))
    print(np.std([float(x[2]) for x in bert_score_whole_dataset]))
    print("Toxic score whole dataset")
    print(np.mean([float(x[2]) for x in toxic_score_whole_dataset]))
    print(np.std([float(x[2]) for x in toxic_score_whole_dataset]))
    print("Style transfer whole dataset")
    print(np.mean([float(x[2]) for x in style_transfer_whole_dataset]))
    print(np.std([float(x[2]) for x in style_transfer_whole_dataset]))
    print("Disagreement triplets")

        

