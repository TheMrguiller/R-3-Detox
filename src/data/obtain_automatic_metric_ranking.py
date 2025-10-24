import os
import glob
import argparse
project_path=os.path.abspath(__file__).split('src')[0]
import sys
sys.path.append(project_path)
from typing import List
import random
from tqdm import tqdm
import pandas as pd
from src.evaluation.metrics import ReferenceFreeExperimentMetric
from src.evaluation.toxicity import ToxicMetric
from scipy.optimize import linprog

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

def generate_ranking_per_evaluation(result:str,model_names:List[str],rank:int):
    final_result = {}
    result_base = result.split(" ")
    result= []
    for i in range(len(result_base)):
        result.append(int(result_base[i]))
    for i in range(len(result)):
        final_result[model_names[i]] = result[i]
    sorted_result = sorted(final_result.items(), key=lambda x: x, reverse=False)
    ranked_result = {}
    max_value = sorted_result[0][1]
    for i in range(len(sorted_result)):
        if sorted_result[i][1] == max_value:
            ranked_result[sorted_result[i][0]] = rank
        else:
            rank+=1
            max_value = sorted_result[i][1]
            ranked_result[sorted_result[i][0]] = rank
    return ranked_result


def rank_models_fractional(data: dict,reversed=True):
    # Sorting the dictionary by values in descending order
    sorted_data = sorted(data.items(), key=lambda item: item[1], reverse=reversed)

    # Initialize variables
    ranked_data = {}
    i = 0

    while i < len(sorted_data):
        # Find the group of tied values
        start = i
        while i + 1 < len(sorted_data) and sorted_data[i][1] == sorted_data[i + 1][1]:
            i += 1
        
        # Compute the fractional rank as the average of the ranks of this group
        fractional_rank = (start + 1 + i + 1) / 2

        # Assign the fractional rank to all items in the group
        for j in range(start, i + 1):
            ranked_data[sorted_data[j][0]] = fractional_rank

        # Move to the next group
        i += 1
    return ranked_data

def rank_by_score(data:dict,reversed=True,rank:int=1):
    sorted_data = sorted(data.items(), key=lambda item: item[1], reverse=reversed)
    ranked_data = {}
    for i in range(len(sorted_data)):
        ranked_data[sorted_data[i][0]] = rank
        rank+=1
    return ranked_data

def rank_by_bert_score(bertscores:List[float],model_names:List[str],rank:int=1):
    bert_score_dict = dict(zip(model_names,bertscores))
    ranks = rank_by_score(bert_score_dict,rank=rank)
    return ranks
def rank_by_bleu_score(bleuscores:List[float],model_names:List[str],rank:int=1):
    bleu_score_dict = dict(zip(model_names,bleuscores))
    ranks = rank_by_score(bleu_score_dict,rank=rank)
    return ranks
def rank_by_toxicity_score(toxicity_scores:List[float],model_names:List[str],rank:int=1):
    toxicity_score_dict = dict(zip(model_names,toxicity_scores))
    ranks = rank_by_score(toxicity_score_dict,reversed=False,rank=rank)
    return ranks

def compute_joint_score(content_similarities:List[float],style_transfer_scores:List[float],fluency_scores:List[float]):
    joint_scores = []
    for i in range(len(content_similarities)):
        joint_scores.append(content_similarities[i]*style_transfer_scores[i]*fluency_scores[i])
    return joint_scores

def rank_by_joint_score(joint_scores:List[float],model_names:List[str],rank:int=1):
    joint_score_dict = dict(zip(model_names,joint_scores))
    ranks = rank_by_score(joint_score_dict,rank=rank)
    return ranks

def get_top_rows(df, criteria, top_n=3, weights=None):
    """
    Get the top N rows from the DataFrame that satisfy the optimization criteria, with optional weighting.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        criteria (dict): A dictionary specifying the criteria for each column. 
                         Keys are column names, and values are either 'max' or 'min'.
        top_n (int): The number of top rows to retrieve.
        weights (dict): Optional. A dictionary specifying the weight for each column. 
                        Keys are column names, and values are the weights (default is 1 for all).

    Returns:
        pd.DataFrame: The top N rows that satisfy the criteria.
    """
    if weights is None:
        weights = {col: 1.0 for col in criteria.keys()}  # Default weight of 1.0 for all columns

    # Ensure all criteria columns exist in the DataFrame
    for col in criteria.keys():
        if col not in df.columns:
            raise ValueError(f"Column '{col}' in criteria is not in the DataFrame.")

    # Convert criteria and weights into a weighted objective
    objective = []
    for col in df.columns:
        if col in criteria:
            weight = weights.get(col, 1.0)  # Default weight is 1.0 if not specified
            if criteria[col] == 'max':
                objective.append(-weight * df[col].values)  # Negate for maximization
            elif criteria[col] == 'min':
                objective.append(weight * df[col].values)  # Positive for minimization
            else:
                raise ValueError("Criteria values must be either 'max' or 'min'.")
        else:
            continue

    # Flatten the objective to match row selection
    c = sum(objective)

    # Constraint: At most `top_n` rows can be selected
    A_eq = [[1] * len(df)]  # Sum of selected rows must be 1
    b_eq = [1]
    bounds = [(0, 1) for _ in range(len(df))]  # Binary variables for row selection

    top_rows = []
    available_rows = df.copy()

    for _ in range(top_n):
        # Run the optimization
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if result.success:
            selected_row_index = result.x.argmax()
            top_rows.append(available_rows.iloc[selected_row_index])
            available_rows = available_rows.drop(index=available_rows.index[selected_row_index])

            # Update bounds and objective for remaining rows
            c = sum([(-weights.get(col, 1.0) * available_rows[col].values if criteria.get(col) == 'max'
                      else weights.get(col, 1.0) * available_rows[col].values)
                     for col in criteria.keys()])
            A_eq = [[1] * len(available_rows)]
            b_eq = [1]
            bounds = [(0, 1) for _ in range(len(available_rows))]
        else:
            break

    return pd.DataFrame(top_rows)

def joint_metrics_ranking(df:pd.DataFrame,criteria:dict,weights:dict,rank:int=1):
    ordered_ranking=get_top_rows(df, criteria, top_n=len(df), weights=weights)
    ranking = {}
    for i in range(len(ordered_ranking)):
        ranking[ordered_ranking.iloc[i]['model_name']] = i+1
    ranks = rank_by_score(ranking,reversed=False,rank=rank)
    return ranks    

def get_metric(path_to_reference_free_metrics:str,path_to_reference:str):
    df_reference_free = pd.read_csv(path_to_reference_free_metrics)
    df_reference = pd.read_csv(path_to_reference)
    df_reference.drop(columns=['bert_scores',"rouge_scores","bleu_scores","source"], inplace=True)
    df = pd.merge(df_reference_free, df_reference, on='idx')
    return df

# def obtain_ranking_results(rankings:List[dict],models:List[str]):
#     models_dicts = {}
#     for model in models:
#         models_dicts[model] = {}
#     for ranking in rankings:
#         for model in models:
#             if str(ranking[model]) not in models_dicts[model]:
#                 models_dicts[model][str(ranking[model])] = 1
#             else:
#                 models_dicts[model][str(ranking[model])] += 1
#     return models_dicts

def get_best_result_models(ranking:dict):
    min_value = min(ranking.values())
    best_models = []
    for model in ranking.keys():
        if ranking[model] == min_value:
            best_models.append(model)
    best_model = random.choice(best_models)
    return best_model

def compute_final_rank(first_round,second_round):
    for model_name in second_round.keys():
        for idx,first_round_triplet in enumerate(first_round):
            if model_name in first_round_triplet.keys():
                rank_new = second_round[model_name]
                rank_old = first_round_triplet[model_name]
                for model_name_first_round in first_round_triplet.keys():
                    if first_round_triplet[model_name_first_round]==rank_old:
                        first_round[idx][model_name_first_round]=rank_new
    
    for idx,first_round_triplet in enumerate(first_round):
        for model_name in first_round_triplet.keys():
            if model_name not in second_round.keys():
                second_round[model_name] = first_round_triplet[model_name]
    return second_round

def obtain_ranks_based_on_source(model_names,source,df):
    df = df[df["source"]==source]
    df.reset_index(drop=True,inplace=True)
    toxic_rank = obtain_ranking_results(df["toxic_ranks"].to_list(),model_names)
    toxic_rank = sorted(toxic_rank.items(), key=lambda x: x[1], reverse=True)
    bert_rank = obtain_ranking_results(df["bert_ranks"].to_list(),model_names)
    bert_rank = sorted(bert_rank.items(), key=lambda x: x[1], reverse=True)
    joint_rank = obtain_ranking_results(df["joint_ranks"].to_list(),model_names)
    joint_rank = sorted(joint_rank.items(), key=lambda x: x[1], reverse=True)
    bleu_rank = obtain_ranking_results(df["bleu_ranks"].to_list(),model_names)
    bleu_rank = sorted(bleu_rank.items(), key=lambda x: x[1], reverse=True)
    joint_metrics_rank = obtain_ranking_results(df["joint_metrics_ranks"].to_list(),model_names)
    joint_metrics_rank = sorted(joint_metrics_rank.items(), key=lambda x: x[1], reverse=True)
    output_path_report_best_rankings = project_path+'results/reports/ranking_reports/'
    if not os.path.exists(output_path_report_best_rankings):
        os.makedirs(output_path_report_best_rankings)
    with open(output_path_report_best_rankings+f'{source}_toxic_ranks.txt','w') as f:
        f.write(str(toxic_rank))
    with open(output_path_report_best_rankings+f'{source}_bert_ranks.txt','w') as f:
        f.write(str(bert_rank))
    with open(output_path_report_best_rankings+f'{source}_joint_ranks.txt','w') as f:
        f.write(str(joint_rank))
    with open(output_path_report_best_rankings+f'{source}_bleu_ranks.txt','w') as f:
        f.write(str(bleu_rank))
    with open(output_path_report_best_rankings+f'{source}_joint_metrics_ranks.txt','w') as f:
        f.write(str(joint_metrics_rank))
if __name__ == "__main__":
    df = pd.read_csv(project_path+'data/interim/paraphrasing_eval_dataset/paraphrasing_evaluation_dataset.csv')
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
    indexes=df["idx"].to_list()
    model_names = []
    criteria = {
            'toxic_scores': 'min',
            'bert_scores': 'max',
            'joint_scores': 'max',
            'bleu_scores': 'max'
        }
    weights = {
        'toxic_scores': 3,
        'bert_scores': 1,
        'joint_scores': 2,
        'bleu_scores': 1
    }
    df_ranks_automatic_metrics = pd.DataFrame(columns=["idx","source","toxic_ranks","bert_ranks","joint_ranks","bleu_ranks","joint_metrics_ranks"])
    #TODO: Eliminate the ones that are blank with no values in each of the models
    # for idx, group in tqdm(df.groupby("idx")):
    #     toxic_rankings = []
    #     bert_rankings = []
    #     joint_rankings = []
    #     bleu_rankings = []
    #     joint_metrics_rankings = []
    #     for i in range(len(group)):
    #         model_names = [group.iloc[i]["modelA"],group.iloc[i]["modelB"],group.iloc[i]["modelC"]]
    #         if idx in df_dict[model_names[0]]["idx"].to_list():
    #             toxic_scores = [df_dict[model_name][df_dict[model_name]["idx"]==idx]["toxic_scores"].values[0] for model_name in model_names]
    #             bert_scores = [df_dict[model_name][df_dict[model_name]["idx"]==idx]["bert_scores"].values[0] for model_name in model_names]
    #             content_similarities = [df_dict[model_name][df_dict[model_name]["idx"]==idx]["content_similarities"].values[0] for model_name in model_names]
    #             style_transfer_scores = [df_dict[model_name][df_dict[model_name]["idx"]==idx]["style_transfer_scores"].values[0] for model_name in model_names]
    #             fluency_scores = [df_dict[model_name][df_dict[model_name]["idx"]==idx]["fluency_scores"].values[0] for model_name in model_names]
    #             bleu_scores = [df_dict[model_name][df_dict[model_name]["idx"]==idx]["bleu_scores"].values[0] for model_name in model_names]
    #             toxicity_ranks = rank_by_toxicity_score(toxic_scores,model_names,rank=3)
    #             bert_ranks = rank_by_bert_score(bert_scores,model_names,rank=3)
    #             joint_scores = compute_joint_score(content_similarities,style_transfer_scores,fluency_scores)
    #             joint_ranks = rank_by_joint_score(joint_scores,model_names,rank=3)
    #             bleu_ranks = rank_by_bleu_score(bleu_scores,model_names,rank=3)
    #             df_joint_metrics = pd.DataFrame(list(zip(model_names,toxic_scores,bert_scores,joint_scores,bleu_scores)),columns=["model_name","toxic_scores","bert_scores","joint_scores","bleu_scores"])
    #             joint_metrics_ranks = joint_metrics_ranking(df_joint_metrics,criteria,weights,rank=3)
    #             toxic_rankings.append(toxicity_ranks)
    #             bert_rankings.append(bert_ranks)
    #             joint_rankings.append(joint_ranks)
    #             bleu_rankings.append(bleu_ranks)
    #             joint_metrics_rankings.append(joint_metrics_ranks)
    #     if len(toxic_rankings)==0:
    #         continue
    #     winners_toxic = [get_best_result_models(toxic_ranking) for toxic_ranking in toxic_rankings]
    #     winners_bert = [get_best_result_models(bert_ranking) for bert_ranking in bert_rankings]
    #     winners_joint = [get_best_result_models(joint_ranking) for joint_ranking in joint_rankings]
    #     winners_bleu = [get_best_result_models(bleu_ranking) for bleu_ranking in bleu_rankings]
    #     winners_joint_metrics = [get_best_result_models(joint_metrics_ranking) for joint_metrics_ranking in joint_metrics_rankings]
        
    #     toxic_scores = [df_dict[model_name][df_dict[model_name]["idx"]==idx]["toxic_scores"].values[0] for model_name in winners_toxic]
    #     bert_scores = [df_dict[model_name][df_dict[model_name]["idx"]==idx]["bert_scores"].values[0] for model_name in winners_bert]
    #     content_similarities = [df_dict[model_name][df_dict[model_name]["idx"]==idx]["content_similarities"].values[0] for model_name in winners_joint]
    #     style_transfer_scores = [df_dict[model_name][df_dict[model_name]["idx"]==idx]["style_transfer_scores"].values[0] for model_name in winners_joint]
    #     fluency_scores = [df_dict[model_name][df_dict[model_name]["idx"]==idx]["fluency_scores"].values[0] for model_name in winners_joint]
    #     bleu_scores = [df_dict[model_name][df_dict[model_name]["idx"]==idx]["bleu_scores"].values[0] for model_name in winners_bleu]
    #     joint_scores = compute_joint_score(content_similarities,style_transfer_scores,fluency_scores)
    #     toxic_ranks = rank_by_toxicity_score(toxic_scores,winners_toxic)
    #     bert_ranks = rank_by_bert_score(bert_scores,winners_bert)
    #     joint_ranks = rank_by_joint_score(joint_scores,winners_joint)
    #     bleu_ranks = rank_by_bleu_score(bleu_scores,winners_bleu)
    #     df_joint_metrics = pd.DataFrame(list(zip(model_names,toxic_scores,bert_scores,joint_scores,bleu_scores)),columns=["model_name","toxic_scores","bert_scores","joint_scores","bleu_scores"])
    #     joint_metrics_ranks = joint_metrics_ranking(df_joint_metrics,criteria,weights)
    #     final_toxic_rank = compute_final_rank(toxic_rankings,toxic_ranks)
    #     final_bert_rank = compute_final_rank(bert_rankings,bert_ranks)
    #     final_joint_rank = compute_final_rank(joint_rankings,joint_ranks)
    #     final_bleu_rank = compute_final_rank(bleu_rankings,bleu_ranks)
    #     final_joint_metrics_rank = compute_final_rank(joint_metrics_rankings,joint_metrics_ranks)
    #     df_ranks_automatic_metrics.loc[len(df_ranks_automatic_metrics)] = [idx,df.loc[df["idx"]==idx]["source"].values[0],final_toxic_rank,final_bert_rank,final_joint_rank,final_bleu_rank,final_joint_metrics_rank]
    # if not os.path.exists(project_path+'results/metrics/paraphrase_automatic_metrics/final/'):
    #     os.makedirs(project_path+'results/metrics/paraphrase_automatic_metrics/final/')
    # df_ranks_automatic_metrics = df_ranks_automatic_metrics[df_ranks_automatic_metrics["bert_ranks"]!={}]
    # df_ranks_automatic_metrics.to_csv(project_path+'results/metrics/paraphrase_automatic_metrics/final/ranks_automatic_metrics.csv',index=False)
    df_ranks_automatic_metrics = pd.read_csv(project_path+'results/metrics/paraphrase_automatic_metrics/final/ranks_automatic_metrics.csv')
    df_ranks_automatic_metrics["bert_ranks"] = df_ranks_automatic_metrics["bert_ranks"].apply(eval)
    df_ranks_automatic_metrics["toxic_ranks"] = df_ranks_automatic_metrics["toxic_ranks"].apply(eval)
    df_ranks_automatic_metrics["joint_ranks"] = df_ranks_automatic_metrics["joint_ranks"].apply(eval)
    df_ranks_automatic_metrics["bleu_ranks"] = df_ranks_automatic_metrics["bleu_ranks"].apply(eval)
    df_ranks_automatic_metrics["joint_metrics_ranks"] = df_ranks_automatic_metrics["joint_metrics_ranks"].apply(eval)

    model_names = list(df_ranks_automatic_metrics.loc[0]["toxic_ranks"].keys())
    obtain_ranks_based_on_source(model_names,"APPDIA",df_ranks_automatic_metrics)
    obtain_ranks_based_on_source(model_names,"paradetox",df_ranks_automatic_metrics)
    obtain_ranks_based_on_source(model_names,"parallel_detoxification",df_ranks_automatic_metrics)
    toxic_rank = obtain_ranking_results(df_ranks_automatic_metrics["toxic_ranks"].to_list(),model_names)
    toxic_rank = sorted(toxic_rank.items(), key=lambda x: x[1], reverse=True)
    bert_rank = obtain_ranking_results(df_ranks_automatic_metrics["bert_ranks"].to_list(),model_names)
    bert_rank = sorted(bert_rank.items(), key=lambda x: x[1], reverse=True)
    joint_rank = obtain_ranking_results(df_ranks_automatic_metrics["joint_ranks"].to_list(),model_names)
    joint_rank = sorted(joint_rank.items(), key=lambda x: x[1], reverse=True)
    bleu_rank = obtain_ranking_results(df_ranks_automatic_metrics["bleu_ranks"].to_list(),model_names)
    bleu_rank = sorted(bleu_rank.items(), key=lambda x: x[1], reverse=True)
    joint_metrics_rank = obtain_ranking_results(df_ranks_automatic_metrics["joint_metrics_ranks"].to_list(),model_names)
    joint_metrics_rank = sorted(joint_metrics_rank.items(), key=lambda x: x[1], reverse=True)
    output_path_report_best_rankings = project_path+'results/reports/ranking_reports/'
    if not os.path.exists(output_path_report_best_rankings):
        os.makedirs(output_path_report_best_rankings)
    with open(output_path_report_best_rankings+f'general_toxic_ranks.txt','w') as f:
        f.write(str(toxic_rank))
    with open(output_path_report_best_rankings+f'general_bert_ranks.txt','w') as f:
        f.write(str(bert_rank))
    with open(output_path_report_best_rankings+f'general_joint_ranks.txt','w') as f:
        f.write(str(joint_rank))
    with open(output_path_report_best_rankings+f'general_bleu_ranks.txt','w') as f:
        f.write(str(bleu_rank))
    with open(output_path_report_best_rankings+f'general_joint_metrics_ranks.txt','w') as f:
        f.write(str(joint_metrics_rank))

    # for model_name in df_dict.keys():
    #         model_names.append(model_name)
    # df_ranks_automatic_metrics = pd.DataFrame(columns=["idx","source","toxic_ranks","bert_ranks","joint_ranks","bleu_ranks","joint_metrics_ranks"])
    # annotation_index = df_dict["paradetox"]["idx"].to_list()
    # for index in tqdm(indexes):
    #     toxic_scores = []
    #     bert_scores = []
    #     bleu_scores = []
    #     content_similarities = []
    #     style_transfer_scores = []
    #     fluency_scores = []
        
    #     for model_name in model_names:
    #         if index in annotation_index:
    #             toxic_scores.append(round(df_dict[model_name][df_dict[model_name]["idx"]==index]["toxic_scores"].values[0],4))
    #             bert_scores.append(round(df_dict[model_name][df_dict[model_name]["idx"]==index]["bert_scores"].values[0],3))
    #             content_similarities.append(round(df_dict[model_name][df_dict[model_name]["idx"]==index]["content_similarities"].values[0],3))
    #             style_transfer_scores.append(df_dict[model_name][df_dict[model_name]["idx"]==index]["style_transfer_scores"].values[0])
    #             fluency_scores.append(df_dict[model_name][df_dict[model_name]["idx"]==index]["fluency_scores"].values[0])
    #             bleu_scores.append(round(df_dict[model_name][df_dict[model_name]["idx"]==index]["bleu_scores"].values[0],3))

    #             # bert_scores.append(df_dict[model_name].loc[index]['bert_scores'])
    #             # content_similarities.append(df_dict[model_name].loc[index]['content_similarities'])
    #             # style_transfer_scores.append(df_dict[model_name].loc[index]['style_transfer_scores'])
    #             # fluency_scores.append(df_dict[model_name].loc[index]['fluency_scores'])
    #             # bleu_scores.append(df_dict[model_name].loc[index]['bleu_scores'])
    #     toxicity_ranks = rank_by_toxicity_score(toxic_scores,model_names)
    #     bert_ranks = rank_by_bert_score(bert_scores,model_names)
    #     joint_scores = compute_joint_score(content_similarities,style_transfer_scores,fluency_scores)
    #     joint_ranks = rank_by_joint_score(joint_scores,model_names)
    #     bleu_ranks = rank_by_bleu_score(bleu_scores,model_names)
    #     df_joint_metrics = pd.DataFrame(list(zip(model_names,toxic_scores,bert_scores,joint_scores,bleu_scores)),columns=["model_name","toxic_scores","bert_scores","joint_scores","bleu_scores"])
    #     joint_metrics_ranks = joint_metrics_ranking(df_joint_metrics,criteria,weights)
    #     df_ranks_automatic_metrics.loc[len(df_ranks_automatic_metrics)] = [index,df.loc[df["idx"]==index]["source"].values[0],toxicity_ranks,bert_ranks,joint_ranks,bleu_ranks,joint_metrics_ranks]
    # if not os.path.exists(project_path+'results/metrics/paraphrase_automatic_metrics/final/'):
    #     os.makedirs(project_path+'results/metrics/paraphrase_automatic_metrics/final/')
    # df_ranks_automatic_metrics = df_ranks_automatic_metrics[df_ranks_automatic_metrics["bert_ranks"]!={}]
    # df_ranks_automatic_metrics.to_csv(project_path+'results/metrics/paraphrase_automatic_metrics/final/ranks_automatic_metrics.csv',index=False)

    # toxic_ranks = df_ranks_automatic_metrics["toxic_ranks"].to_list()
    # bert_ranks = df_ranks_automatic_metrics["bert_ranks"].to_list()
    # joint_ranks = df_ranks_automatic_metrics["joint_ranks"].to_list()
    # bleu_ranks = df_ranks_automatic_metrics["bleu_ranks"].to_list()
    # joint_metrics_ranks = df_ranks_automatic_metrics["joint_metrics_ranks"].to_list()
    # toxic_ranks_dict = obtain_ranking_results(toxic_ranks,model_names)
    # bert_ranks_dict = obtain_ranking_results(bert_ranks,model_names)
    # joint_ranks_dict = obtain_ranking_results(joint_ranks,model_names)
    # bleu_ranks_dict = obtain_ranking_results(bleu_ranks,model_names)
    # joint_metrics_ranks_dict = obtain_ranking_results(joint_metrics_ranks,model_names)
    # output_path_report_best_rankings = project_path+'results/reports/ranking_reports/'
    # if not os.path.exists(output_path_report_best_rankings):
    #     os.makedirs(output_path_report_best_rankings)
    # with open(output_path_report_best_rankings+'toxic_ranks.txt','w') as f:
    #     f.write(str(toxic_ranks_dict))
    # with open(output_path_report_best_rankings+'bert_ranks.txt','w') as f:
    #     f.write(str(bert_ranks_dict))
    # with open(output_path_report_best_rankings+'joint_ranks.txt','w') as f:
    #     f.write(str(joint_ranks_dict))
    # with open(output_path_report_best_rankings+'bleu_ranks.txt','w') as f:
    #     f.write(str(bleu_ranks_dict))
    # with open(output_path_report_best_rankings+'joint_metrics_ranks.txt','w') as f:
    #     f.write(str(joint_metrics_ranks_dict))

        
        

    

            