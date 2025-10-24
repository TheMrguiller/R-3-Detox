from scipy.stats import spearmanr,permutation_test,chi2,pearsonr,friedmanchisquare
import scikit_posthocs as sp
import pandas as pd
import os
import sys
project_path=os.path.abspath(__file__).split('src')[0]
sys.path.append(project_path)
import glob
import argparse
from src.data.human_paraphrase_final_evaluation_ranking import borda_count,rank_models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.evaluation.metric_correlation_reasoning import desaturate_cmap

def rank_models_fractional(data: dict):
    # Sorting the dictionary by values in descending order
    sorted_data = sorted(data.items(), key=lambda item: item[1], reverse=True)

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

    # Sort the ranked data by original input order
    keys = list(data.keys())
    sorted_keys = sorted(keys, key=lambda x: x, reverse=False)
    sorted_ranked_data = {key: ranked_data[key] for key in sorted_keys}
    
    return [value for value in sorted_ranked_data.values()]

def statistic(x, y):
    # Calculate Spearman correlation for the data x, y
    rs = spearmanr(x, y).statistic
    return rs

# Function to transform Pearson r to Fisher Z
def fisher_z(r):
    if r >= 1.0:
        return np.inf  # Fisher transformation is not defined for rho = 1
    elif r <= -1.0:
        return -np.inf  # Fisher transformation is not defined for rho = -1
    else:
        return 0.5 * np.log((1 + r) / (1 - r))


# Function to transform Fisher Z back to Pearson r
def fisher_z_to_r(z):
    if z == np.inf:
        return 1.0  # Return the maximum correlation value
    elif z == -np.inf:
        return -1.0  # Return the minimum correlation value
    else:
        return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

def aggregate_correlation(r_values):

    z_values = [fisher_z(r) for r in r_values]
    z_values = [z if z != np.inf and z != -np.inf else None for z in z_values]
    valid_z_values = [z for z in z_values if z is not None]
    # Calculate the combined Fisher Z
    average_z = np.mean(valid_z_values) if valid_z_values else 0
    # Transform the combined Fisher Z back to Pearson r
    combined_r = fisher_z_to_r(average_z)

    return combined_r

def chi_squared_statistic_compute(p_values):
    chi_squared = 0
    for p_value in p_values:
        # Avoid log(0) or log(negative) by checking if p_value is positive
        if p_value > 0:
            chi_squared += -2 * np.log(p_value)
        else:
            # Handle case where p-value is zero or negative (adjust or skip)
            # Optionally, you could set the chi-squared term to 0 or a small value.
            chi_squared += 0  # Or skip with `continue` depending on your use case
    return chi_squared

def first_params_appearence(names):
    """
    Function to get the first appearance of a model in the list of names
    """
    first_appearence= -1
    for i,name in enumerate(names):
        if "params" in name:
            first_appearence = i
            break
    return first_appearence

def correlation_ponderation_general(dfs,dfs_names,reference_model,reference_model_name):
    """
    Calculate the Spearman correlation
    """
    ranks = []
    df_scores = []
    reference_scores = reference_model["ranking"].apply(lambda x: borda_count(x)).tolist()
    reference_scores_models = {}
    for score in reference_scores:
        for model in score.keys():
            if model not in reference_scores_models:
                reference_scores_models[model] = score[model]
            else:
                reference_scores_models[model] += score[model]
    # rank_reference = rank_models(reference_scores)
    reference_scores_models = dict(sorted(reference_scores_models.items(), key=lambda item: item[1],reverse=True))
    rank_reference = rank_models(reference_scores_models)
    rank_reference = dict(sorted(rank_reference.items(), key=lambda item: item[0]))
    for df in dfs:
        df_score=df["ranking"].apply(lambda x: borda_count(x)).tolist()
        df_scores_models = {}
        for score in df_score:
            for model in score.keys():
                if model not in df_scores_models:
                    df_scores_models[model] = score[model]
                else:
                    df_scores_models[model] += score[model]
        # rank = rank_models(df_score)
        df_scores_models = dict(sorted(df_scores_models.items(), key=lambda item: item[1],reverse=True))
        rank = rank_models(df_scores_models)
        rank = dict(sorted(rank.items(), key=lambda item: item[0]))
        ranks.append(rank)
        df_scores.append(df_scores_models)
    spearman_correlation = []

    for i in range(len(ranks)):
        spearman_r_correlation = statistic(list(rank_reference.values()), list(ranks[i].values()))
        pvalue=permutation_test((list(rank_reference.values()), list(ranks[i].values())), statistic,n_resamples=5000, alternative='two-sided', permutation_type='pairings').pvalue
        spearman_correlation.append([spearman_r_correlation,pvalue])
    index_params = first_params_appearence(dfs_names)
    roscoe_ranks = ranks[:index_params]
    nemenyi_test,nemenyi_test_p_value=friedmanchisquare(*([list(rank_reference.values())]+[list(roscoe_ranks[i].values()) for i in range(len(roscoe_ranks))]))
    print(f"Nemenyi test value {reference_model_name}:Automatic metrics,value {nemenyi_test} ,p-value: {nemenyi_test_p_value}")
    judge_llm_ranks = ranks[index_params:]
    nemenyi_test,nemenyi_test_p_value=friedmanchisquare(*([list(rank_reference.values())]+[list(judge_llm_ranks[i].values()) for i in range(len(judge_llm_ranks))]))
    print(f"Nemenyi test value {reference_model_name}:Params,value {nemenyi_test} ,p-value: {nemenyi_test_p_value}")
    for name,correlation in zip(dfs_names,spearman_correlation):
        print(f"Spearman correlation: {reference_model_name}:{name},correlation: {correlation}")
    models_name = list(reference_scores_models.keys())
    pearson_correlation = []
    for name,scores in zip(dfs_names,df_scores):
        reference_score = [reference_scores_models[model] for model in models_name]
        df_score = [scores[model] for model in models_name]
        correlation, p_value = pearsonr(reference_score, df_score)
        print(f"Pearson correlation: {reference_model_name}:{name},correlation: {correlation}, p_value: {p_value}")
        pearson_correlation.append([correlation,p_value])
    ## Critical difference diagram
    data = np.array([list(rank_reference.values())]+[list(rank.values()) for rank in ranks]).T
    df = pd.DataFrame(data)
    rankmat = df.rank(axis='columns', ascending=True)
    meanranks = rankmat.mean()

    #Call the Nemenyi posthoc test
    result = sp.posthoc_nemenyi_friedman(data)

    #Plot the result
    sp.sign_plot(result)
    name_critical_difference = [reference_model_name]+dfs_names
    meanranks.index = name_critical_difference
    #Here I'm trying to plot the CD diagram, but a warning/error message appears.
    plt.figure(figsize=(10, 2), dpi=100)
    plt.title('Critical difference diagram of average score ranks')
    sp.critical_difference_diagram(meanranks, result)
    plt.savefig(project_path+"results/reports/critical_difference_diagram_paraphrase.png")
    return spearman_correlation,pearson_correlation

def comment_wise_rank(df):

    df_groups = df.groupby("idx")
    df_groups = {group: data for group, data in df_groups}
    ranks = []
    model_scores = []
    for key in df_groups.keys():
        model_score = obtain_best_models(df_groups[key])
        # rank = rank_models(model_score)
        rank = rank_models_fractional(model_score)
        ranks.append(rank)
        model_scores.append(model_score)
    return ranks,model_scores


def correlation_poderation_specific_commentwise(dfs,dfs_names,reference_model,reference_model_name):
    """
    Calculate the Spearman correlation
    """
    
    ranks_dfs = []
    dfs_scores = []
    reference_scores = reference_model["ranking"].apply(lambda x: borda_count(x)).tolist()
    # rank_reference = rank_models(reference_scores)
    ranks_reference= reference_model["ranking"].tolist()
    ranks_reference = [dict(sorted(rank.items(), key=lambda item: item[0])) for rank in ranks_reference]
    for df in dfs:
        df_score=df["ranking"].apply(lambda x: borda_count(x)).tolist()
        # rank = rank_models(df_score)
        rank = df["ranking"].tolist()
        rank = [dict(sorted(rank[i].items(), key=lambda item: item[0])) for i in range(len(rank))]
        
        ranks_dfs.append(rank)
        dfs_scores.append(df_score)

    index_params = first_params_appearence(dfs_names)
    obtain_nemenyi_test_for_metrics(name="Automatic metrics",reference_ranks=ranks_reference,dfs_ranks=ranks_dfs[:index_params],name_reference=reference_model_name)
    obtain_nemenyi_test_for_metrics(name="Params",reference_ranks=ranks_reference,dfs_ranks=ranks_dfs[index_params:],name_reference=reference_model_name)
    for i in range(len(ranks_dfs)):
        spearman_correlation = []
        pearson_correlation = []
        
        for rank_reference,rank_df in zip(ranks_reference,ranks_dfs[i]):
            spearman_r_correlation = statistic(list(rank_reference.values()), list(rank_df.values()))
            pvalue=permutation_test((list(rank_reference.values()), list(rank_df.values())), statistic,n_resamples=5000, alternative='two-sided', permutation_type='pairings').pvalue
            spearman_correlation.append([spearman_r_correlation,pvalue])
        
        for reference_score, df_score in zip(reference_scores,dfs_scores[i]):
            models_name = list(reference_score.keys())
            reference_score = [reference_score[model] for model in models_name]
            df_score_final = [df_score[model] for model in models_name]
            correlation, p_value = pearsonr(reference_score, df_score_final)
            # print(f"Pearson correlation for {reference_model_name}:{dfs_names[i]},correlation: {correlation}, p_value: {p_value}")
            pearson_correlation.append([correlation,p_value])
        spearman_correlation = np.array(spearman_correlation)

        print(f"Spearman correlation for {reference_model_name}:{dfs_names[i]}")
        print(f"Mean Spearman correlation: {sum(spearman_correlation[:, 0])/len(spearman_correlation)}")
        print(f"Median Spearman correlation: {sorted(spearman_correlation[:, 0])[len(spearman_correlation)//2]}")
        print(f"Standard deviation: {np.std(spearman_correlation[:, 0])}")
        print(f"IQR: {np.percentile(spearman_correlation[:, 0],75)-np.percentile(spearman_correlation[:, 0],25)}")
        print(f"Percentile 25: {np.percentile(spearman_correlation[:, 0],25)}")
        print(f"Percentile 75: {np.percentile(spearman_correlation[:, 0],75)}")
        print(f"Max: {max(spearman_correlation[:, 0])}")
        print(f"Min: {min(spearman_correlation[:, 0])}")
        n_tests = len(spearman_correlation[:, 1])
        chi_squared_statistic = chi_squared_statistic_compute(spearman_correlation[:,1])
        p_value_combined = chi2.sf(chi_squared_statistic, 2 * n_tests)
        print("Combined p-value for Spearman ranking correlation using Fisher's method:", p_value_combined)
        spearman_aggregated=aggregate_correlation(spearman_correlation[:, 0])
        print(f"Aggregated Spearman ranking correlation using Fisher's method: {spearman_aggregated}")

        print(f"Pearson correlation for {reference_model_name}:{dfs_names[i]}")
        pearson_correlation = np.array(pearson_correlation)
        print(f"Mean Pearson correlation: {sum(pearson_correlation[:, 0])/len(pearson_correlation)}")
        print(f"Median Pearson correlation: {sorted(pearson_correlation[:, 0])[len(pearson_correlation)//2]}")
        print(f"Standard deviation: {np.std(pearson_correlation[:, 0])}")
        print(f"IQR: {np.percentile(pearson_correlation[:, 0],75)-np.percentile(pearson_correlation[:, 0],25)}")
        print(f"Percentile 25: {np.percentile(pearson_correlation[:, 0],25)}")
        print(f"Percentile 75: {np.percentile(pearson_correlation[:, 0],75)}")
        print(f"Max: {max(pearson_correlation[:, 0])}")
        print(f"Min: {min(pearson_correlation[:, 0])}")
        n_tests = len(pearson_correlation[:, 1])
        chi_squared_statistic = chi_squared_statistic_compute(pearson_correlation[:,1])
        p_value_combined = chi2.sf(chi_squared_statistic, 2 * n_tests)
        print("Combined p-value for Pearson correlation using Fisher's method:", p_value_combined)
        pearson_aggregated=aggregate_correlation(pearson_correlation[:, 0])
        print(f"Aggregated Pearson correlation using Fisher's method: {pearson_aggregated}")
        print("\n")
        
    return spearman_correlation

def obtain_nemenyi_test_for_metrics(name,reference_ranks,dfs_ranks,name_reference="human_annotations"):

    nemenyi_test_list = []
    for i in range(len(reference_ranks)):
        nemenyi_test,nemenyi_test_p_value=friedmanchisquare(*([list(reference_ranks[i].values())]+[list(df_rank[i].values()) for df_rank in dfs_ranks]))
        nemenyi_test_list.append([nemenyi_test,nemenyi_test_p_value])
    nemenyi_test_list = np.array(nemenyi_test_list)
    n_tests = len(nemenyi_test_list[:, 1])
    chi_squared_statistic = chi_squared_statistic_compute(nemenyi_test_list[:,1])
    p_value_combined = chi2.sf(chi_squared_statistic, 2 * n_tests)
    print(f"Combined p-value for Nemenyi test using Fisher's method {name_reference}-{name}:", p_value_combined)
    nemenyi_aggregated=aggregate_correlation(nemenyi_test_list[:, 0])
    print(f"Aggregated Nemenyi test using Fisher's method {name_reference}-{name}: {nemenyi_aggregated}")
    return nemenyi_test_list

from tqdm import tqdm

def obtain_correlation_matrix(dfs,dfs_names):
    """
    Calculate the Spearman correlation matrix
    """
    spearman_correlation = np.zeros((len(dfs),len(dfs)),dtype=float)
    p_values = np.zeros((len(dfs),len(dfs)),dtype=float)
    ranks = []
    df_scores = []
    for df in dfs:
        df_score=df["ranking"].apply(lambda x: borda_count(x)).tolist()
        df_scores_models = {}
        for score in df_score:
            for model in score.keys():
                if model not in df_scores_models:
                    df_scores_models[model] = score[model]
                else:
                    df_scores_models[model] += score[model]
        # rank = rank_models(df_score)
        df_scores_models = dict(sorted(df_scores_models.items(), key=lambda item: item[1],reverse=True))
        rank = rank_models(df_scores_models)
        rank = dict(sorted(rank.items(), key=lambda item: item[0]))
        ranks.append(rank)
        df_scores.append(df_scores_models)


    for i in tqdm(range(len(ranks))):
        for j in range(len(ranks)):
            spearman_r_correlation = statistic(list(ranks[i].values()), list(ranks[j].values()))
            spearman_correlation[i,j] = spearman_r_correlation
            p_value=permutation_test((list(ranks[i].values()), list(ranks[j].values())), statistic,n_resamples=5000, alternative='two-sided', permutation_type='pairings').pvalue
            p_values[i,j] = p_value
    correlation_df = pd.DataFrame(spearman_correlation, index=dfs_names, columns=dfs_names)
    p_values_df = pd.DataFrame(p_values, index=dfs_names, columns=dfs_names)
    # Plot heatmap
    soft_coolwarm = desaturate_cmap(cmap_name="coolwarm_r",blend=0.3)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_df,
        annot=True, fmt=".2f",
        cmap=soft_coolwarm,  # Light background color
        center=0,
        linewidths=0.5,
        cbar=True,
        square=True,
        annot_kws={"color": "black","size": 12}  # Make annotation text black
    )

    # plt.title("Spearman Correlation Matrix", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=14)
    plt.yticks(fontsize=14, rotation=0)

    # Adjust layout to fix space issues
    plt.gcf().tight_layout()
    plt.subplots_adjust(left=0.2, bottom=0.2, top=0.9, right=0.9)
    plt.savefig(project_path+"results/reports/Spearman_Correlation_Matrix_paraphrase.pdf", format='pdf')
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    soft_coolwarm = desaturate_cmap(cmap_name="coolwarm",blend=0.3)
    sns.heatmap(
        p_values_df,
        annot=True, fmt=".2f",
        cmap=soft_coolwarm,  # Light background color
        center=0.05,  # Center of the color transition
        vmin=0, vmax=1,  # Setting the range explicitly
        linewidths=0.5,
        cbar=True,
        square=True,
        annot_kws={"color": "black", "size": 12}  # Make annotation text black
    )
    plt.xticks(rotation=45, ha="right", fontsize=14)
    plt.yticks(fontsize=14, rotation=0)
    plt.gcf().tight_layout()
    plt.subplots_adjust(left=0.2, bottom=0.2, top=0.9, right=0.9)
    plt.savefig(project_path+"results/reports/Spearman_Correlation_Matrix_p_values_paraphrase.pdf", format='pdf')

if __name__ == "__main__":
    # Metric correlation reasoning
    # dfs = []
    # df_metric_name = []
    # for file in glob.glob(project_path+"data/processed/roscoe_aggregated/*.csv"):
    #     df = pd.read_csv(file)
    #     df_metric_name.append(file.split("/")[-1].split(".")[0])
    #     dfs.append(df)
    # files =glob.glob(project_path+"data/processed/judgellm_pairs_humman_eval_reasoning_aggregated/*.csv")
    # files.sort()
    # for file in files:
    #     df = pd.read_csv(file)
    #     df_metric_name.append(file.split("/")[-1].split(".")[0])
    #     dfs.append(df)
    # df_reference = pd.read_csv(project_path+"data/processed/humman_annotations_aggregated/ReasoningAnnotation_majority_ratings.csv")
    # reference_name = "human_annotations"
    # correlation_ponderation_general(dfs,df_metric_name,df_reference,reference_name)
    # correlation_poderation_specific_commentwise(dfs,df_metric_name,df_reference,reference_name)
    # Metric correlation paraphrasing
    dfs = []
    df_metric_name = []
    df_automatic_ranks = pd.read_csv(project_path+"results/metrics/paraphrase_automatic_metrics/final/ranks_automatic_metrics.csv")
    df_judge_llm_7 = pd.read_json(project_path+"data/processed/judgellm_paraphrasing_ranking/params7.json",lines=True)
    df_judge_llm_13 = pd.read_json(project_path+"data/processed/judgellm_paraphrasing_ranking/params13.json",lines=True)
    df_judge_llm_33 = pd.read_json(project_path+"data/processed/judgellm_paraphrasing_ranking/params33.json",lines=True)
    df_user1_ranking = pd.read_csv(project_path+"data/processed/human_eval_paraphrasing/paraphrasing_human_evaluation_user_1.csv")
    df_user1_ranking["ranking"] = df_user1_ranking["ranking"].apply(lambda x: eval(x))
    df_user2_ranking = pd.read_csv(project_path+"data/processed/human_eval_paraphrasing/paraphrasing_human_evaluation_user_2.csv")
    df_user2_ranking["ranking"] = df_user2_ranking["ranking"].apply(lambda x: eval(x))
    df_user3_ranking = pd.read_csv(project_path+"data/processed/human_eval_paraphrasing/paraphrasing_human_evaluation_user_3.csv")
    df_user3_ranking["ranking"] = df_user3_ranking["ranking"].apply(lambda x: eval(x))
    indexes = list(df_user1_ranking["idx"])
    df_automatic_ranks =df_automatic_ranks[df_automatic_ranks["idx"].isin(indexes)]
    df_judge_llm_7 =df_judge_llm_7[df_judge_llm_7["idx"].isin(indexes)]
    # df_judge_llm_7["ranking"] = df_judge_llm_7["ranking"].apply(lambda x: eval(x))
    df_judge_llm_13 =df_judge_llm_13[df_judge_llm_13["idx"].isin(indexes)]
    # df_judge_llm_13["ranking"] = df_judge_llm_13["ranking"].apply(lambda x: eval(x))
    df_judge_llm_33 =df_judge_llm_33[df_judge_llm_33["idx"].isin(indexes)]
    # df_judge_llm_33["ranking"] = df_judge_llm_33["ranking"].apply(lambda x: eval(x))
    df_automatic_ranks_list = []
    dfs_plot = []
    columns = ["toxic_ranks","bert_ranks","joint_ranks","bleu_ranks","joint_metrics_ranks"]
    for column in columns:
        df_partial = pd.DataFrame(columns=["idx","ranking","source"])
        df_partial["idx"] = df_automatic_ranks["idx"]
        df_partial["ranking"] = df_automatic_ranks[column]
        df_partial["ranking"] = df_partial["ranking"].apply(lambda x: eval(x))
        df_partial["source"] = df_automatic_ranks["source"]
        df_automatic_ranks_list.append(df_partial)
        if column != "joint_metrics_ranks":
            dfs_plot.append(df_partial)
    dfs.extend(df_automatic_ranks_list)
    df_metric_name.extend(columns)
    dfs.extend([df_judge_llm_7,df_judge_llm_13,df_judge_llm_33])
    df_metric_name.extend(["params7","params13","params33"])
    # Correlation with human evaluation
    metric_name_for_plot = ["Tox","BERT-F1","J","BLEU","J-LM 7B","J-LM 13B","J-LM 33B"]
    dfs_plot = dfs_plot+[df_judge_llm_7,df_judge_llm_13,df_judge_llm_33]
    obtain_correlation_matrix([df_user1_ranking,df_user2_ranking,df_user3_ranking]+dfs_plot,["Annot1","Annot2","Annot3"]+metric_name_for_plot)
    correlation_ponderation_general(dfs,df_metric_name,df_user1_ranking,"user1")
    correlation_poderation_specific_commentwise(dfs,df_metric_name,df_user1_ranking,"user1")
    correlation_ponderation_general(dfs,df_metric_name,df_user2_ranking,"user2")
    correlation_poderation_specific_commentwise(dfs,df_metric_name,df_user2_ranking,"user2")
    correlation_ponderation_general(dfs,df_metric_name,df_user3_ranking,"user3")
    correlation_poderation_specific_commentwise(dfs,df_metric_name,df_user3_ranking,"user3")

    # Correlation between users
    correlation_ponderation_general([df_user2_ranking,df_user3_ranking],["user2","user3"],df_user1_ranking,"user1")
    correlation_poderation_specific_commentwise([df_user2_ranking,df_user3_ranking],["user2","user3"],df_user1_ranking,"user1")
    correlation_ponderation_general([df_user1_ranking,df_user3_ranking],["user1","user3"],df_user2_ranking,"user2")
    correlation_poderation_specific_commentwise([df_user1_ranking,df_user3_ranking],["user1","user3"],df_user2_ranking,"user2")
    correlation_ponderation_general([df_user1_ranking,df_user2_ranking],["user1","user2"],df_user3_ranking,"user3")
    correlation_poderation_specific_commentwise([df_user1_ranking,df_user2_ranking],["user1","user2"],df_user3_ranking,"user3")

