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

def extract_unique_users_dataframes(df,user_dict_names):
    
    # df["reasoning_response_rating.responses"] = df["reasoning_response_rating.responses"].apply(eval)
    
    dfs = []
    users = list(user_dict_names.keys())
    for user in users:
        df_ = pd.DataFrame(columns=["idx","modelA","modelB","result"])
        dfs.append(df_)
    for index, row in df.iterrows():
        user_list=row["reasoning_response_rating.responses.users"]
        values_list= row["reasoning_response_rating.responses"]
        modelA = row["metadata.ModelA_name"][0]
        modelB = row["metadata.ModelB_name"][0]
        idx = row["metadata.idx"]
        for user,value in zip(user_list,values_list):
            for user_in_list in users:
                if user_in_list==user:
                    dfs[users.index(user_in_list)].loc[len(dfs[users.index(user_in_list)])] = [idx,modelA,modelB,value]
    return dfs,users
                

                
def get_unique_annotators(df):
    # Get the number of unique users and their corresponding annotations. Only keep users with more than 200 annotations
    user_dict_names = {}
    for index, row in df.iterrows():
        user_list=row["reasoning_response_rating.responses.users"]
        for user in user_list:
            if user not in user_dict_names:
                user_dict_names[user]=1
            else:
                user_dict_names[user]+=1
    new_dict = user_dict_names.copy()
    for user in user_dict_names.keys():
        if new_dict[user]<200:
            new_dict.pop(user)
    return new_dict

def obtain_average_ratings(ratings:np.ndarray):
    final_ratings = []
    for rating in ratings:
        counter=Counter(rating)

        most_common = counter.most_common(1)
        if most_common[0][1] ==1:
            final_ratings.append(0)
            pass
        else:
            final_ratings.append(most_common[0][0])
    return final_ratings

def flatten_tuple(t):
    """
    Flattens a tuple by expanding any list elements inside it.
    """
    result = []
    for elem in t:
        if isinstance(elem, list):
            result.extend(elem)  # Expand the list into the result
        else:
            result.append(elem)  # Keep other elements as they are
    return tuple(result)
if __name__ == "__main__":
    annotations_path = project_path+"data/interim/humman_annotations/ReasoningAnnotation.json"
    if os.path.exists(annotations_path):
        df = pd.read_json(annotations_path)
    else:
        ds = load_dataset("TheMrguiller/ReasoningAnnotation")
        df=ds["train"].to_pandas()
        if not os.path.exists(project_path+"data/interim/humman_annotations/"):
            os.makedirs(project_path+"data/interim/humman_annotations/")
        df.to_json(annotations_path)
        # .to_csv(project_path+"data/interim/humman_annotations/ReasoningAnnotation.csv",index=False)
    
    # df["metadata.ModelA_name"] = df["metadata.ModelA_name"].apply(eval)
    # df["metadata.ModelB_name"] = df["metadata.ModelB_name"].apply(eval)
    # df["reasoning_response_rating.responses.users"] = df["reasoning_response_rating.responses.users"].apply(eval)
    user_dict_names = get_unique_annotators(df)
    # 
    #TODO: Per user store their unique annotations
    dfs,users = extract_unique_users_dataframes(df,user_dict_names)
    annotator_1 = dfs[0]["result"].tolist()
    annotator_2 = dfs[1]["result"].tolist()
    annotator_3 = dfs[2]["result"].tolist()
    annotations = list(zip(annotator_1,annotator_2,annotator_3))
    annotations = np.array(annotations)
    fleish_kappa = irr.fleiss_kappa(irr.aggregate_raters(annotations)[0], method='fleiss')
    print(f"Fleiss Kappa base annotations style: {fleish_kappa}")
    # annotator_1 = [[1,2] if x==0 else x for x in annotator_1]
    # annotator_2 = [[1,2] if x==0 else x for x in annotator_2]
    # annotator_3 = [[1,2] if x==0 else x for x in annotator_3]
    # annotations = list(zip(annotator_1,annotator_2,annotator_3))
    # annotations = [flatten_tuple(x) for x in annotations]
    # # annotations = np.array(annotations)
    # fleish_kappa = irr.fleiss_kappa(irr.aggregate_raters(annotations)[0], method='fleiss')
    # print(f"Fleiss Kappa modified annotation style: {fleish_kappa}")
    for user,df_user in zip(users,dfs):
        df_user.to_csv(project_path+"data/interim/humman_annotations/ReasoningAnnotation_"+user+".csv",index=False)
    #TODO: Per user store their best model results
    for user,df_user in zip(users,dfs):
        best_models = obtain_best_models(df_user)
        open(project_path+'data/interim/humman_annotations/'+user+'_best_models.txt', 'w').write(str(best_models))
    #TODO: Obtain majority rating of comments idx,modelA,modelB,result
    result_list = []
    for user,df_user in zip(users,dfs):
        result_list.append(df_user["result"].tolist())
    result_list = np.array(result_list)
    result_list = result_list.T
    majority_ratings = obtain_average_ratings(result_list)
    majority_ratings_df = pd.DataFrame(columns=["idx","modelA","modelB","result"])
    for idx,row in df.iterrows():
        majority_ratings_df.loc[len(majority_ratings_df)] = [row["metadata.idx"],row["metadata.ModelA_name"][0],row["metadata.ModelB_name"][0],majority_ratings[idx]]
    if not os.path.exists(project_path+"data/processed/humman_annotations_aggregated/"):
        os.makedirs(project_path+"data/processed/humman_annotations_aggregated/")
    majority_ratings_df.to_csv(project_path+"data/processed/humman_annotations_aggregated/ReasoningAnnotation_majority_ratings.csv",index=False)
    #TODO: Obtain best model results for the average
    best_models = obtain_best_models(majority_ratings_df)
    open(project_path+'data/processed/humman_annotations_aggregated/majority_ratings_best_models.txt', 'w').write(str(best_models))
    