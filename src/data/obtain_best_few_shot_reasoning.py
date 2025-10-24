import pandas as pd
import os
import glob
import argparse
project_path=os.path.abspath(__file__).split('src')[0]
import sys
sys.path.append(project_path)
from src.data.aggregate_judge_llm_predictions import find_reversed_pairs,obtain_best_models

def aggregate_judge_predictions(files,output_path):
    """
    Aggregate the predictions from the judge model for the given pairs of directories.

    Parameters:
        data_dir (str): The path to the directory containing subfolders.
        files (list): List of files to aggregate.

    Returns:
        pd.DataFrame: A DataFrame containing the aggregated predictions.
    """
    base_name = "params33"
    df_base = pd.DataFrame(columns=["idx","modelA","modelB","result"])
    for file in files:
        
        base_df = pd.read_json(file, lines=True)
        for index, row in base_df.iterrows():
            result_base = row["pred_text"].split(" ")
            result= []
            for i in range(len(result_base)):
                result.append(float(result_base[i]))
            if result[0] > result[1]:
                result = 1
            elif result[0] < result[1]:
                result = 2
            elif result[0] == result[1]:
                result = 0
            df_base.loc[len(df_base)] = [row["question_id"],row["answer1_model_id"],row["answer2_model_id"],result]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    best_models = obtain_best_models(df_base)
    print(f"Best models for {base_name}: {best_models}")
    open(output_path+base_name+'_best_models.txt', 'w').write(str(best_models))
    df_base.to_csv(output_path+base_name+'_aggregated.csv',index=False)
    return df_base
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_prediction_path", default=project_path + 'data/processed/judgellm_few_shots_pair_reasoning/')
    parser.add_argument("--output_path", default=project_path + 'data/processed/judgellm_few_shots_pair_reasoning_aggregated/')
    args = parser.parse_args()
    data_dir = args.judge_prediction_path
    output_path = args.output_path
    files = glob.glob(data_dir+"params33"+"/*.json")
    df_marco_o1=pd.read_csv(project_path+'data/processed/few_shot_reasoning/few_shot_reasoning_marco-o1.csv')
    df_openO1=pd.read_csv(project_path+'data/processed/few_shot_reasoning/few_shot_reasoning_openO1.csv')
    df_qwq=pd.read_csv(project_path+'data/processed/few_shot_reasoning/few_shot_reasoning_qwq_preview.csv')
    df_few_shot_reasoning = pd.DataFrame(columns=df_marco_o1.columns)
    for column in df_few_shot_reasoning.columns:
        if column!="reasoning":
            df_few_shot_reasoning[column]=df_marco_o1[column]

    pair_tournament_result_df=aggregate_judge_predictions(files,output_path)
    pair_tournament_result_df = pd.read_csv(output_path+'params33_aggregated.csv')
    groups=pair_tournament_result_df.groupby(["idx"])
    #TODO: mirar el orden de los groups
    for name, group in groups:
        best_models=obtain_best_models(group)
        best_model_name = max(best_models, key=best_models.get)

        idx = group["idx"].iloc[0]
        if best_model_name == "marco-o1":
            df_few_shot_reasoning.loc[idx,"reasoning"] = df_marco_o1.loc[idx]["reasoning"]
        elif best_model_name == "openO1":
            df_few_shot_reasoning.loc[idx,"reasoning"] = df_openO1.loc[idx]["reasoning"]
        elif best_model_name == "qwq_preview":
            df_few_shot_reasoning.loc[idx,"reasoning"] = df_qwq.loc[idx]["reasoning"]
        if pd.isna(df_few_shot_reasoning.loc[idx,"reasoning"]) :
            pass
    
    reasoning_path = project_path + 'data/processed/final_few_shot_reasoning/few_shot_reasoning.csv'
    if not os.path.exists(project_path + 'data/processed/final_few_shot_reasoning/'):
        os.makedirs(project_path + 'data/processed/final_few_shot_reasoning/')
    df_few_shot_reasoning.to_csv(reasoning_path,index=False)
    







    

    