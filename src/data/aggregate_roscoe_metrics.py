import pandas as pd
import os
import sys
project_path=os.path.abspath(__file__).split('src')[0]
sys.path.append(project_path)
import glob
import argparse
from src.data.aggregate_judge_llm_predictions import obtain_best_models

def obtain_best_metric(valueA,valueB):
    if valueA > valueB:
        return 1
    elif valueA < valueB:
        return 2
    elif valueA == valueB:
        return 0
    
def obtain_roscoe_metrics(df):
    df_temp = pd.DataFrame(columns=["idx","ROSCOE_SA","ROSCOE_SS","ROSCOE_LC","discourse_representation","coherence"])
    for index, row in df.iterrows():
        df_temp.loc[len(df_temp)] = [row["ID"],(row["faithfulness"]+row["informativeness_step"])/2,row["informativeness_chain"],row["grammar_step"],row["discourse_representation"],row["coherence_step_vs_step"]]
    return df_temp

def aggregate_values(data_dir):
    files = glob.glob(data_dir+"*.tsv")
    df = pd.DataFrame(columns=["idx","modelA","modelB","ROSCOE-SA","ROSCOE-SS","ROSCOE-LC","DR","Coh"])
    df_models_list= []
    df_model_names = []
    for file in files:
        filename = file.split("/")[-1].split(".")[0].split("scores_reasoning_")[1]
        df_temp = pd.read_csv(file, delim_whitespace=True)
        df_temp = obtain_roscoe_metrics(df_temp)
        df_models_list.append(df_temp)
        df_model_names.append(filename)
    for i in range(len(df_models_list)):
        for j in range(i+1,len(df_models_list)):
            for index, row in df_models_list[i].iterrows():

                roscoe_sa = obtain_best_metric(row["ROSCOE_SA"],df_models_list[j].loc[index]["ROSCOE_SA"])
                roscoe_ss = obtain_best_metric(row["ROSCOE_SS"],df_models_list[j].loc[index]["ROSCOE_SS"])
                roscoe_lc = obtain_best_metric(row["ROSCOE_LC"],df_models_list[j].loc[index]["ROSCOE_LC"])
                discourse_representation = obtain_best_metric(row["discourse_representation"],df_models_list[j].loc[index]["discourse_representation"])
                coherence = obtain_best_metric(row["coherence"],df_models_list[j].loc[index]["coherence"])
                df.loc[len(df)] = [row["idx"],df_model_names[i],df_model_names[j],roscoe_sa,roscoe_ss,roscoe_lc,discourse_representation,coherence]
    
    for metric in df.columns[3:]:

        df_temp = df[["idx","modelA","modelB",metric]]
        df_temp = df_temp.rename(columns={metric:"result"})
        if not os.path.exists(project_path+"data/processed/roscoe_aggregated/"):
            os.makedirs(project_path+"data/processed/roscoe_aggregated/")
        best_models = obtain_best_models(df_temp)
        print(f"Best models for {metric}: {best_models}")
        open(project_path+'/data/processed/roscoe_aggregated/'+metric+'_best_models.txt', 'w').write(str(best_models))
        df_temp.to_csv(project_path+"data/processed/roscoe_aggregated/"+metric+".csv",index=False)
            
                
                


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Aggregate ROSCOE metrics')
    parser.add_argument('--data_dir', type=str, help='Path to the directory containing the ROSCOE metrics',default=project_path+"src/evaluation/parlai-app/projects/roscoe/roscoe_data/human_eval_output/roscoe-512-roberta-base/")
    args = parser.parse_args()
    data_dir=args.data_dir
    aggregate_values(data_dir)
