import pandas as pd
import os
import sys
project_path=os.path.abspath(__file__).split('src')[0]
sys.path.append(project_path)
from src.evaluation.metrics import ReferenceExperimentMetric,ReferenceFreeExperimentMetric
import glob
from tqdm import tqdm

def preprocess_data(df:pd.DataFrame,folder_name:str):
    """
    Preprocess the data to be evaluated
    """
    df = df.dropna(subset=["paraphrase"])
    # df["result"] = df["result"].apply(lambda x: "" if pd.isna(x) else x)
    df = df.dropna(subset=["result"])
    df.reset_index(drop=True,inplace=True)
    num_rows_with_nan = df["result"].isna().sum()
    print(f"Number of rows with NaN values: {num_rows_with_nan}")
    #TODO: Add more folder name when finished the experiments
    if folder_name == "detoxllm" or folder_name == "llama3_1_8B" or folder_name == "marco-o1" or folder_name == "openO1" or folder_name == "qwen2_5_7B" or folder_name == "qwq_preview":
        for index,row in df.iterrows():
            result = row["result"]
            if isinstance(result, str) and result.startswith("{") and result.endswith("}"):
                result = eval(result)
                df.at[index,"result"] = result["paraphrase"]
            else:
                df.at[index,"result"] = result
        # df["result"] = df["result"].apply(eval)
        # df["result"] = df["result"].apply(lambda x: x["paraphrase"])

    return df
    
def evaluate_main_metrics(folder:str,reference_based_metric:ReferenceExperimentMetric,reference_free_metric:ReferenceFreeExperimentMetric,output_folder:str,batch_size:int=32):
    """
    Evaluate the main metrics of the paraphrase generation
    """
    files = glob.glob(folder+"/*.csv")
    # files = [files[6]]
    for file in files:
        df = pd.read_csv(file)
        folder_name = folder.split("/")[-1]
        model_name = file.split("/")[-1].rsplit(".",1)[0]
        df = preprocess_data(df,folder_name)
        # df = df[:32]
        df_reference_free = pd.DataFrame(columns=["idx","source","bert_scores","bleu_scores","content_similarities","fluency_scores","style_transfer_scores"])
        df_reference = pd.DataFrame(columns=["idx","source","toxic_scores","bert_scores","rouge_scores","bleu_scores"])
        for index in tqdm(range(0,len(df),batch_size),desc=f"Evaluating {model_name}"):
            end = min(index + batch_size, len(df))
    
            # Slice the DataFrame safely
            sentences = df["sentence"].iloc[index:end].tolist()
            paraphrases = df["paraphrase"].iloc[index:end].tolist()
            model_paraphrase = df["result"].iloc[index:end].tolist()
            sources = df["source"].iloc[index:end].tolist()

            # Metrics are in this order: bert_scores, bleu_scores, content_similarities, fluency_scores, style_transfer_scores
            results_reference_free = reference_free_metric.evaluate_batch(original_list=sentences, paraphrased_list=model_paraphrase)
            for i, (result, source) in enumerate(zip(results_reference_free, sources)):
                df_reference_free.loc[len(df_reference_free)] = [index + i, source] + result
            
            # Metrics are in this order: toxic_scores, bert_scores, rouge_scores, bleu_scores
            results_reference = reference_based_metric.evaluate_batch(reference_list=paraphrases, candidate_list=model_paraphrase)
            for i, (result, source) in enumerate(zip(results_reference, sources)):
                df_reference.loc[len(df_reference)] = [index + i, source] + result
        
        output_folder_name = output_folder+folder_name+f"/{model_name}/"
        if not os.path.exists(output_folder_name):
            os.makedirs(output_folder_name)
        df_reference_free.to_csv(output_folder_name+"reference_free_metrics.csv",index=False)
        df_reference.to_csv(output_folder_name+"reference_metrics.csv",index=False)
        sources = df["source"].unique().tolist()
        for source in sources:
            df_reference_free_source = df_reference_free[df_reference_free["source"]==source]
            df_reference_source = df_reference[df_reference["source"]==source]
            reference_free_json_results = {
                "bert_scores":df_reference_free_source["bert_scores"].mean(),
                "bleu_scores":df_reference_free_source["bleu_scores"].mean(),
                "content_similarities":df_reference_free_source["content_similarities"].mean(),
                "fluency_scores":df_reference_free_source["fluency_scores"].mean(),
                "style_transfer_scores":df_reference_free_source["style_transfer_scores"].mean(),
                "joint_score":reference_free_metric.obtain_joint_score(content_similarity=df_reference_free_source["content_similarities"].tolist(),fluency=df_reference_free_source["fluency_scores"].tolist(),style_transfer=df_reference_free_source["style_transfer_scores"].tolist())
            }
            reference_json_results = {
                "toxic_scores":df_reference_source["toxic_scores"].mean(),
                "bert_scores":df_reference_source["bert_scores"].mean(),
                "rouge_scores":df_reference_source["rouge_scores"].mean(),
                "bleu_scores":df_reference_source["bleu_scores"].mean()
            }
            with open(output_folder_name+f"{source}_reference_free_metrics.json","w") as f:
                f.write(str(reference_free_json_results))
            with open(output_folder_name+f"{source}_reference_metrics.json","w") as f:
                f.write(str(reference_json_results))
        
                
            

    return reference_based_metric,reference_free_metric

import argparse

if __name__ == "__main__":
    args = argparse.ArgumentParser() 
    args.add_argument("--folder",type=str,help="Folder we want to extract the paraphrase from",default="llama3_1_8B")#llama3_1_8B,openO1,marco-o1,qwen2_5_7B
    folder = args.parse_args().folder
    # folders = glob.glob(project_path+"data/processed/final_paraphrases/*")
    reference_based_metric = ReferenceExperimentMetric()
    reference_free_metric = ReferenceFreeExperimentMetric()
    if "basellm" in folder:
        output_folder = project_path+"results/metrics/paraphrase_automatic_metrics/basellm/"
    else:
        output_folder = project_path+"results/metrics/paraphrase_automatic_metrics/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # folders.pop(0)
    folder = project_path+"data/processed/final_paraphrases/"+folder
    print(f"Evaluating folder: {folder}")
    evaluate_main_metrics(folder=folder,reference_based_metric=reference_based_metric,reference_free_metric=reference_free_metric,output_folder=output_folder)
