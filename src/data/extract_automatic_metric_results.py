import pandas as pd
import glob
import os
import sys
project_path=os.path.abspath(__file__).split('src')[0]
sys.path.append(project_path)
import json

def read_json(file:str):
    with open(file) as f:
        content = f.read()
        return  eval(content)
if __name__ == "__main__":
    path_of_metrics = project_path+"results/metrics/paraphrase_automatic_metrics/"
    output_folder = project_path+"results/reports/paraphrase_automatic_metrics/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    base_folders = glob.glob(path_of_metrics+"*")
    for base_folder in base_folders:
        folders_results = glob.glob(base_folder+"/*")
        experiment_name = base_folder.split("/")[-1]
        df_APPDIA = pd.DataFrame(columns=["source","Free_bert_scores","Free_bleu_scores","Free_content_similarities","Free_fluency_scores","Free_style_transfer_scores","Free_joint_score",
                                          "Ref_toxic_scores","Ref_bert_scores","Ref_rouge_scores","Ref_bleu_scores"])
        df_paradetox = pd.DataFrame(columns=["source","Free_bert_scores","Free_bleu_scores","Free_content_similarities","Free_fluency_scores","Free_style_transfer_scores","Free_joint_score",
                                            "Ref_toxic_scores","Ref_bert_scores","Ref_rouge_scores","Ref_bleu_scores"])
        df_parallel = pd.DataFrame(columns=["source","Free_bert_scores","Free_bleu_scores","Free_content_similarities","Free_fluency_scores","Free_style_transfer_scores","Free_joint_score",
                                             "Ref_toxic_scores","Ref_bert_scores","Ref_rouge_scores","Ref_bleu_scores"])
        # {'bert_scores': , 'bleu_scores': , 'content_similarities':, 'fluency_scores': , 'style_transfer_scores': , 'joint_score':}
        # {'toxic_scores':, 'bert_scores': , 'rouge_scores': , 'bleu_scores': }
        for folder_results in folders_results:
            files = glob.glob(folder_results+"/*.json")
            source = folder_results.split("/")[-1]
            if any("APPDIA" in file for file in files):
                appdia_free:dict = read_json(folder_results+"/APPDIA_reference_free_metrics.json")
                appdia_ref:dict = read_json(folder_results+"/APPDIA_reference_metrics.json")
                
                df_APPDIA.loc[len(df_APPDIA)] = [source]+[appdia_free[key] for key in appdia_free.keys()]+[appdia_ref[key] for key in appdia_ref.keys()]
            if any("paradetox" in file for file in files):
                paradetox_free = read_json(folder_results+"/paradetox_reference_free_metrics.json")
                paradetox_ref = read_json(folder_results+"/paradetox_reference_metrics.json")
                df_paradetox.loc[len(df_paradetox)] = [source]+[paradetox_free[key] for key in paradetox_free.keys()]+[paradetox_ref[key] for key in paradetox_ref.keys()]
            if any("parallel" in file for file in files):
                parallel_free = read_json(folder_results+"/parallel_detoxification_reference_free_metrics.json")
                parallel_ref = read_json(folder_results+"/parallel_detoxification_reference_metrics.json")
                df_parallel.loc[len(df_parallel)] = [source]+[parallel_free[key] for key in parallel_free.keys()]+[parallel_ref[key] for key in parallel_ref.keys()]
        if len(df_APPDIA)>0:
            df_APPDIA.to_csv(output_folder+f"{experiment_name}_APPDIA.csv",index=False)
        if len(df_paradetox)>0:
            df_paradetox.to_csv(output_folder+f"{experiment_name}_paradetox.csv",index=False)
        if len(df_parallel)>0:
            df_parallel.to_csv(output_folder+f"{experiment_name}_parallel.csv",index=False)

    files = glob.glob(output_folder+"*")
    file_APPDIA = [file for file in files if "APPDIA" in file]
    file_paradetox = [file for file in files if "paradetox" in file]
    file_parallel = [file for file in files if "parallel" in file]
    df_APPDIA = pd.concat([pd.read_csv(file) for file in file_APPDIA])
    df_paradetox = pd.concat([pd.read_csv(file) for file in file_paradetox])
    df_parallel = pd.concat([pd.read_csv(file) for file in file_parallel])
    output_folder = output_folder+"final/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    df_APPDIA.to_csv(output_folder+"APPDIA.csv",index=False)
    df_paradetox.to_csv(output_folder+"paradetox.csv",index=False)
    df_parallel.to_csv(output_folder+"parallel.csv",index=False)

    
            
            
            
            
            
            
