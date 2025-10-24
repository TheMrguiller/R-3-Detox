import pandas as pd
import os
project_path=os.path.abspath(__file__).split('src')[0]


if __name__=="__main__":
    df_appdia = pd.read_csv(project_path+"data/external/CAPP_article/APPDIA_Generated-Paraphrases.csv")
    df_appdia["source"] = "APPDIA"
    df_paradetox = pd.read_csv(project_path+"data/external/CAPP_article/ParaDetox_Generated-Paraphrases.csv")
    df_paradetox["source"] = "paradetox"
    df = pd.read_csv(project_path+"data/processed/final_few_shot_reasoning/few_shot_reasoning.csv")
    df_capp = pd.concat([df_paradetox,df_appdia])
    df_capp.reset_index(drop=True, inplace=True)

    for index,row in df_capp.iterrows():
        utterance = row["Utterance"]
        paraphrase = df[df["sentence"]==utterance]["paraphrase"].values
        if len(paraphrase) == 0:
            continue
        df_capp.at[index,"Gold-Standard"] = paraphrase[0]
    del df_appdia,df_paradetox,df
    columns = df_capp.columns.to_list()
    columns.remove("Gold-Standard")
    columns.remove("source")
    columns.remove("Utterance")
    for model_result in columns:
        df = pd.DataFrame(columns=["source","sentence","paraphrase","result"])
        df["source"] = df_capp["source"]
        df["sentence"] = df_capp["Utterance"]
        df["paraphrase"] = df_capp["Gold-Standard"]
        df["result"] = df_capp[model_result]
        if not os.path.exists(project_path+"data/processed/final_paraphrases/capp/"):
            os.makedirs(project_path+"data/processed/final_paraphrases/capp/")
        df.to_csv(project_path+"data/processed/final_paraphrases/capp/"+model_result+".csv",index=False)
    
        
