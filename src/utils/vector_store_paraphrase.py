import pandas as pd
import lancedb
import pyarrow as pa
import os
project_path=os.path.abspath(__file__).split('src')[0]
import sys
sys.path.append(project_path)
from sentence_transformers import SentenceTransformer
from typing import List
from src.evaluation.text_similarity import levenshtein_distance
from src.evaluation.toxicity import ToxicMetric
import torch

class Vector_store_Paraphrase:
    def __init__(self, db_path:str,collection:str="paraphrasing", model_name:str="all-mpnet-base-v2"):
        if not os.path.exists(db_path):
            os.makedirs(db_path)
        self.db_path = db_path
        self.model_name = model_name
        self.client = lancedb.connect(db_path)
        tables = self.client.table_names()
        self.collection_name = collection
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = SentenceTransformer(model_name,device=device)
        if self.collection_name not in tables:
            schema = pa.schema([
                pa.field("id", pa.int64()),
                pa.field("sentence", pa.string()),
                pa.field("reasoning", pa.string()),
                pa.field("paraphrase", pa.string()),
                pa.field("shap_values", pa.list_(pa.string())),
                pa.field("label", pa.string()),
                pa.field("source", pa.string()),
                pa.field("vector", pa.list_(pa.float32(),self.embedder.get_sentence_embedding_dimension()))
            ])
            self.collection = self.client.create_table(self.collection_name, schema=schema)
        else:
            self.collection = self.client.open_table(self.collection_name)

    def get_collection_count(self):
        #TODO: get collection count
        count=self.collection.count_rows()
        return count
    
    def get_all_records(self):
        #TODO: get all records
        df = self.collection.to_pandas()
        return df
    
    def add_record(self, sentence:str,reasoning:List[str],paraphrase:List[str],shap_values:List[str],label:int,id:int,source:str):
        sentence_embedding = self.embedder.encode(sentence)
        label = "Toxic" if label == 1 else "Non-Toxic"
        data=[
            {"vector":sentence_embedding.tolist(),"sentence":sentence,"reasoning":reasoning,"paraphrase":paraphrase,"shap_values":shap_values,"label":label,"id":id,"source":source}
        ]
        self.collection.add(data)

    def check_records(self, df:pd.DataFrame):
        #TODO: check if all records are inserted
        length = len(df)
        if length == self.get_collection_count():
            return True
        else:
            records = self.get_all_records()
            for i in range(self.get_collection_count()):
                if records[i]["sentence"] :
                    return False
                
    def query(self, sentence:str,label:int,n_results:int=5):
        label = "Toxic" if label == 1.0 else "Non-Toxic"
        
        where_statement = f"label='{label}'"
        res=self.collection.search(query=self.embedder.encode(sentence),vector_column_name="vector")\
            .metric("cosine")\
            .limit(n_results+3)\
            .where(where_statement, prefilter=True)\
            .to_list()
        final_res = []
        for i in range(len(res)):
            if res[i]["sentence"] != sentence:
                final_res.append({
                    "sentence":res[i]["sentence"],
                    "reasoning":res[i]["reasoning"],
                    "paraphrase":res[i]["paraphrase"],
                    "shap_values":res[i]["shap_values"],
                    "label":res[i]["label"],
                    "source":res[i]["source"]
                })
        return final_res[:n_results]
    
    def find_best_comment(self,comments, toxicities, distances, weight_t=1, weight_l=1):
        scores = [
            weight_t * toxicities[i] + weight_l * distances[i]
            for i in range(len(comments))
        ]
        min_index = scores.index(min(scores))
        return min_index

def get_best_paraphrase(sentence:str,paraphrases:List[str],toxic_metric:ToxicMetric,weight_t=1,weight_l=1):
    toxicities = calculate_toxicity(paraphrases,toxic_metric)
    distances = levenshtein_distance(sentence,paraphrases)
    best_index = vector_store.find_best_comment(paraphrases,toxicities,distances,weight_t,weight_l)
    return paraphrases[best_index],best_index

def calculate_toxicity(texts:List[str],metric:ToxicMetric):
    return metric.pred(texts)
 
if __name__=="__main__":
    project_path=os.path.abspath(__file__).split('src')[0]
    db_path = project_path+"data/vector_store/"
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    model_name = "all-mpnet-base-v2"
    toxic_metric = ToxicMetric()
    vector_store = Vector_store_Paraphrase(db_path=db_path,model_name=model_name)
    counts = vector_store.get_collection_count()
    print(f"No of records in the collection: {counts}")
    # all_values=vector_store.get_all_records()
    df = pd.read_csv(project_path+"data/processed/final_few_shot_reasoning/few_shot_reasoning.csv")
    df["shap_values"] = df["shap_values"].apply(eval) 
    df_groups = df.groupby("sentence")
    if counts != len(df_groups):
        for idx,(name,groups) in enumerate(df_groups):
            
            # groups["reasoning_sampling"] = groups["reasoning_sampling"].apply(eval)
            # groups["reasoning_greedy"] = groups["reasoning_greedy"].apply(lambda x: [x])
            # groups["shap_values"] = groups["shap_values"].apply(eval)
            # groups["reasoning"] =groups["reasoning_greedy"]+groups["reasoning_sampling"]
            # groups["paraphrase"] = groups["paraphrase"].apply(lambda x: x if pd.notna(x) else "")
            if len(groups) > 1:
                
                result = groups.iloc[:1]
                result.reset_index(inplace=True,drop=True)
                paraphrases = groups["paraphrase"].tolist()
                sentece = result["sentence"].iloc[0]
                reasonings = groups["reasoning"].tolist()
                best_paraphrase,best_index = get_best_paraphrase(sentece,paraphrases,toxic_metric)
                result["paraphrase"] = best_paraphrase
                result["reasoning"] = reasonings[best_index]
                print(idx)
            else:
                pass
                # groups["reasoning"] = groups["reasoning"].apply(lambda x: [x])
                # groups["paraphrase"] = groups["paraphrase"].apply(lambda x: [x])
                result = groups
                result.reset_index(inplace=True,drop=True)
            pass
            if result.iloc[0]["label"] == 0:
                vector_store.add_record(sentence=result.iloc[0]["sentence"],
                                        reasoning=result.iloc[0]["reasoning"],
                                        source=result.iloc[0]["source"],
                                        paraphrase="",
                                        shap_values=result.iloc[0]["shap_values"],
                                        label=result.iloc[0]["label"],
                                        id=idx)
            else:
                if not pd.isna(result.iloc[0]["paraphrase"]):
                    vector_store.add_record(sentence=result.iloc[0]["sentence"],
                                            reasoning=result.iloc[0]["reasoning"],
                                            source=result.iloc[0]["source"],
                                            paraphrase= result.iloc[0]["paraphrase"],
                                            shap_values= result.iloc[0]["shap_values"],
                                            label= result.iloc[0]["label"],
                                            id=idx)
    result=vector_store.query(sentence="mentally scarred for life by those cockroach swarmed shit holes .",
                              label=1,source="paradetox",n_results=3)
    print(result)
        