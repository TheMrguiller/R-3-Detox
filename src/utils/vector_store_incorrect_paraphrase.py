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

class Vector_store_Evaluation:
    def __init__(self, db_path:str,collection:str="evaluation_metric", model_name:str="all-mpnet-base-v2"):
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
                pa.field("correct_paraphrase", pa.string()),
                pa.field("incorrect_paraphrase", pa.string()),
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
    
    def add_record(self, sentence:str,correct_paraphrase:str,incorrect_paraphrase:str,id:int):
        sentence_embedding = self.embedder.encode(sentence)
        
        data=[
            {"vector":sentence_embedding.tolist(),"sentence":sentence,"correct_paraphrase":correct_paraphrase,"incorrect_paraphrase":incorrect_paraphrase,"id":id}
        ]
        self.collection.add(data)

    def query(self, paraphrase:str,n_results:int=5):
        
        res=self.collection.search(query=self.embedder.encode(paraphrase),vector_column_name="vector")\
            .metric("cosine")\
            .limit(n_results+3)\
            .to_list()
        final_res = []
        for i in range(len(res)):
            if res[i]["correct_paraphrase"] != paraphrase:
                final_res.append({
                    "correct_paraphrase":res[i]["correct_paraphrase"],
                    "incorrect_paraphrase":res[i]["incorrect_paraphrase"],
                    "sentence":res[i]["sentence"],
                    "id":res[i]["id"]
                })
        return final_res[:n_results]
    

if __name__=="__main__":
    project_path=os.path.abspath(__file__).split('src')[0]
    db_path = project_path+"data/vector_store/"
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    model_name = "all-mpnet-base-v2"
    toxic_metric = ToxicMetric()
    vector_store = Vector_store_Evaluation(db_path=db_path,model_name=model_name)
    counts = vector_store.get_collection_count()
    print(f"No of records in the collection: {counts}")
    # all_values=vector_store.get_all_records()
    df = pd.read_csv(project_path+"data/processed/incorrect_paraphrases/incorrect_paraphrases.csv")
    
    df.reset_index(drop=True,inplace=True)
    df["incorrect_paraphrase"] = df["incorrect_paraphrase"].apply(lambda x: eval(x))
    if counts != len(df):
        for idx,row in df.iterrows():
            vector_store.add_record(sentence=row["sentence"],
                                    correct_paraphrase=row["paraphrase"],
                                    incorrect_paraphrase=row["incorrect_paraphrase"]["incorrect_paraphrase"],
                                    id=idx)
    
    result=vector_store.query(paraphrase="mentally scarred for life by those cockroach swarmed shit holes .",
                              n_results=3)
    print(result)
        