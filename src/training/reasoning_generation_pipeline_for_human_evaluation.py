import sys
import os
proyect_path = os.path.abspath(__file__).split('src')[0]
sys.path.append(proyect_path)
import pandas as pd
import argparse
from src.utils.llms.llm_huggingface_proxy import LLMHuggingFace
from src.utils.llms.llm_openai_proxy import LLMOpenAIProxy
import yaml
import random
random.seed(42)
import ast
from itertools import chain

def get_config_path(config_name:str):
    return proyect_path + f'src/utils/llms/configs/{config_name}.yaml'

def generate_user_prompt(template:str,original_message:str,paraphrase:str,toxic_words:str,label:str):
    return template.format(sentence=original_message,paraphrase=paraphrase,toxic_words=toxic_words,label=label)

def generate_messages(toxic_template:str,non_toxic_template:str,original_messages:list,paraphrases:list,toxic_words:list,labels:list):
    messages=[]
    for original_message, paraphrase, toxic_words, label in zip(original_messages, paraphrases, toxic_words, labels):
        if label == 1:
            label = "Toxic"
            user_prompt = generate_user_prompt(toxic_template,original_message,paraphrase,toxic_words,label)
            messages.append(user_prompt)
        else:
            label = "Non-toxic"
            user_prompt = generate_user_prompt(non_toxic_template,original_message,paraphrase,toxic_words,label)
            messages.append(user_prompt)
    return messages
    
def extract_huggingface_reasoning(config_path:str,df:pd.DataFrame,toxic_prompt_template:str,non_toxic_prompt_template:str,batch_size:int=8):
    llm = LLMHuggingFace(config_path)
    df["reasoning"] = ""
    for idx in range(0, len(df), batch_size):
        batch = df[idx:idx+batch_size]
        original_messages = batch['sentence'].tolist()
        paraphrases = batch['paraphrase'].tolist()
        labels = batch['label'].tolist()
        toxic_words = batch['shap_values'].tolist()
        user_messages=generate_messages(toxic_prompt_template,non_toxic_prompt_template,original_messages,paraphrases,toxic_words,labels)
        responses = llm.query(user_messages=user_messages)
        responses = list(chain.from_iterable(responses))
        for index, response in enumerate(responses):
            # print(f"Index: {idx+index}")
            # print(f"Sentence: {original_messages[index]}")
            # print(f"Paraphrase: {paraphrases[index]}")
            # print(f"Label: {labels[index]}")
            # print(f"Reasoning: {response}")
            if "marco-o1" in config_path or "openO1" in config_path:
                response = llm.postprocess_response(response)["reasoning"]
            df.loc[idx+index, "reasoning"] = response
    return df

def extract_open_ai_reasoning(config_path:str,df:pd.DataFrame,toxic_prompt_template:str,non_toxic_prompt_template:str,batch_size:int=8):
    llm = LLMOpenAIProxy(config_path)
    df["reasoning"] = ""
    for idx in range(0, len(df), batch_size):
        batch = df[idx:idx+batch_size]
        original_messages = batch['sentence'].tolist()
        paraphrases = batch['paraphrase'].tolist()
        labels = batch['label'].tolist()
        toxic_words = batch['shap_values'].tolist()
        user_messages=generate_messages(toxic_prompt_template,non_toxic_prompt_template,original_messages,paraphrases,toxic_words,labels)
        responses = llm.query(user_messages=user_messages)
        for index, response in enumerate(responses):
            df.loc[idx+index, "reasoning"] = response
    return df

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--config_name",default="openai_o1",choices=["marco-o1","openO1","qwq_preview","skywork-o1","openai_o1"],help="Name of the c to use")
    parser.add_argument("--proxy_type",default="openai",choices=["vllm","huggingface","openai"],help="Type of proxy to use")
    config_name= parser.parse_args().config_name
    proxy_type= parser.parse_args().proxy_type
    df = pd.read_csv(proyect_path + 'data/processed/shap_values_aggregated/processed_shap_values.csv')
    # df=df.explode("paraphrase")
    df_non_toxic = df[df["label"] == 0].sample(n=5,replace=False, random_state=42)
    df_appdia = df[df["source"] == "APPDIA"].sample(n=5,replace=False, random_state=42)
    df_paradetox = df[df["source"] == "paradetox"].sample(n=5,replace=False, random_state=42)
    df_parallel = df[df["source"] == "parallel_detoxification"].sample(n=5,replace=False, random_state=42)
    df = pd.concat([df_non_toxic,df_appdia,df_paradetox,df_parallel])
    df.reset_index(drop=True, inplace=True)
    df["paraphrase"] = df["paraphrase"].apply(lambda x: random.choice(ast.literal_eval(x)) if pd.notna(x) else "")
    df = df.sample(frac=1,replace=False, random_state=42)
    df.reset_index(drop=True, inplace=True)
    with open(proyect_path + "src/utils/llms/prompts/generate_reasoning_prompt.yaml", "r") as file:
        prompt_config = yaml.safe_load(file)
    toxic_prompt_template = prompt_config["toxic_template"]
    non_toxic_prompt_template = prompt_config["non_toxic_template"]
    if proxy_type == "huggingface":
        df=extract_huggingface_reasoning(get_config_path(config_name),df,toxic_prompt_template,non_toxic_prompt_template)
    elif proxy_type == "openai":
        df=extract_open_ai_reasoning(get_config_path(config_name),df,toxic_prompt_template,non_toxic_prompt_template)
    file.close()
    if not os.path.exists(proyect_path + 'data/interim/reasoning_human_eval/'):
        os.makedirs(proyect_path + 'data/interim/reasoning_human_eval/')
    df.to_csv(proyect_path + f'data/interim/reasoning_human_eval/reasoning_human_eval_{config_name}_.csv',index=False)
