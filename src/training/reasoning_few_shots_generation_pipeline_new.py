import sys
import os
proyect_path = os.path.abspath(__file__).split('src')[0]
sys.path.append(proyect_path)
import pandas as pd
import argparse
from src.utils.llms.llm_huggingface_proxy import LLMHuggingFace
from src.utils.llms.llm_vllm_offline_proxy import LLMVLLMOfflineProxy
import yaml
import random
random.seed(42)
import ast
from tqdm import tqdm
import time

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

def set_llm_greedy_settings(llm:LLMHuggingFace):
    
    llm.config["temperature"] = 0.0
    llm.config["do_sample"] = False
    llm.config["num_return_sequences"] = 1
    return llm
def set_llm_sample_settings(llm:LLMHuggingFace):
    llm.config["temperature"] = 0.7
    llm.config["do_sample"] = True
    llm.config["top_p"] = 0.9
    llm.config["num_return_sequences"] = 4
    return llm

def extract_huggingface_reasoning(config_path:str,df:pd.DataFrame,toxic_prompt_template:str,non_toxic_prompt_template:str,batch_size:int=8):
    llm= LLMHuggingFace(config_path)
    df["reasoning_sampling"] = ""
    df["reasoning_greedy"] = ""
    for idx in tqdm(range(0, len(df), batch_size), total=len(df)//batch_size):
        batch = df[idx:idx+batch_size]
        original_messages = batch['sentence'].tolist()
        paraphrases = batch['paraphrase'].tolist()
        labels = batch['label'].tolist()
        toxic_words = batch['shap_values'].tolist()
        user_messages=generate_messages(toxic_prompt_template,non_toxic_prompt_template,original_messages,paraphrases,toxic_words,labels)
        responses_greedy = llm.query(user_messages=user_messages)
        llm= set_llm_sample_settings(llm)
        responses_sample = llm.query(user_messages=user_messages)
        llm= set_llm_greedy_settings(llm)
        # print(f"len responses_greedy: {len(responses_greedy)}")
        # print(f"len responses_sample: {len(responses_sample)}")
        # print(f"Response Greedy: {responses_greedy}")
        # print(f"Response Sample: {responses_sample}")
        # print(f"Config: {config_path}") 
        for index, (response_greedy, response_sample) in enumerate(zip(responses_greedy, responses_sample)):
            # print(f"Index: {idx+index}")
            # print(f"Sentence: {original_messages[index]}")
            # print(f"Paraphrase: {paraphrases[index]}")
            # print(f"Label: {labels[index]}")
            # print(f"Reasoning: {response}")ç
            # print(f"len response_greedy: {len(response_greedy)}")
            # print(f"len response_sample: {len(response_sample)}")
            responses = response_greedy + response_sample
            process_responses = []
            for response in responses:
                if "marco-o1" in config_path or "openO1" in config_path:
                    response = llm.postprocess_response(response)["reasoning"]
                    process_responses.append(response)
                else:
                    process_responses.append(response)
            # print(f"Lenght of process_responses: {len(process_responses)}")
            df.loc[idx+index, "reasoning_greedy"] = process_responses.pop(0)
            # print(f"Lenght of process_responses: {len(process_responses)}")
            df.at[idx+index, "reasoning_sampling"] = process_responses
    return df

def extract_vllm_reasoning(config_path:str,df:pd.DataFrame,toxic_prompt_template:str,non_toxic_prompt_template:str,batch_size:int=8,num_return_sequences:int=4):
    llm= LLMVLLMOfflineProxy(config_path)
    df["reasoning"] = ""

    for idx in tqdm(range(0, len(df), batch_size),desc="Generating reasoning"):
        batch = df[idx:idx+batch_size]
        original_messages = batch['sentence'].tolist()
        paraphrases = batch['paraphrase'].tolist()
        labels = batch['label'].tolist()
        toxic_words = batch['shap_values'].tolist()
        user_messages=generate_messages(toxic_prompt_template,non_toxic_prompt_template,original_messages,paraphrases,toxic_words,labels)
        
        responses_greedy = llm.query(user_messages=user_messages)
        
        # print(f"len responses_greedy: {len(responses_greedy)}")
        # print(f"len responses_sample: {len(responses_sample)}")
        # print(f"Response Greedy: {responses_greedy}")
        # print(f"Response Sample: {responses_sample}")
        # print(f"Config: {config_path}") 
        for index,response in enumerate(responses_greedy):
            if "marco-o1" in config_path or "openO1" in config_path:
                response = llm.postprocess_response(response)["reasoning"]
                df.loc[idx+index, "reasoning"] = response
            else:
                df.loc[idx+index, "reasoning"] = response
    return df

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--config_name",default="qwq_preview",choices=["marco-o1","openO1","qwq_preview"],help="Name of the c to use")
    parser.add_argument("--proxy_type",default="vllm",choices=["vllm","huggingface"],help="Type of proxy to use")
    config_name= parser.parse_args().config_name
    proxy_type= parser.parse_args().proxy_type
    df = pd.read_csv(proyect_path + 'data/processed/shap_values_aggregated/processed_shap_values_left_no_toxic.csv')
    # df=df.explode("paraphrase")
    df["paraphrase"] = df["paraphrase"].apply(lambda x: random.choice(ast.literal_eval(x)) if pd.notna(x) else "")
    df = df.sample(frac=1,replace=False, random_state=42)
    # df = df[:24]
    
    # df = df[df["source"]=="parallel_detoxification"]
    df.reset_index(drop=True, inplace=True)
    with open(proyect_path + "src/utils/llms/prompts/generate_reasoning_prompt.yaml", "r") as file:
        prompt_config = yaml.safe_load(file)
    toxic_prompt_template = prompt_config["toxic_template"]
    non_toxic_prompt_template = prompt_config["non_toxic_template"]
    file.close()
    # start = time.time()
    if proxy_type == "huggingface":
        df=extract_huggingface_reasoning(get_config_path(config_name),df,toxic_prompt_template,non_toxic_prompt_template,batch_size=2)
    if proxy_type == "vllm":
        df=extract_vllm_reasoning(get_config_path(config_name),df,toxic_prompt_template,non_toxic_prompt_template,batch_size=256)
    if not os.path.exists(proyect_path + 'data/interim/few_shot_reasoning/'):
        os.makedirs(proyect_path + 'data/interim/few_shot_reasoning/')
    df.to_csv(proyect_path + f'data/interim/few_shot_reasoning/few_shot_reasoning_{config_name}_left_no_toxic.csv',index=False)
