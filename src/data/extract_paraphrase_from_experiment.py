from typing import List
import pandas as pd
import sys
import os
project_path = os.path.abspath(__file__).split('src')[0]
sys.path.append(project_path)
import glob
import re
from markdown import markdown
from bs4 import BeautifulSoup
import argparse
from src.utils.chains.structured_output_enforcer import StructuredOutputEnforcer
from tqdm import tqdm
from transformers import AutoTokenizer
import yaml

def strip_markdown(md_text):
    html = markdown(md_text)  # Convert Markdown to HTML
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()

def preprocess_based_on_model_tags(text:str):

    if "<Output>" in text and "</Output>" in text:
        text = text.split("<Output>")[1]
        text = text.split("</Output>")[0]
    return text

# def eliminate_double_quotes_inside_value(text: str) -> str:
#     # Define the regex pattern
#     regex = r'\{\s*"reasoning":\s*"(.*?)",\s*"paraphrase":\s*"(.*?)"\s*\}'
    
#     # Function to replace the match with modified values
#     def replace_quotes(match):
#         # Extract values for reasoning and paraphrase
#         reasoning = match.group(1).replace('"', "'")  # Replace double quotes with single quotes
#         paraphrase = match.group(2).replace('"', "'")  # Replace double quotes with single quotes
#         # Return the updated string with modified values
#         return f'{{ "reasoning": "{reasoning}", "paraphrase": "{paraphrase}" }}'
    
#     # Use re.sub to apply the replacement function
#     updated_text = re.sub(regex, replace_quotes, text, flags=re.DOTALL | re.MULTILINE)
    
#     return updated_text

def preprocess_text(text:str):
    text = preprocess_based_on_model_tags(text)
    text = strip_markdown(text)
    text = text.replace("\t","").replace("\\t","").replace("\\","").replace('""','').replace("**","").replace("```","")
    if "json" in text:
        text = text.replace("json","").replace("}","").replace("{","").replace('"',"")
    # text = eliminate_double_quotes_inside_value(text)
    
    return text.strip()

def extract_paraphrase(text):
    # Use a regular expression to extract the content after 'Final Paraphrase:' until a newline or end of document
    text = re.sub(r' *\n+"', '"', text)  # Replace \n followed by "
    text = re.sub(r' *\n+\'', "'", text)  # Replace \n followed by '
    text = re.sub(r': *\n+', ':', text)  # Replace : followed by \n
    text = re.sub(r'Final Paraphrase *\n+', 'Final Paraphrase', text)
    match = re.search(r"Final Paraphrase(.*?)(\n|$)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None
    
def extract_reasoning(text):
    # Use a regular expression to extract the content between 'Final Reasoning:' and 'Final Paraphrase:'
    match = re.search(r"Final Reasoning(.*?)Final Paraphrase", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None

def extract_reasoning_adaptative(text:str,pattern:str):
    # Use a regular expression to extract the content between 'Final Reasoning:' and 'Final Paraphrase:'
    match = re.search(r"Final Reasoning(.*?)"+pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None

def extract_adaptative(text:str,pattern_start:str=None,pattern_end:str=None):
    # Use a regular expression to extract the content after 'Final Paraphrase:' until a newline or end of document

    text = re.sub(r' *\n+"', '"', text)  # Replace \n followed by "
    text = re.sub(r' *\n+\'', "'", text)  # Replace \n followed by '
    text = re.sub(r': *\n+', ':', text)  # Replace : followed by \n
    text = re.sub(re.escape(pattern_start) + r' *\n+', pattern_start, text)
    if pattern_end == None:

        match = re.search(pattern_start+r"(.*?)(\n|$)", text, re.DOTALL)
    else:
        
        match = re.search(pattern_start+r"(.*?)"+pattern_end, text, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        return None
    
def postprocess_text(text:str):
    text = text.replace("```","")
    text = text.replace("**","")
    
    text = text.strip()

    text = text[1:] if text.startswith(":") else text
    text = text.strip()
    text = text[1:] if text.startswith("'") else text
    text = text[:-1] if text.endswith("'") else text

    text = text[:-1] if text.endswith('"') else text
    text = text[1:]if text.startswith('"') else text
    text = text.replace("'.",".")
    text = text.replace('".','.')
    
    return text.strip()
def extract_dict(text:str,enforcer:StructuredOutputEnforcer=None,tokenizer:AutoTokenizer=None):
    # We eliminate any markdown tags
    
    result_dict = {"reasoning":None,"paraphrase":None}
    #Base pattern
    if "Final Reasoning" and "Final Paraphrase" in text:
        reasoning = extract_reasoning(text)
        if reasoning != None:
            reasoning = postprocess_text(reasoning)
            result_dict["reasoning"] = reasoning
        paraprase = extract_paraphrase(text)
        if paraprase != None:
            paraprase = paraprase.replace("\n","")
            paraprase = paraprase.strip()
            paraprase = postprocess_text(paraprase)
            
            result_dict["paraphrase"] = paraprase
        
        
        result_dict["paraphrase"] = paraprase
    else:
        if "Final Reasoning" in text and result_dict["reasoning"]==None:
            possible_outcome_reasoning = extract_reasoning_adaptative(text,"The final paraphrase generated is:")

            if possible_outcome_reasoning != None:
                possible_outcome_reasoning = postprocess_text(possible_outcome_reasoning)
                result_dict["reasoning"] = possible_outcome_reasoning

        if result_dict["paraphrase"]==None:
            possible_outcome_paraphrase = extract_adaptative(text,pattern_start="The final paraphrase generated is")
            if possible_outcome_paraphrase != None:
                possible_outcome_paraphrase = possible_outcome_paraphrase.replace("\n","")
                possible_outcome_paraphrase = possible_outcome_paraphrase.strip()
                possible_outcome_paraphrase = postprocess_text(possible_outcome_paraphrase)
                
                if "This version" in possible_outcome_paraphrase and "The final paraphrase generated is" in text:
                    possible_outcome_paraphrase = extract_adaptative(text,pattern_start="The final paraphrase generated is",pattern_end="This version")
                    if possible_outcome_paraphrase != None:
                        possible_outcome_paraphrase = possible_outcome_paraphrase.replace("\n","")
                        possible_outcome_paraphrase = possible_outcome_paraphrase.strip()
                        possible_outcome_paraphrase = postprocess_text(possible_outcome_paraphrase)
                        
                result_dict["paraphrase"] = possible_outcome_paraphrase.replace("\n","")
           
        
        if result_dict["reasoning"]==None or result_dict["paraphrase"]==None:
            if "final_reasoning" in text and "final_paraphrase" in text:

                reasoning = extract_adaptative(text,"final_reasoning","final_paraphrase")
                
                paraprase = extract_adaptative(text,"final_paraphrase")
                
                if reasoning != None:
                    reasoning = postprocess_text(reasoning)
                    result_dict["reasoning"] = reasoning
                if paraprase != None:
                    paraprase = paraprase.replace("\n","")
                    paraprase = paraprase.strip()
                    paraprase = postprocess_text(paraprase)
                    result_dict["paraphrase"] = paraprase.replace("\n","")
            elif "final_reasoning" in text and "Final Answer" in text:
                reasoning = extract_adaptative(text,"final_reasoning","Final Answer")
                paraprase = extract_adaptative(text,"Final Answer","]")
                if reasoning != None:
                    reasoning = postprocess_text(reasoning)
                    result_dict["reasoning"] = reasoning
                if paraprase != None:
                    paraprase = paraprase.replace("\n","")
                    paraprase = paraprase.strip()
                    paraprase = postprocess_text(paraprase)
                    paraprase = paraprase.replace("[ boxedext", "")
                    paraprase = paraprase.replace("]","")
                    result_dict["paraphrase"] = paraprase.replace("\n","")
            elif "final_reasoning" in text and "Proposed Paraphrase" in text:
                reasoning = extract_adaptative(text,"final_reasoning","Proposed Paraphrase")
                paraprase = extract_adaptative(text,"Proposed Paraphrase","Version 2")
                if reasoning != None:
                    reasoning = postprocess_text(reasoning)
                    result_dict["reasoning"] = reasoning
                if paraprase != None:
                    paraprase = paraprase.replace("Version 1:", "")
                    paraprase = paraprase.strip()
                    paraprase = paraprase.replace("\n","")
                    paraprase = postprocess_text(paraprase)
                    result_dict["paraphrase"] = paraprase
            elif "final_reasoning" in text and "Refine the Paraphrase" in text:
                reasoning = extract_adaptative(text,"final_reasoning","Refine the Paraphrase")
                paraprase = extract_adaptative(text,"We will suggest a more refined paraphrase","This version")
                if reasoning != None:
                    reasoning = postprocess_text(reasoning)
                    result_dict["reasoning"] = reasoning
                if paraprase != None:
                    
                    paraprase = paraprase.replace("\n","")
                    paraprase = paraprase.strip()
                    paraprase = postprocess_text(paraprase)
                    result_dict["paraphrase"] = paraprase
            elif "final_reasoning" in text and "Transforming the Sentence" in text:
                reasoning = extract_adaptative(text,"final_reasoning","Transforming the Sentence")
                
                paraprase = extract_adaptative(text,"The paraphrase provided is")
                if reasoning != None:
                    reasoning = postprocess_text(reasoning)
                    result_dict["reasoning"] = reasoning
                if paraprase != None:
                    paraprase = paraprase.replace("\n","")
                    paraprase = paraprase.strip()
                    paraprase = postprocess_text(paraprase)
                    result_dict["paraphrase"] = paraprase
            elif "final_reasoning" in text and "Analyze Changes for Paraphrase:" in text:
                reasoning = extract_adaptative(text,"final_reasoning","Analyze Changes for Paraphrase:")
                paraprase = None
                if "Compare with the paraphrase" in text:
                    paraprase = extract_adaptative(text,"Compare with the paraphrase")
                elif "Consider the provided paraphrase" in text:
                    paraprase = extract_adaptative(text,"Consider the provided paraphrase")

                if reasoning != None:
                    reasoning = postprocess_text(reasoning)
                    result_dict["reasoning"] = reasoning
                if paraprase != None:
                    paraprase = paraprase.replace("\n","")
                    paraprase = paraprase.strip()
                    paraprase = postprocess_text(paraprase)
                    result_dict["paraphrase"] = paraprase
            elif "final_reasoning" in text and "Generate Paraphrase" in text:
                reasoning = extract_adaptative(text,"final_reasoning","Generate Paraphrase")
                paraprase = extract_adaptative(text,"The revised sentence will be")
                if reasoning != None:
                    reasoning = postprocess_text(reasoning)
                    result_dict["reasoning"] = reasoning
                if paraprase != None:
                    paraprase = paraprase.replace("\n","")
                    paraprase = paraprase.strip()
                    paraprase = postprocess_text(paraprase)
                    result_dict["paraphrase"] = paraprase
            elif "final_reasoning" in text and "Construct Paraphrase:" in text:
                reasoning = extract_adaptative(text,"final_reasoning","Construct Paraphrase:")
                paraprase = extract_adaptative(text,"New sentence:")
                if reasoning != None:
                    reasoning = postprocess_text(reasoning)
                    result_dict["reasoning"] = reasoning
                if paraprase != None:
                    paraprase = paraprase.replace("\n","")
                    paraprase = paraprase.strip()
                    paraprase = postprocess_text(paraprase)
                    result_dict["paraphrase"] = paraprase
        

        #The final paraphrase generated is:
        #The final paraphrase generated is: 'Rory, let's have a respectful discussion on Reddit. Everyone is welcome here.' This version removes the aggressive language, commands, and personal targeting, promoting a more inclusive and respectful tone.\n\n```"
    if result_dict["reasoning"]==None or result_dict["paraphrase"]==None or result_dict["reasoning"]=="" or result_dict["paraphrase"]=="":
        if enforcer != None:
            text=tokenizer.encode(text,return_tensors="pt", truncation=True,add_special_tokens=False)
            
            length = len(text[0])
            text = tokenizer.decode(text[0])
            if length > 2048:
                return "Model hallucinated"
            
            result_dict = enforcer.obtain_clean_response(text)
            if result_dict != None:
                return result_dict
        return "Model hallucinated"
    
    return result_dict

def row_dict_value_empty(row):
    if type(row)==dict:
        for key in row:
            if row[key]=="":
                return True
    return False
    
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--folder",type=str,help="Folder we want to extract the paraphrase from",default="qwq_preview")#llama3_1_8B,openO1,marco-o1,qwen2_5_7B,qwq_preview
    args.add_argument("--use_enforcer",type=str,help="Use a model that enforces the structure of the output",default="True")
    args.add_argument("--model_config",type=str,help="Model configuration to use for the enforcer",default="qwen2_5")#phi_mini
    folder = args.parse_args().folder
    use_enforcer = args.parse_args().use_enforcer
    model_config = args.parse_args().model_config
    
    print(f"Folder: {folder}, use enforcer: {use_enforcer}, model config: {model_config}")
    if use_enforcer == "True":
        model_config = project_path+f"src/utils/llms/configs/{model_config}.yaml"
        enforcer = StructuredOutputEnforcer(model_config)
    else:
        enforcer = None
    output_path = project_path+"data/interim/final_paraphrases_postprocess/"+folder+"/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    data_path = project_path+"data/interim/paraphrasing_experiment/"+folder+"/"
    files = glob.glob(data_path+"*.csv")
    number_of_non_correct_json = 0
    number_of_hallucinations = 0
    config_path = project_path+f"src/utils/llms/configs/{folder}.yaml"
    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)['config']
        except yaml.YAMLError as exc:
            print(exc)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    files.sort(reverse=True)
    files = [files[6]]

    for file in files:
        print(f"File: {file}")
        df = pd.read_csv(file)
        # df["Correct"] = False
        number_of_hallucinations = 0
        for idx in tqdm(range(len(df))):
            reasoning_input = df["result"].iloc[idx]
            reasoning_input = preprocess_text(reasoning_input)
            model_output = extract_dict(reasoning_input,enforcer,tokenizer)
            if model_output == "Model hallucinated":
                number_of_hallucinations += 1
            df.at[idx,"result"] = model_output
        file = file.split("/")[-1]
        df_with_hallucinations = df[(df["result"]=="Model hallucinated") | (df["result"]=="") | (df["result"].apply(row_dict_value_empty))]
        print(f"Number of non correct json: {number_of_non_correct_json}, number of hallucinations: {len(df_with_hallucinations)}, file: {file}")
        df.to_csv(output_path+file,index=False)
