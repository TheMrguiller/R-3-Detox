import sys
import os
project_path = os.path.abspath(__file__).split('src')[0]
sys.path.append(project_path)
from src.utils.llms.llm_vllm_offline_proxy import LLMVLLMOfflineProxy
from src.utils.llms.llm_huggingface_proxy import LLMHuggingFace
from src.utils.vector_store_paraphrase import Vector_store_Paraphrase
from src.utils.structured_output.paraphrase_generation import ParaphraseGeneration
from langchain_core.output_parsers import JsonOutputParser
from transformers import AutoTokenizer
import json
from typing import List
import yaml
from tqdm import tqdm

# class ParaphraseExperimentOnline:

#     def __init__(self,model_config:str,experiment_type:str,num_examples:int):
#         """
#         Args:
#             model_config (str): Path to the model configuration file
#             experiment_type (str): Type of experiment to run. Can be one of "zero_shot", "one_shot", or "few_shot"
#             num_examples (int): Number of examples to generate
#         """
#         self.llm = LLMVLLMOfflineProxy(model_config)
#         # self.llm = LLMHuggingFace(model_config)
#         self.tokenizer = AutoTokenizer.from_pretrained(self.llm.config["model_name"])
#         self.max_tokens = self.llm.config["max_tokens"]
#         self.experiment_type = experiment_type
#         self.num_examples = num_examples
#         self.vector_store = Vector_store_Paraphrase(db_path=project_path+"data/vector_store/")
#         # print(f"Count:{self.vector_store.get_collection_count()}")
#         self.parser = JsonOutputParser(pydantic_object=ParaphraseGeneration)
#         self.parser = 'The output should be formatted as a JSON instance that conforms to the JSON schema below.Here is the output schema:\n```json\n{"reasoning": "The reasoning process generated", "paraphrase": "The final paraphrase generated"}\n```'
#         with open(project_path + "src/utils/llms/prompts/generate_paraphrasing.yaml", "r") as file:
#             base_prompt = yaml.safe_load(file) 
#         self.zero_shot_system_prompt = base_prompt["system_prompt_zero_shot"].replace("\\n","\n").replace("\\t","\t")
#         self.base_prompt = base_prompt["base_prompt"].replace("\\n","\n").replace("\\t","\t")
#         self.user_prompt = base_prompt["user_prompt"].replace("\\n","\n").replace("\\t","\t")

#     def generate_few_shot_examples(self,examples:List[dict]):
#         example_prompt="\nBelow are some examples of how to complete the task.\n"
#         for idx,example in enumerate(examples):
#             sentence = example["sentence"]
#             shap_values = example["shap_values"]
#             input_ = self.base_prompt.format(sentence=sentence,toxic_words=shap_values)
#             output=ParaphraseGeneration(reasoning=example["reasoning"].replace('"',"'"),paraphrase=example["paraphrase"]).model_dump()
#             example_prompt+=f"**Example {str(idx+1)}:**\n**Input:**\n{input_}**Output:**\n{output}\n"
#         return example_prompt
    
#     def generate_prompt_template(self,sentence:str,shap_values:List[str],label:int):
#         if self.experiment_type == "zero_shot":
#             prompt = self.base_prompt.format(sentence=sentence,toxic_words=shap_values)
#             return self.zero_shot_system_prompt + f"{self.parser}",self.user_prompt+prompt 
#             return self.zero_shot_system_prompt + f"{self.parser.get_format_instructions()}",self.user_prompt+prompt 
            
#         else:
#             prompt = self.base_prompt.format(sentence=sentence,toxic_words=shap_values)
#             examples = self.vector_store.query(sentence,label,n_results=self.num_examples)
#             examples_string=self.generate_few_shot_examples(examples)
#             return self.zero_shot_system_prompt + f"{self.parser}"+ examples_string,self.user_prompt+prompt 
#             return self.zero_shot_system_prompt + f"{self.parser.get_format_instructions()}"+ examples_string,self.user_prompt+prompt 
            
            

#     def run_experiment(self,sentences:List[str],labels:List[int],shap_values:List[List[str]],sources:List[str],batch_size:int=32):
#         results = []
#         for i in tqdm(range(0,len(sentences),batch_size)):
#             batch_sentences = sentences[i:i+batch_size]
#             batch_labels = labels[i:i+batch_size]
#             batch_shap_values = shap_values[i:i+batch_size]
#             # batch_sources = sources[i:i+batch_size]
#             prompts=[]
#             #system_prompts=[]
#             is_length_valid = []
#             for sentence,label,shap_value in zip(batch_sentences,batch_labels,batch_shap_values):
#                 system_prompt,prompt=self.generate_prompt_template(sentence=sentence,label=label,shap_values=shap_value)
#                 token_lenght=len(self.tokenizer.tokenize(system_prompt+"\n"+prompt))+20
#                 print(f"Token lenght:{token_lenght}")
#                 if token_lenght>=self.max_tokens-1000: #We are leaving some space for the model to generate the output
#                     is_length_valid.append(False)
#                 else:
#                     is_length_valid.append(True)
#                 prompts.append(system_prompt+"\n"+prompt)
#                 # prompts.append(prompt)
#                 #system_prompts.append(system_prompt)
#             #TODO: Handle the case where the prompt is too long
#             #system_prompts_= [system_prompts[i] for i in range(len(system_prompts)) if is_length_valid[i]]
#             prompts_ = [prompts[i] for i in range(len(prompts)) if is_length_valid[i]]
#             if len(prompts_)!= len(prompts):
#                 results.extend([None]*batch_size)
#                 return results
#             else:
#                 # result = self.llm.query(system_messages=system_prompts_,user_messages=prompts_)
#                 result = self.llm.query(user_messages=prompts_)
#                 #TODO: Handle the result whre the prompt was too long
#                 results.extend(result)
#         return results
    
class ParaphraseExperimentOffline:

    def __init__(self,model_config:str,experiment_type:str,num_examples:int):
        """
        Args:
            model_config (str): Path to the model configuration file
            experiment_type (str): Type of experiment to run. Can be one of "zero_shot", "one_shot", or "few_shot"
            num_examples (int): Number of examples to generate
        """
        # self.llm = LLMVLLMOfflineProxy(model_config)
        self.llm = LLMHuggingFace(model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm.config["model_name"])
        self.max_tokens = self.llm.config["max_tokens"]
        self.experiment_type = experiment_type
        self.num_examples = num_examples
        try:
            with open(project_path+"data/processed/precomputed_few_shots_examples/precomputed_few_shot_examples.json","r") as f:
                self.precomputed_examples = json.load(f)
        except:
            raise Exception("Precomputed examples not found")
        # self.vector_store = Vector_store_Paraphrase(db_path=project_path+"data/vector_store/")
        # print(f"Count:{self.vector_store.get_collection_count()}")
        # self.parser = JsonOutputParser(pydantic_object=ParaphraseGeneration)
        # self.parser = 'The output should be formatted as a JSON instance that conforms to the JSON schema below.Here is the output schema:\n```json\n{"reasoning": "The reasoning process generated", "paraphrase": "The final paraphrase generated"}\n```'
        self.parser = "The final output must be the following plain text:\n ```Final Reasoning: The reasoning process generated\n Final Paraphrase: The final paraphrase generated```"
        with open(project_path + "src/utils/llms/prompts/generate_paraphrasing.yaml", "r") as file:
            base_prompt = yaml.safe_load(file) 
        self.zero_shot_system_prompt = base_prompt["system_prompt_zero_shot"].replace("\\n","\n").replace("\\t","\t")
        self.base_prompt = base_prompt["base_prompt"].replace("\\n","\n").replace("\\t","\t")
        self.user_prompt = base_prompt["user_prompt"].replace("\\n","\n").replace("\\t","\t")

    # def generate_few_shot_examples(self,examples:List[dict]):
    #     example_prompt="\nBelow are some examples of how to complete the task.\n"
    #     for idx,example in enumerate(examples):
    #         sentence = example["sentence"]
    #         shap_values = example["shap_values"]
    #         input_ = self.base_prompt.format(sentence=sentence,toxic_words=shap_values)
    #         output=ParaphraseGeneration(reasoning=example["reasoning"].replace('"',"'"),paraphrase=example["paraphrase"]).model_dump()
    #         example_prompt+=f"**Example {str(idx+1)}:**\n**Input:**\n{input_}**Output:**\n{output}\n"
    #     return example_prompt

    def generate_few_shot_examples(self,examples:List[dict]):
        example_prompt="\nBelow are some examples of how to complete the task.\n"
        for idx,example in enumerate(examples):
            sentence = example["sentence"]
            shap_values = example["shap_values"]
            input_ = self.base_prompt.format(sentence=sentence,toxic_words=shap_values)
            reasoning = example["reasoning"].replace('"',"'")
            paraphrase = example["paraphrase"]
            # output=ParaphraseGeneration(reasoning=,paraphrase=example["paraphrase"]).model_dump()
            example_prompt+=f'**Example {str(idx+1)}:**\n{input_}\n```Final Reasoning:"{reasoning}"\nFinal Paraphrase:"{paraphrase}"```\n'
        return example_prompt
    
    def generate_prompt_template(self,sentence:str,shap_values:List[str],label:int,index:int):
        if self.experiment_type == "zero_shot":
            prompt = self.base_prompt.format(sentence=sentence,toxic_words=shap_values)
            return self.zero_shot_system_prompt + f"{self.parser}" , self.user_prompt+prompt
            return self.zero_shot_system_prompt + f"{self.parser}",self.user_prompt+prompt 
            return self.zero_shot_system_prompt + f"{self.parser.get_format_instructions()}",self.user_prompt+prompt 
            
        else:
            prompt = self.base_prompt.format(sentence=sentence,toxic_words=shap_values)
            examples = self.precomputed_examples[str(index)]
            examples = examples[:self.num_examples]
            # examples = self.vector_store.query(sentence,label,n_results=self.num_examples)
            examples_string=self.generate_few_shot_examples(examples)
            return self.zero_shot_system_prompt + f"{self.parser}" + examples_string, self.user_prompt+prompt
            return self.zero_shot_system_prompt + f"{self.parser}"+ examples_string,self.user_prompt+prompt 
            return self.zero_shot_system_prompt + f"{self.parser.get_format_instructions()}"+ examples_string,self.user_prompt+prompt 
            
            

    def run_experiment(self,sentences:List[str],labels:List[int],shap_values:List[List[str]],sources:List[str],indexes:List[int],batch_size:int=32):
        results = []
        for i in tqdm(range(0,len(sentences),batch_size)):
            batch_sentences = sentences[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            batch_shap_values = shap_values[i:i+batch_size]
            batch_index = indexes[i:i+batch_size]
            # batch_sources = sources[i:i+batch_size]
            prompts=[]
            #system_prompts=[]
            is_length_valid = []
            for sentence,label,shap_value,index in zip(batch_sentences,batch_labels,batch_shap_values,batch_index):
                system_prompt,prompt=self.generate_prompt_template(sentence=sentence,label=label,shap_values=shap_value,index=index)
                token_lenght=len(self.tokenizer.tokenize(system_prompt+"\n"+prompt))+20
                print(f"Token lenght:{token_lenght}")
                if token_lenght>=self.max_tokens-1000: #We are leaving some space for the model to generate the output
                    is_length_valid.append(False)
                else:
                    is_length_valid.append(True)
                prompts.append(system_prompt+"\n"+prompt)
                # prompts.append(prompt)
                #system_prompts.append(system_prompt)
            #TODO: Handle the case where the prompt is too long
            #system_prompts_= [system_prompts[i] for i in range(len(system_prompts)) if is_length_valid[i]]
            prompts_ = [prompts[i] for i in range(len(prompts)) if is_length_valid[i]]
            if len(prompts_)!= len(prompts):
                results.extend([None]*batch_size)
                return results
            else:
                # result = self.llm.query(system_messages=system_prompts_,user_messages=prompts_)
                result = self.llm.query(user_messages=prompts_)
                #TODO: Handle the result whre the prompt was too long
                results.extend(result)
        return results

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv(project_path+"data/processed/final_few_shot_reasoning/few_shot_reasoning.csv")
    df.drop(columns=["reasoning"],inplace=True)
    df = df[df["source"]!="non_toxic"]
    df.reset_index(drop=True,inplace=True)
    sentences = df["sentence"].tolist()
    labels = df["label"].tolist()
    shap_values = df["shap_values"].apply(eval).tolist()
    sources = df["source"].tolist()
    model_config = project_path + "src/utils/llms/configs/marco-o1.yaml"
    for experiment_type in ["one_shot","zero_shot","few_shot"]:
        if experiment_type == "few_shot":
            num_examples = 2
        elif experiment_type == "one_shot":
            num_examples = 1
        elif experiment_type == "zero_shot":
            num_examples = 0
        experiment = ParaphraseExperimentOffline(model_config,experiment_type,num_examples)
        results = experiment.run_experiment(sentences=sentences,labels=labels,shap_values=shap_values,sources=sources)
        df_try = pd.DataFrame(columns=["sentence","label","shap_values","result"])
        df_try["sentence"] = sentences
        df_try["label"] = labels
        df_try["shap_values"] = shap_values
        df_try["result"] = results
        if not os.path.exists(project_path+f"data/interim/final_response_paraphrase/"):
            os.makedirs(project_path+f"data/interim/final_response_paraphrase/")
        df_try.to_csv(project_path+f"data/interim/final_response_paraphrase/paradetox_{experiment_type}_results.csv",index=False)
            
                



