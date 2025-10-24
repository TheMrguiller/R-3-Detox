import sys
import os
project_path = os.path.abspath(__file__).split('src')[0]
sys.path.append(project_path)
from src.utils.llms.llm_vllm_offline_proxy import LLMVLLMOfflineProxy
from src.utils.llms.llm_vllm_server_proxy import LLMVLLMProxy
from src.utils.llms.llm_huggingface_proxy import LLMHuggingFace
from src.utils.vector_store_paraphrase import Vector_store_Paraphrase
from src.utils.structured_output.paraphrase_generation import ParaphraseGeneration
from src.data.extract_paraphrase_from_experiment_v2 import extract_dict,preprocess_text
from src.utils.chains.structured_output_enforcer import StructuredOutputEnforcer
from langchain_core.output_parsers import JsonOutputParser
from transformers import AutoTokenizer
import json
from typing import List
import yaml
from tqdm import tqdm
import torch

class ParaphraseExperimentOfflineWithErrorsFixing:

    def __init__(self,generation_model_config:str,data_extractor_config:str,experiment_type:str,num_examples:int):
        """
        Args:
            model_config (str): Path to the model configuration file
            experiment_type (str): Type of experiment to run. Can be one of "zero_shot", "one_shot", or "few_shot"
            num_examples (int): Number of examples to generate
        """
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm = LLMVLLMOfflineProxy(generation_model_config,temperature=0.7,top_p=0.9)
        # if self.device == "cuda":
        #     self.llm = LLMVLLMOfflineProxy(generation_model_config,temperature=0.7,top_p=0.9)
        # else:
        #     self.llm = LLMHuggingFace(generation_model_config,temperature=0.7,top_p=0.9,do_sample=True)
        self.temperature = 0.7
        self.top_p = 0.9
        self.seed = 42
        self.enforcer=StructuredOutputEnforcer(data_extractor_config)
        # self.llm = LLMHuggingFace(model_config)
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
            if type(self.llm) == LLMHuggingFace:
                self.llm.change_parameters(temperature=self.temperature,do_sample=True,seed=self.seed,top_p=self.top_p,num_return_sequences=1)
            else:
                self.llm.change_parameters(temperature=self.temperature,seed=self.seed,top_p=self.top_p,num_return_sequences=1)

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
                
            prompts_ = [prompts[i] for i in range(len(prompts)) if is_length_valid[i]]
            if len(prompts_)!= len(prompts): #If the prompt was too long we just finish the experiment from here
                results.extend([None]*batch_size)
                return results
            else:

                correct_format =False
                result_correctness_dict = {}
                #Generate the initial elements to generate and thus correct
                for idx in range(len(prompts_)):
                    result_correctness_dict[idx] = False
                final_result = [""]*len(prompts_)
                # Generate the result until a correct format is obtained
                verification_prompts = []
                number_of_trials = 0
                while not correct_format:
                    #Generate prompt and get the result
                    for idx in range(len(result_correctness_dict.keys())):
                        if result_correctness_dict[idx]:
                            continue
                        verification_prompts.append(prompts_[idx])
                    result = self.llm.query(user_messages=verification_prompts)
                    #Check if the result is correct
                    print(result)
                    num = 0
                    for idx in range(len(result_correctness_dict.keys())):
                        if result_correctness_dict[idx]:
                            continue
                        result[num] = preprocess_text(result[num])
                        post_process_result = extract_dict(text=result[num],enforcer=self.enforcer,tokenizer=self.tokenizer)
                        if post_process_result == "Model hallucinated":
                            result_correctness_dict[idx] = False
                        else:
                            if final_result[idx]=="":
                                if any([value=="" for value in post_process_result.values()]):
                                    result_correctness_dict[idx] = False
                                    continue 
                                result_correctness_dict[idx] = True
                                final_result[idx]=post_process_result
                        num+=1
                    print(f"Number of trials:{number_of_trials}")
                    all_values = list(result_correctness_dict.values())
                    verification_prompts.clear()
                    #If all the values are True, then the format is correct
                    if all(all_values):
                        correct_format = True
                        results.extend(final_result)
                    number_of_trials+=1
                    if type(self.llm) == LLMHuggingFace:
                        self.llm.change_parameters(temperature=self.temperature,do_sample=True,seed=self.seed+number_of_trials,top_p=self.top_p,num_return_sequences=1)
                    else:
                        self.llm.change_parameters(temperature=self.temperature,seed=self.seed+number_of_trials,top_p=self.top_p,num_return_sequences=1)
                    if number_of_trials>5:
                        results.extend([None]*batch_size)
                        print("The model is not able to generate the correct format")
                        return results
        return results