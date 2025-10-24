import yaml
from langchain_openai import ChatOpenAI
import re
import os
import random
from typing import List
random.seed(42)
class LLMOpenAIProxy:
    config = None
    llm = None

    def __init__(self, yaml_config_file) -> None:
        self._read_yaml_config(yaml_config_file)
        self.llm = self._set_up_llm(temperature=1.0)

        # self._get_prompt("Hello", "How are you?", "I am fine.")

    def _set_up_llm(self, temperature):
         
        openai = ChatOpenAI(
            openai_api_key= os.getenv("OPENAI_API_KEY") if self.config['api_key'] else "Empty",

            model_name=self.config['model_name'],
            temperature=temperature,
        )
        # set temperature
        return openai

    def _read_yaml_config(self, yaml_config_file):
        # read yaml config file into a dictionary

        with open(yaml_config_file, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)['config']
            except yaml.YAMLError as exc:
                print(exc)
    def random_capitalize_with_prob(self,text, prob=0.5):
        result = ''.join([char.upper() if random.random() < prob else char.lower() for char in text])
        return result

    def query(self, system_messages:List[str]=None, user_messages:List[str]=None, assistant_prefill=""):
        messages = []
        if system_messages is None:
            for user_message in user_messages:
                prompt = self.random_capitalize_with_prob(user_message)
                message = [
                ("user", prompt),
                ]
                messages.append(message)
        elif user_messages is None:
            for system_message in system_messages:
                prompt = self.random_capitalize_with_prob(system_message)
                message = [
                ("user", prompt),
                ]
                messages.append(message)
        else:
            for system_message, user_message in zip(system_messages, user_messages):
                prompt =  system_message+"\n"+user_message
                

                message = [
                    ("user", prompt),
                ]
                messages.append(message)
        
        try:
            response = self.llm.batch(messages)
        except Exception as e:
            print(e)
            return ["Error"]*len(system_messages)
        response_content_list = []
        for response_content in response:
            response_content_list.append(response_content.content)
        return response_content_list
    
if __name__ == "__main__":
    project_path=os.path.abspath(__file__).split('src')[0]
    config_path = project_path+"src/utils/llms/configs/openai_o1.yaml"
    llm = LLMOpenAIProxy(config_path)
    system = """Given a toxic sentence, your task is to perform a detailed reasoning process. First, analyze the words or phrases in the sentence that convey toxic behavior, explaining why they are toxic within the context. Then, determine the specific changes needed to rewrite the sentence in a non-toxic manner while preserving its original meaning (the target non-toxic sentence is the one in "Paraphrase"). Provide a clear explanation of how these changes remove the toxicity.

To support your reasoning, consider any external information provided, such as a non-toxic rewritten sentence, possible relevant words that could express toxicity (though not necessarily), and the label. However, do not acknowledge or reference the existence of this external information during your reasoning process. Do not state that you have external information saying that it is toxic, that you have some initial relevant words, or that you have the final paraphrase sentence.
"""
    user ="""Toxic sentence: Clinton is a loser, twice.
Relevant words: loser
Label: Toxic
Paraphrase: Clinton lost twice."""
    system = "Hi"
    user = "Hello"

    print(llm.query([system,system], [user,user]))
    # print(llm.query("Hello", "How are you?", "I am fine."))
