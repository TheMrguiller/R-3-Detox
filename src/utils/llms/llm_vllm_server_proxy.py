import yaml
from langchain_community.llms import VLLMOpenAI
import re
import os
from typing import List


class LLMVLLMProxy:
    config = None
    llm = None

    def __init__(self, yaml_config_file,temperature=0.0) -> None:
        self._read_yaml_config(yaml_config_file)
        self.llm = self._set_up_llm(temperature=temperature)

        # self._get_prompt("Hello", "How are you?", "I am fine.")

    def _set_up_llm(self, temperature):
        openai = VLLMOpenAI(
            openai_api_key= os.getenv("API_KEY") if self.config['api_key'] else "Empty",
            openai_api_base=self.config['endpoint'],
            model_name=self.config['model_name'],
            temperature=temperature,
            max_tokens=4096,
            request_timeout=600,
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

    def _get_user_turn(self, content: str):
        return self._get_complete_turn_string(self.config['USER_TURN_ID'], content)

    def _get_system_turn(self, content: str):
        return self._get_complete_turn_string(self.config['SYSTEM_TURN_ID'], content)

    def _get_assistant_turn(self, content: str):
        return f"{self.config['START_ID_HEADER']}{self.config['ASSISTANT_TURN_ID']}{self.config['END_ID_HEADER']}{self.config['START_OF_TURN']}{content}"

    def _get_complete_turn_string(self, turn_id: str, content: str):
        return f"{self.config['START_ID_HEADER']}{turn_id}{self.config['END_ID_HEADER']}{self.config['START_OF_TURN']}{content}{self.config['END_OF_TURN']}"

    def _get_prompt(self, system_message, user_message, assistant_prefill):
        return f"{self.config['BOT']}{self._get_system_turn(system_message)}{self._get_user_turn(user_message)}{self._get_assistant_turn(assistant_prefill)}"

    def postprocess_response(self, response:str)->dict:
        preprocess_response = {
            "reasoning": response,
            "output": response,
        }
        if "special_thought_tokens" in self.config:
            token = self.config["special_thought_tokens"]
            # Escape special characters in the token to avoid regex issues
            escaped_token = re.escape(token)
            patron = f'{escaped_token}(.*?){escaped_token}'
            match = re.search(patron, response)
            if match:
                preprocess_response["reasoning"]=match.group(1)
        if "special_output_tokens" in self.config:
            token = self.config["special_output_tokens"]
            # Escape special characters in the token to avoid regex issues
            escaped_token = re.escape(token)
            patron = f'{escaped_token}(.*?){escaped_token}'
            match = re.search(patron, response)
            if match:
                preprocess_response["output"]=match.group(1)
        return preprocess_response
    
    def query(self, system_messages:List[str], user_messages:List[str], assistant_prefill=""):
        messages = []
        for system_message, user_message in zip(system_messages, user_messages):
            prompt = self._get_prompt(
                system_message, user_message, assistant_prefill)
            messages.append(prompt)
        # print(prompt)
        response = self.llm.batch(messages)
        response_content_list = []
        for response_content in response:
            response_content_list.append(response_content)
        return response_content_list
if __name__=="__main__":
    project_path=os.path.abspath(__file__).split('src')[0]
    config_path = project_path+"src/utils/llms/configs/qwen2_5.yaml"
    llm = LLMVLLMProxy(config_path)
    system_message= ("You are given a reasoning process of a model and your task is to clean the reasoning process by eliminating the following information:\n"
            # "\t1-Omit the parts where the models explain the task that they are performing, in this case generate the reasoning process  where a toxic sentence is converted into a non-toxic sentence.\n"
            "\t1-Omit the part where the model says the task it is performing.\n"
            "\t2-If the model acknowledges that the sentence is toxic before doing the toxic words analysis, you should remove the acknowledgment.\n"
            "\t3-If the model acknowledges that a non-toxic sentence is given, you should remove the acknowledgment.\n"
            "\t4-Eliminate any sentence that states the final answer.\n"
            "Maintaing the original reasoning step flow. Do not add any explanations or apologies in your response."
    )
    prompt = (
"""
Reasoning Process:
'''So I've got this sentence: "Clinton is a loser, twice." And I need to figure out why it's considered toxic and how to make it non-toxic while keeping the same meaning. Let's break this down step by step.

First off, the word "loser" jumps out at me. It's pretty clear that calling someone a "loser" is insulting. It's a derogatory term that demeans the person's worth and achievements. In this context, it's being used to describe Clinton, suggesting that he's unsuccessful or inadequate in some way. That's why it's toxic—it's harmful and disrespectful language.

Now, the phrase "twice" likely refers to Clinton losing elections or something similar. So, the sentence is pointing out that Clinton has lost twice, and by calling him a "loser," it's emphasizing his failures in a negative and hurtful way.

To make this non-toxic, I need to convey the same information without using insulting language. Instead of calling Clinton a "loser," I can simply state that he lost twice. That's factual and doesn't involve name-calling or derogatory labels.

So, the paraphrase would be: "Clinton lost twice."

This version removes the toxicity by eliminating the insulting term and just stating the fact. It's straightforward and respectful.

Let me think about this some more. Is there any other way this sentence could be interpreted as toxic? Well, the use of the comma after "loser" might be slightly off, but I think the main issue is the word "loser" itself.

Also, depending on the context, repeating "twice" might imply frustration or emphasis on his losses, which could also come across as negative. But in the paraphrase, by just saying "lost twice," it's neutral and doesn't carry that negative tone.

I should also consider if there's any other way to rephrase it that might be even more positive or neutral. For example, maybe saying, "Clinton was unsuccessful in his attempts two times." But that seems a bit wordy and still focuses on the negative aspect.

Alternatively, if I had more context, maybe I could frame it in a way that highlights learning or growth from those losses. But with the information given, that might not be appropriate.

So, sticking with "Clinton lost twice" seems to be the most straightforward and non-toxic way to convey the original message without the insult.

One more thing: I should make sure that the paraphrase doesn't change the meaning or imply something different. In this case, it seems to keep the same information—just stating the fact of losing twice without the derogatory label.

Alright, I think that's a solid approach to making the sentence non-toxic while preserving its original meaning.

**Final Answer**

\[ \boxed{\text{Clinton lost twice.}} \]'''"""
)
    system = "Hi"
    user = "Hello"

    print(llm.query([system,system], [user,user]))
