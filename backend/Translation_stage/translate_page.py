import autogen
import os
import ollama
from openai import OpenAI
import base64
import json
import ast
import regex as re
#from bert_score import score

os.environ["AUTOGEN_USE_DOCKER"] = "false"

# Load configuration from JSON file
with open("./Translation_stage/config.json", "r") as file:
    config = json.load(file)

# Access models and settings
vlm_model_name = config["models"]["gpt_based"]["vlm"]
agent_model = config["models"]["gpt_based"]["agent_model"]
agents = config["agents"]
llm_config = config["llm_config"]

def encode(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class Translator():
    def __init__(self, model_type):
        self.model_type= model_type
        self.vlm_agent = [
            {"role": "system", "content": """
            You are an advanced Vision-Language AI assistant.
            You will be given a manga page consisting of a few panels of scenes, you will be given the corrdinates and size of each scene on the image and it is given in the order that you should read those scenes.
            Analyze manga page and provide intelligent infer on what's going.
            You can describe important objects, infer actions and relationships among characters, and provide insights based on visual input.
            Refer to the message history to see what happens before to help infer actuions cohesively with story line.
            """}
        ]
        self.agent1 = self.create_agent(agents[0]['prompt'], agent_model, agents[0]['name'])
        self.agent2 = self.create_agent(agents[1]['prompt'], agent_model, agents[1]['name'])
        self.agent3 = self.create_agent(agents[2]['prompt'], agent_model, agents[2]['name'])
        if model_type == 'gpt':
            self.client = OpenAI(
                api_key=os.environ['API_KEY'],  # This is the default and can be omitted
            )
        self.context_history = []
    
    def clear_history(self):
        self.context_history = []
        return
    
    def vlm_infer(self, image_path, coordinates):
        image_base64 = encode(image_path)
        user_prompt = """
            Analyze this page. The coordinates (x, y, width and height in pixel) of each panel in reading order are {coordinates}.
        """
        self.vlm_agent.append({"role": "user", "content": user_prompt, "images": [image_base64]})
        response = ollama.chat(model="llava:7b", messages=self.vlm_agent)
        self.vlm_agent.append({"role": "assistant", "content": response['message']['content']})
        return response['message']['content']
    
    def create_agent(self, system_prompt, model_type, agent_name):
        if self.model_type == 'llama':
            config_list = [{
                'model': model_type,
                'base_url': "http://localhost:11434/v1",
                'api_key': "ollama"
            }]
        else:
            config_list = [{
                'model': model_type,
                'api_key': os.environ['API_KEY']
            }]

        system_message = {'role': 'system', 'content': system_prompt}

        agent = autogen.agentchat.AssistantAgent(
          name=agent_name,
          system_message=system_prompt,
          human_input_mode="NEVER",
          llm_config={
            "config_list": config_list,
            "timeout": 180,
            "temperature": 0.2},
        )
        return agent
    
    def generate_translation(self, text, target_language, image_path, coordinates):
        if len(self.context_history) > 3:
            self.context_history.pop(0)
        context_message = self.vlm_infer(image_path, coordinates)
        print('finished vlm infer.')
        user = autogen.agentchat.UserProxyAgent(
            name="supervisor",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
            is_termination_msg=lambda x: x.get("content", "").strip().find("TERMINATE") != -1,
        )

        config_list = [{
            'model': agent_model,
            'base_url': "http://localhost:11434/v1",
            'api_key': "ollama"
        }]

        groupchat = autogen.agentchat.GroupChat(agents=[user, self.agent1, self.agent2, self.agent3], messages=[], max_round=6, speaker_selection_method="round_robin")
        manager = autogen.agentchat.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})
        print('set up group chat.')
        task=f"""
        Please translate dialogues this manga page to {target_language} while preserving the nuance and tone emotion. 
        **Do not combine, omit, or split any verses**. The **number of verses in the translation must exactly match** the number of verses provided ({len(text)} verses).
        For verses with negative emotion, try best to translate:

        {text}

        Consider the visual context of this scene while translating: {context_message}.
        Refer to previous content of the story for coherency in translation: {self.context_history}.
        Return end translation result for verses in a list with format (double quotes around each verse) like this: result: ["How are you", "what's up"]
        """

        user.initiate_chat(manager,  message=task)
        final_translation = groupchat.messages[-1]['content']
        print('finished translation.')
        # print("final reply: ", final_translation)
        retry_trans=True
        num_tries = 0
        while retry_trans and num_tries < 3:
            try:
                translations_list = self.extract_trans_result(final_translation)
                retry_trans = False
                break
            except IndexError as e:
                print(e, ", retry with first output")
                num_tries += 1
                final_translation = groupchat.messages[2]['content']
        i=0
        while len(translations_list) != len(text) & i < 3:
            translations_list = self.validate_translation(text, translations_list)
            i+=1
        print(translations_list)
        self.context_history.append(translations_list)
        return translations_list, groupchat.messages
    
    def extract_trans_result(self, final_message):
        # Look for 'result: [...]' in the response
        list_str = final_message.split("result: ")[1]
        list_str = list_str.split("]")[0] + "]"
        try:
            translations_list = ast.literal_eval(list_str)
        except SyntaxError:
            fixed = re.sub(r"(?<=\W)'\s*(.*?)\s*'(?=\W)", r'"\1"', list_str)
            translations_list = ast.literal_eval(fixed)
        return translations_list
    
    def validate_translation(self, input_verses, output_verses):
        if len(input_verses) != len(output_verses):
            print("Mismatch detected! Re-prompting for correction...")
            # Modify the prompt to emphasize correction
            new_prompt = f'''
                The translation must have the **exact same number of verses** as the input.
                Input has {len(input_verses)} verses while your output has a mismatch with {len(output_verses)} verses.
                Your current output: {output_verses}
                Your input for reference: {input_verses}
                Please correct the output and return the result in the format like this: result: ["How are you", "what's up"]
            '''
            # Re-run the model with the new prompt
            if self.model_type == 'llama':
                corrected_output = ollama.chat(model='llama3.1:8b', messages=[{'role':'user', 'content':new_prompt}])
                reformated_output = self.extract_trans_result(corrected_output['message']['content'])
            else:
                corrected_output = self.client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": new_prompt}])
                reformated_output = self.extract_trans_result(corrected_output.choices[0].message.content)
            return reformated_output
        return output_verses
    '''
    def translation_eval_bert_score(self, generated_translation, reference_translation):
        # the score for each individual verse (P=how much is relevant, R = how much is captured, F1=balance between two)
        P, R, F1 = score(generated_translation, reference_translation, lang="en", rescale_with_baseline=True)
        # the overall score for the entire page
        average_f1 = sum(F1) / len(F1)
        average_P = sum(P) / len(P)
        average_R = sum(R) / len(R)
        return P, R, F1, average_P, average_R, average_f1'''
    
if __name__ == "__main__":
    with open("content.json", "r") as file:
        image_content = json.load(file)
    image_path = image_content["image_paths"]['ja']
    print('file read complete.')
    texts = []
    for d in image_content['text']:
        texts.append(d['text_ja'])
    panels_coordinates = image_content['frame']
    print('date prep complete.')
    target_language = "English"
    translator_system = Translator()
    print('created translator instance.')
    translation_result = translator_system.generate_translation(texts, target_language, image_path, panels_coordinates)
    print('obtained translation.')
    print(translation_result)

    result = {"image": image_path, "text": []}
    for i in range(len(image_content['text'])):
        updated = image_content['text'][i]
        updated['text_translated'] = translation_result[i]
        result['text'].append(updated)
    print('organized result.')
    with open('result.json', 'w') as json_file:
        json.dump(result, json_file, indent=4)

