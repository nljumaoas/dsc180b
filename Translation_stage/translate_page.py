import autogen
import os
import ollama
import base64
import json
import re

os.environ["AUTOGEN_USE_DOCKER"] = "false"

# Load configuration from JSON file
with open("config.json", "r") as file:
    config = json.load(file)

# Access models and settings
vlm_model_name = config["models"]["vlm"]
agent_model = config["models"]["agent_model"]
openai_api = config["api_keys"]["openai"]
agents = config["agents"]
llm_config = config["llm_config"]

def encode(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class Translator():
    def __init__(self):
        self.vlm_agent = [
            {"role": "system", "content": """
            You are an advanced Vision-Language AI assistant named LLaVA-Agent.
            You will be given an image that contains a few panels of scenes in it, you will be given the corrdinates and size of each scene on the image and it is given in the order that you should read those scenes.
            Your task is to analyze images and provide intelligent, detailed responses.
            You can describe important objects, infer actions, and provide insights based on visual input.
            You can refer to the message history to see what happens before to help infer actuions cohesively with story line. Be concise, informative, and helpful.
            """}
        ]
        self.agent1 = self.create_agent(agents[0]['prompt'], agent_model, agents[0]['name'])
        self.agent2 = self.create_agent(agents[1]['prompt'], agent_model, agents[1]['name'])
        self.agent3 = self.create_agent(agents[2]['prompt'], agent_model, agents[2]['name'])
        self.context_history = []
    
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
        config_list = [{
            'model': model_type,
            'base_url': "http://localhost:11434/v1",
            'api_key': openai_api
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
        context_message = self.vlm_infer(image_path, coordinates)
        print('finished vlm infer.')
        user = autogen.agentchat.UserProxyAgent(
            name="supervisor",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
            is_termination_msg=lambda x: x.get("content", "").strip().endswith("TERMINATE"),
        )

        config_list = [{
            'model': agent_model,
            'base_url': "http://localhost:11434/v1",
            'api_key': openai_api
        }]

        groupchat = autogen.agentchat.GroupChat(agents=[user, self.agent1, self.agent2, self.agent3], messages=[], max_round=6, speaker_selection_method="round_robin")
        manager = autogen.agentchat.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})
        print('set up group chat.')
        task=f"""
        Please translate to {target_language} while preserving the nuance and tone emotion, for verses with negative emotion, try best to translate:

        {text}

        Consider the visual context of this scene while translating: {context_message}.
        Refer to previous content of the story for coherency in translation: {self.context_history}.
        """

        user.initiate_chat(manager,  message=task)
        final_translation = groupchat.messages[-1]['content']
        print('finished translation.')
        # Look for 'result: [...]' in the response
        match = re.search(r'result:\s*\[(.*?)\]', final_translation, re.DOTALL)
        translations_str = match.group(1)

        # Convert the string list to an actual Python list
        if translations_str.strip().startswith('["') or translations_str.strip().startswith("['"):
            translations_str = translations_str.strip()[1:-1]  # Remove outer quotes

        # Now split the cleaned string
        translations_list = re.findall(r'"([^"]+)"', translations_str)
        self.context_history.append(translations_list)
        return translations_list
    
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

SYSTEM_PROMPT = """
You are a knowledgeable and professional translator that translate content assigned to you based on the requirement.

Constraints:
- Be accurate and precise.
- Answer briefly, in few words.
- Be careful and make sure the translated text are sementically correct and free from gramatical errors in the language translate into.
- Think step by step.
"""

SYSTEM_PROMPT_2 = """
You are a context checker professional at the cultural context of different languages. 
Verify the translation for accuracy and nuance, adjusting for cultural/contextual appropriateness while also consider the visual context for current page and previous content of story for cohesiveness.

Constraints:
- Be accurate and precise.
- Answer briefly, in few words.
- Think step by step.
"""

SYSTEM_PROMPT_3 = """
You are a experienced and professional senior editor. Refine the text for fluency and readability in translated language before finalizing.
If you are satisfied with the translation, reply TERMINATE. Otherwise, provide feedback.

Constraints:
- Be critical when evaluating the translation shown to you.
- Be accurate and precise on advise and feedback.
- Answer briefly, in few words.
- Think step by step.
"""
