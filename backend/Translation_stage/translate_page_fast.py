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

class TranslatorFast():
    def __init__(self, model_type):
        self.model_type= model_type
        self.vlm_agent = [
            {"role": "system", "content": """
            You are an advanced Vision-Language AI assistant.
            You will be given a manga page consisting of a few panels of scenes, you will be given the corrdinates and size of each scene on the image and it is given in the order that you should read those scenes.
            Analyze manga page and provide intelligent infer on what's going.
            You can describe important objects, infer actions and relationships among characters, and provide insights based on visual input.
            Refer to the message history to see what happens before to help infer actuions cohesively with story line.
            """}, {'role': "user", "content": "nothing yet"}
        ]
        self.agent1_prompt = '''
    You are professional Japanese manga content analyzer.
    Briefly analyze and infer the scene. Then identify which verses are said by which character. Be mindful of the pronouns, refer to previous story for continuity.  
    Return result in json format: {"analysis": your analysis of the plot, "summary": one sentence summary}.
    But if it's a title page, translate to english and return {"book_name": , "author": }
    '''
        self.agent2_prompt = '''
    You are a professional Japanese to English translator.
    Translate the verses in each cluster, cut off in middle if necessary to keep output number of verses the same. Be accurate but structured.
    Return translation result in json format: {"1": , "2": ...}   
    If the original Japanese text is **simple (no slangs/idioms) and unambiguous**, type outside the json dictionary "TERMINATE" at the end. 
    '''
        self.agent3_prompt = '''
    You are professional Japanese to English localizer reviewing provided translation.
    Keep the **same number of verses** and structure as original Japanese, don't omit or change sound effects.
    Make only **necessary** changes—**do not over-interpret** or alter meaning. 
    Ensure smooth, natural English while keeping the **original intent and tone**.   
    Localize slang and idioms when needed, but stay faithful to the original text. 
    Return translation result in json format: {"1": , "2": ...}
    '''
        if model_type == 'gpt':
            self.client = OpenAI(
                api_key=os.environ['API_KEY'],  # This is the default and can be omitted
            )
        self.story_context = ["This is the first page."]
        self.first_page = True
    
    def clear_history(self):
        self.story_context = []
        return
    
    def vlm_infer(self, image_path, coordinates):
        image_base64 = encode(image_path)
        user_prompt = """
            Analyze this page. The coordinates (x, y, width and height in pixel) of each panel in reading order are {coordinates}.
            Return only one Summary with three sentences at maximum.
        """
        self.vlm_agent[1] = {"role": "user", "content": user_prompt, "images": [image_base64]}
        response = ollama.chat(model="llava:7b", messages=self.vlm_agent)
        return response['message']['content']
    
    def generate_translation(self, input_text, target_language, image_path, coordinates):
        if len(self.story_context) > 3:
            self.story_context.pop(0)
        context_message = self.vlm_infer(image_path, coordinates)
        print('finished vlm infer.')
        input_text = {f"{i+1}": input_text[i] for i in range(len(input_text))}
        task=f"""
        Original content: {input_text}
        Current page image inference: {context_message}
        Previous story: {self.story_context}
        """
        response = self.client.chat.completions.create(model='gpt-4o-mini', messages=[{'role':'system', 'content': self.agent1_prompt}, 
            {'role':'user', 'content':task}])
        agent1_out = response.choices[0].message.content
        print(agent1_out)
        history = [{'role': 'user', 'content': task}, response.choices[0].message]
        start = agent1_out.find("{")
        end = agent1_out.rfind("}") + 1  # Include the closing brace
        json_part = agent1_out[start:end]  # Extract the JSON substring
        parsed_json = json.loads(json_part)
        if self.first_page:
            self.first_page = False
            self.story_context = []
        else:
            self.story_context.append(parsed_json['summary'])
        if ("TITLE" in agent1_out or "book_name" in agent1_out):
            print(agent1_out)
            return [parsed_json['book_name'], parsed_json['author']], history
        
        response2 = self.client.chat.completions.create(model='gpt-4o-mini', messages = [{'role':'system', 'content':self.agent2_prompt},
            {'role':'user', 'content':task}, {'role':'assistant', 'content':agent1_out}])
        agent2_out = response2.choices[0].message.content
        print(agent2_out)
        history.append(response2.choices[0].message)
        if ('TERMINATE' in agent2_out):
            print(agent1_out)
            print(agent2_out)
            start = agent2_out.find("{")
            end = agent2_out.rfind("}") + 1  # Include the closing brace
            json_part = agent2_out[start:end]  # Extract the JSON substring
            parsed_json = json.loads(json_part)
            output = list(parsed_json.values())
            return output, history
        response3 = self.client.chat.completions.create(model='gpt-4o-mini', messages = [{'role':'system', 'content':self.agent3_prompt},
            {'role':'user', 'content':task}, {'role':'assistant', 'content':agent2_out}])
        agent3_out = response3.choices[0].message.content
        print(agent3_out)
        start = agent3_out.find("{")
        end = agent3_out.rfind("}") + 1  # Include the closing brace
        json_part = agent3_out[start:end]  # Extract the JSON substring
        parsed_json = json.loads(json_part)
        output = list(parsed_json.values())
        history.append(response3.choices[0].message)
        return output, history
    
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
    translator_system = TranslatorFast("gpt")
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

