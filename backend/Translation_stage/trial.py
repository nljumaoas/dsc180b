import autogen
import os
os.environ["AUTOGEN_USE_DOCKER"] = "false"

config_list_llama3 = [
    {
        'model': "llama2:7b",
        'base_url': "http://localhost:11434/v1",
        'api_key': "ollama"
    }
]


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
Verify the translation for accuracy and adjust for cultural/contextual appropriateness.

Constraints:
- Be accurate and precise.
- Answer briefly, in few words.
- Think step by step.
"""

SYSTEM_PROMPT_3 = """
You are a experienced and professional senior editor. Refine the text for fluency and readability before finalizing.
If you are satisfied with the translation, reply TERMINATE. Otherwise, provide feedback.

Constraints:
- Be critical when evaluating the translation shown to you.
- Be accurate and precise on advise and feedback.
- Answer briefly, in few words.
- Think step by step.
"""

system_message = {'role': 'system', 'content': SYSTEM_PROMPT}
system_message2 = {'role': 'system', 'content': SYSTEM_PROMPT_2}
system_message3 = {'role': 'system', 'content': SYSTEM_PROMPT_3}

general_translator = autogen.agentchat.AssistantAgent(
  name="general_translator",
  system_message=SYSTEM_PROMPT,
  human_input_mode="NEVER",
  llm_config={
    "config_list": config_list_llama3,
    "timeout": 180,
    "temperature": 0.5},
)

context_checker = autogen.agentchat.AssistantAgent(
  name="context_checker",
  system_message=SYSTEM_PROMPT_2,
  human_input_mode="NEVER",
  llm_config={
    "config_list": config_list_llama3,
    "timeout": 240,
    "temperature": 0.5},
)

senior_editor = autogen.agentchat.AssistantAgent(
  name="general_translator",
  system_message=SYSTEM_PROMPT_3,
  human_input_mode="NEVER",
  llm_config={
    "config_list": config_list_llama3,
    "timeout": 240,
    "temperature": 0.3},
)

user = autogen.agentchat.UserProxyAgent(
  name="supervisor",
  human_input_mode="NEVER",
  max_consecutive_auto_reply=2,
  is_termination_msg=lambda x: x.get("content", "").strip().endswith("TERMINATE"),
)

groupchat = autogen.agentchat.GroupChat(agents=[user, general_translator, context_checker, senior_editor], messages=[], max_round=6, speaker_selection_method="round_robin")
manager = autogen.agentchat.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list_llama3})

task="""
Please translate to Chinese while preserving the nuance:

"It isn't just brave that she died for me; it is brave that she did it without announcing it, without hesitation, and without appearing to consider another option."

Please let contenct_checker and senior editor refine the answer.
"""

user.initiate_chat(manager,  message=task)
