from isaaclab_eureka.managers import LLMManager
from isaaclab_eureka.config import CONTEXT_CODE_SUMMARIZATION_PROMPT, SUCCESS_METRIC_SUMMARIZATION_PROMPT, PPO_SUMMARIZATION_PROMPT
from isaaclab_eureka.test import *
import os
GPT_MODEL = "deepseek/deepseek-r1:free"
NUM_PROCESSES = 3
TEMPERATURE = 1
class TestLLMManager:
    def __init__(self):
        self.lm = LLMManager(gpt_model=GPT_MODEL, 
                                      num_suggestions=1,
                                      temperature=TEMPERATURE,)
    def test_conversation(self):
        self.lm.clear_prompts()
        self.lm.append_user_prompt("I'll tell you where my friends are from. Later I'll ask you who is from where. You should remember the information I tell you. If I ask about a stranger, you should say I didn't tell you about them.")
        response_dict = self.lm.call_llm()
        self.lm.append_assistant_prompt(response_dict["raw_outputs"][0])
        self.lm.append_user_prompt("Andres is from Korea.")
        response_dict = self.lm.call_llm()
        self.lm.append_assistant_prompt(response_dict["raw_outputs"][0])
        self.lm.append_user_prompt("Boshi is from China.")
        response_dict = self.lm.call_llm()
        self.lm.append_assistant_prompt(response_dict["raw_outputs"][0])
        self.lm.append_user_prompt("Who is from Japan?")
        response_dict = self.lm.call_llm()
        self.lm.append_assistant_prompt(response_dict["raw_outputs"][0])

        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(SCRIPT_DIR, "test_outputs")
        os.makedirs(output_dir, exist_ok=True)
        # Save env context file inside it
        with open(os.path.join(output_dir, "llm_conversation.txt"), "w") as f:
            for chat in self.lm._prompts:
                f.write(f"{chat['role']}: {chat['content']}\n")
        print("[✅] Saved llm conversation to test_outputs/llm_conversation.txt")

    def test_summarization(self):
        self.lm.clear_prompts()
        self.lm.append_user_prompt(CONTEXT_CODE_SUMMARIZATION_PROMPT + CONTEXT_CODE)
        response_dict = self.lm.call_llm_for_summary()
        context_code_summary = response_dict["raw_output"]
        self.lm.clear_prompts()

        self.lm.append_user_prompt(SUCCESS_METRIC_SUMMARIZATION_PROMPT + SUCCESS_METRIC_CODE)
        response_dict = self.lm.call_llm_for_summary()
        success_metric_code_summary = response_dict["raw_output"]
        self.lm.clear_prompts()
        
        self.lm.append_user_prompt(PPO_SUMMARIZATION_PROMPT + PPO_CODE)
        response_dict = self.lm.call_llm_for_summary()
        ppo_code_summary = response_dict["raw_output"]
        self.lm.clear_prompts()

        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(SCRIPT_DIR, "test_outputs")
        os.makedirs(output_dir, exist_ok=True)
        # Save env context file inside it
        with open(os.path.join(output_dir, "summarization.txt"), "w") as f:
            f.write(f"context code summary:\n {context_code_summary}\n\n")
            f.write(f"success metric code summary:\n {success_metric_code_summary}\n\n")
            f.write(f"ppo code summary:\n {ppo_code_summary}\n\n")
        print("[✅] Saved llm conversation to test_outputs/llm_conversation.txt")
        print("[✅] Summarization test passed.")
        

if __name__ == "__main__":
    test_llm_manager = TestLLMManager()
    # test_llm_manager.test_conversation()
    test_llm_manager.test_summarization()