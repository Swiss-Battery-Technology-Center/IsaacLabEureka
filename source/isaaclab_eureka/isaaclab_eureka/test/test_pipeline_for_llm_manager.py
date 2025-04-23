from isaaclab_eureka.managers import LLMManager
from isaaclab_eureka.config import MANAGER_BASED_WEIGHT_TUNING_INITIAL_PROMPT, MANAGER_BASED_PPO_TUNING_INITIAL_PROMPT
import os
GPT_MODEL = "deepseek/deepseek-r1:free"
NUM_PROCESSES = 4
TEMPERATURE = 1
SYSTEM_PROMPT = MANAGER_BASED_WEIGHT_TUNING_INITIAL_PROMPT
class TestLLMManager:
    def __init__(self):
        self.lm = LLMManager(gpt_model=GPT_MODEL, 
                                      num_suggestions=NUM_PROCESSES,
                                      temperature=TEMPERATURE,
                                      system_prompt=SYSTEM_PROMPT,)
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

        

if __name__ == "__main__":
    test_llm_manager = TestLLMManager()
    test_llm_manager.test_conversation()