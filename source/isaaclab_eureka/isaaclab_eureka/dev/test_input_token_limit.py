import openai
import os
_client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
    
)
_gpt_model = "deepseek/deepseek-r1:free"
current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, "texts", "full_context.txt")
output_path = "/workspace/isaaclab/_isaaclab_eureka/logs/eureka/SBTC-Unscrew-Franka-OSC-v0/reward_weight_tuning/randstart/2025-04-06_17-44-00/eureka_iterations.txt"
with open(output_path, "r", encoding="utf-8") as f:
    context = f.read()
_prompts = [{"role": "user", "content": context}]
responses = _client.chat.completions.create(
    model=_gpt_model,
    messages=_prompts,
)
print(responses) 
print("\n")