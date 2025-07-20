import openai
import os

# read api key from file
# api_key_path = os.path.expanduser("~/.keys/openai.api.key")
# api_key_path = os.path.expanduser("~/.keys/anthropic.api.key")
# api_key_path = os.path.expanduser("~/.keys/deepseek.api.key")
api_key_path = os.path.expanduser("~/.keys/fireworks.api.key")
with open(api_key_path, "r") as f:
    api_key = f.read().strip()
# print(f"🔍 [Chat API] API key: {api_key}")

client = openai.OpenAI(api_key=api_key, base_url="https://api.fireworks.ai/inference/v1")

response = client.chat.completions.create(    
    # model="gpt-4o",  # or "gpt-3.5-turbo"
    # model="claude-3-5-sonnet-20240620",
    # model="deepseek-chat",
    model="accounts/fireworks/models/deepseek-v3-0324",
    messages=[{"role": "user", "content": "What is machine learning?"}],
    logprobs=True,
    # stop=["<|im_end|>"],
    max_tokens=2048,
)

print("\nLogprobs:")
print(response.choices[0].logprobs)
print("\nFull response:")
print(response.choices[0].message.content)
