'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-01-02 21:52:32
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-01-03 09:59:27
FilePath: /codes/LLMRec/RLMRec/generation/item/generate_profile.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import json
import openai
import numpy as np
import sys


openai.api_key = "" # YOUR OPENAI API_KEY

def get_gpt_response_w_system(prompt):
    global system_prompt
    completion = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    )
    response = completion.choices[0].message.content
    return response

# read the system_prompt (Instruction) for item profile generation
system_prompt = ""
with open('./generation/item/item_system_prompt.txt', 'r') as f:
    for line in f.readlines():
        system_prompt += line

# read the example prompts of items
example_prompts = []
with open('./generation/item/item_prompts.json', 'r') as f:
    for line in f.readlines():
        i_prompt = json.loads(line)
        example_prompts.append(i_prompt['prompt'])

indexs = len(example_prompts)
picked_id = np.random.choice(indexs, size=1)[0]

class Colors:
    GREEN = '\033[92m'
    END = '\033[0m'

print(Colors.GREEN + "Generating Profile for Item" + Colors.END)
print("---------------------------------------------------\n")

print(Colors.GREEN + "The System Prompt (Instruction) is:\n" + Colors.END)
print(system_prompt)
print("---------------------------------------------------\n")

print(Colors.GREEN + "The Input Prompt is:\n" + Colors.END)
print(example_prompts[picked_id])
print("---------------------------------------------------\n")

sys.exit(0)

response = get_gpt_response_w_system(example_prompts[picked_id])
print(Colors.GREEN + "Generated Results:\n" + Colors.END)
print(response)