import os
import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pickle

def main(prompts_file, model_name='gpt2', output_dir='./output', shard_size=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    
    with open(prompts_file, 'r') as file:
        prompts = json.load(file)
    
    outputs = []
    shard_count = 0

    for i, prompt in enumerate(prompts):
        try:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            outputs.append(model.generate(input_ids, max_length=512))
            
            if len(outputs) >= shard_size:
                with open(f'{output_dir}/shard_{shard_count}.pkl', 'wb') as f:
                    pickle.dump(outputs, f)
                shard_count += 1
                outputs = []
        except Exception as e:
            print(f"Error processing prompt {prompt}: {str(e)}")
    
    if outputs:
        with open(f'{output_dir}/shard_{shard_count}.pkl', 'wb') as f:
            pickle.dump(outputs, f)

if __name__ == "__main__":
    main(prompts_file='data/wiki_test.json')
