import json
import os
import pickle
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Optional
import lzma
import sys

# Set device, data type, and other configurations
DTYPE = torch.bfloat16
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def write_shard(outputs, shard_path, compress=False):
    try:
        if compress:
            with lzma.open(shard_path, "wb") as fp:
                pickle.dump(outputs, fp, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(shard_path, "wb") as fp:
                pickle.dump(outputs, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Wrote shard {shard_path}...")
    except Exception as e:
        print(f"Error writing shard {shard_path}: {e}", file=sys.stderr)

def main(*, prompts_json_path, output_dir, checkpoint_path, model_size="774M", tokenizer_path=None, output_shard_size=2500, compress=False):
    print("Starting processing...")
    try:
        os.makedirs(output_dir, exist_ok=True)
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path if tokenizer_path else 'gpt2')
        model = GPT2LMHeadModel.from_pretrained(checkpoint_path).to(DEVICE).eval().to(DTYPE)

        with open(prompts_json_path, "r") as fp:
            prompts = json.load(fp)

        shard_count = 0
        outputs = {}
        for i, (key, prompt) in enumerate(sorted(prompts.items(), key=lambda t: t[0])):
            print(f"Processing prompt {key}: {prompt[:50]}...")  # Print first 50 characters of the prompt
            input_ids = tokenizer.encode(prompt, return_tensors='pt', max_length=1024, truncation=True).to(DEVICE)
            
            if input_ids.nelement() == 0:
                print(f"Skipping empty input for prompt {key}")
                continue
            
            if input_ids.shape[1] == 0:  # Added check for empty after tokenization
                print(f"Skipping prompt {key} as it leads to empty input_ids after tokenization.")
                continue

            if i % output_shard_size == 0 and i != 0:
                shard_path = os.path.join(output_dir, f"gpt2_{model_size}_shard_{shard_count}.{'xz' if compress else 'pickle'}")
                write_shard(outputs, shard_path, compress)
                shard_count += 1
                outputs = {}

            with torch.no_grad():
                outputs_dict = model(input_ids, labels=input_ids)
                logits = outputs_dict.logits

            logits = logits.squeeze(0).cpu()
            outputs[key] = logits
            print(f"Generated logits for prompt {key}")

        if outputs:
            shard_path = os.path.join(output_dir, f"gpt2_{model_size}_shard_{shard_count}.{'xz' if compress else 'pickle'}")
            write_shard(outputs, shard_path, compress)
            print("Saved the final shard.")

        if shard_count == 0:
            print("No outputs were saved. Check if all prompts were skipped or input data was invalid.")

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run GPT-2 Text Generation')
    parser.add_argument('--prompts_json_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--tokenizer_path', default=None)
    parser.add_argument('--model_size', default='774M')
    parser.add_argument('--output_shard_size', type=int, default=2500)
    parser.add_argument('--compress', action='store_true')
    args = parser.parse_args()

    main(
        prompts_json_path=args.prompts_json_path,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        model_size=args.model_size,
        tokenizer_path=args.tokenizer_path,
        output_shard_size=args.output_shard_size,
        compress=args.compress,
    )