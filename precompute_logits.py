import json
import os
import pickle
import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import lzma
import logging
import pdb

# set up debugging log
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

DTYPE = torch.float32
DEVICE = torch.device('cuda:0')

def write_shard(outputs, shard_path, compress=False):
    try:
        if compress:
            with lzma.open(shard_path, "wb") as fp:
                pickle.dump(outputs, fp, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(shard_path, "wb") as fp:
                pickle.dump(outputs, fp, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"Wrote shard {shard_path}...")
    except Exception as e:
        logging.error(f"Failed to write shard {shard_path}: {e}", exc_info=True)

def main(*, prompts_json_path, output_dir, checkpoint_path, model_size="774M", tokenizer_path=None, output_shard_size=2500, compress=False):
    logging.info("Starting processing...")
    os.makedirs(output_dir, exist_ok=True)

    try:
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path if tokenizer_path else 'gpt2')
        model = GPT2LMHeadModel.from_pretrained(checkpoint_path).to(DEVICE).eval().to(DTYPE)
    except Exception as e:
        logging.critical(f"Failed to load model or tokenizer: {e}", exc_info=True)
        return

    try:
        with open(prompts_json_path, "r") as fp:
            prompts = json.load(fp)
    except Exception as e:
        logging.critical(f"Failed to load prompts: {e}", exc_info=True)
        return

    shard_count = 0
    outputs = {}
    try:
        for i, (key, prompt) in enumerate(sorted(prompts.items(), key=lambda t: t[0])):
            # pdb.set_trace()
            logging.debug(f"Processing prompt {key}: {prompt[:50]}...")  
            input_ids = tokenizer.encode(prompt, return_tensors='pt', max_length=1024, truncation=True).to(DEVICE)
    
            if input_ids.nelement() == 0:
                logging.warning(f"Skipping empty input for prompt {key}")
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
            logging.debug(f"Generated logits for prompt {key}")

        if outputs:
            shard_path = os.path.join(output_dir, f"gpt2_{model_size}_shard_{shard_count}.{'xz' if compress else 'pickle'}")
            write_shard(outputs, shard_path, compress)
            logging.info("Saved the final shard.")
    except Exception as e:
        logging.error(f"An error occurred during processing: {e}", exc_info=True)

    if shard_count == 0:
        logging.warning("No outputs were saved. Check if all prompts were skipped or input data was invalid.")

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
