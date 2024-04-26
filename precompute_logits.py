import json
import os
import pickle
import lzma
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Optional

# Set device, data type and other configurations
DTYPE = torch.bfloat16
DEVICE = torch.device('cuda:0')

def write_shard(outputs, shard_path, compress=False):
    if compress:
        with lzma.open(shard_path, "wb") as fp:
            pickle.dump(outputs, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(shard_path, "wb") as fp:
            pickle.dump(outputs, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote shard {shard_path}...")

def main(
    *,
    prompts_json_path: str,
    output_dir: str,
    checkpoint_path: str, 
    model_size: str = "774M",
    tokenizer_path: Optional[str] = None,
    output_shard_size: int = 2500,
    compress: bool = False,
) -> None:
    """Generates text samples based on a pre-trained GPT-2 model and tokenizer.

    Args:
        prompts_json_path: A JSON file containing data points
        output_dir: Where to save output pickle files
        checkpoint_path: The checkpoint path to load.
        tokenizer_path: The tokenizer path to load.
        output_shard_size: Number of outputs per output shard
        compress: Whether to compress outputs
    """
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path if tokenizer_path else 'gpt2')
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path).to(DEVICE).eval()
    model.to(DTYPE)  # Convert model to specified data type

    # Load the prompts
    with open(prompts_json_path, "r") as fp:
        data = json.load(fp)

    shard_count = 0
    outputs = {}
    for i, item in enumerate(data):
        prompt = f"Series {item['Series']} at X={item['X']} results in Y={item['Y']}"
        if i % output_shard_size == 0 and i != 0:
            shard_path = os.path.join(output_dir, f"gpt2_{model_size}_shard_{shard_count}.{'xz' if compress else 'pickle'}")
            write_shard(outputs, shard_path, compress)
            shard_count += 1
            outputs = {}

        # Tokenize the prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)

        # Generate output using the model
        with torch.no_grad():
            outputs_dict = model(input_ids, labels=input_ids)
            logits = outputs_dict.logits

        logits = logits.squeeze(0).cpu()
        outputs[i] = logits  # Use index as key to keep track of order

    # Save the last shard
    if outputs:
        shard_path = os.path.join(output_dir, f"gpt2_{model_size}_shard_{shard_count}.{'xz' if compress else 'pickle'}")
        write_shard(outputs, shard_path, compress)

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
