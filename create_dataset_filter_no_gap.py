import json
import os
import pickle
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Optional

# Configuration
DTYPE = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_embeddings_from_dir(directory):
    embeddings = {}
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if os.path.isfile(path) and path.endswith('.pt'):
            key = os.path.splitext(os.path.basename(filename))[0]
            embeddings[key] = torch.load(path)
    return embeddings

def main(
    *,
    precomputed_small_emb_dir: str,
    precomputed_large_emb_dir: str,
    output_dir: str,
    dataset_json_path: str,
    small_checkpoint_path: str,
    large_checkpoint_path: str,
    tokenizer_path: Optional[str] = None,
    small_model_size: str = "small",
    large_model_size: str = "large",
    entropy_min: float = 2.0,
    entropy_max: float = -1,
    entropy_delta: float = 0.1, 
    zero_entropy_threshold: float = 0.2,
    balanced_classes: bool = True,
    shard_output: bool = False,
    save_zeros: bool = False,
    shard_size: int = 200, 
    seed: int = 42,
):
    """ Filter embeddings based on entropy calculations. """

    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    entropy_max = float("inf") if entropy_max == -1 else entropy_max

    # Load the GPT-2 models and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path if tokenizer_path else 'gpt2')
    small_model = GPT2LMHeadModel.from_pretrained(small_checkpoint_path).to(DEVICE).eval()
    large_model = GPT2LMHeadModel.from_pretrained(large_checkpoint_path).to(DEVICE).eval()

    # Load embeddings
    small_embeddings = load_embeddings_from_dir(precomputed_small_emb_dir)
    large_embeddings = load_embeddings_from_dir(precomputed_large_emb_dir)

    # Prepare data structures for results
    filt = {}
    by_label = {"0": {}, "1": {}}
    small_entropy_dict = {}
    large_entropy_dict = {}

    # Processing embeddings
    for key in small_embeddings:
        if key not in large_embeddings:
            continue

        small_emb = small_embeddings[key].to(DEVICE)
        large_emb = large_embeddings[key].to(DEVICE)

        # Calculate logits and softmax probabilities
        with torch.no_grad():
            small_logits = small_model(small_emb)['logits']
            large_logits = large_model(large_emb)['logits']
            small_probs = F.softmax(small_logits, dim=-1)
            large_probs = F.softmax(large_logits, dim=-1)

            # Calculate entropies
            small_entropy = -torch.sum(small_probs * torch.log(small_probs + 1e-10), dim=-1)
            large_entropy = -torch.sum(large_probs * torch.log(large_probs + 1e-10), dim=-1)

        # Entropy filtering
        is_small_entropy_in_range = (small_entropy >= entropy_min) & (small_entropy <= entropy_max)
        is_large_entropy_zero = large_entropy < zero_entropy_threshold
        is_large_entropy_near_small = (large_entropy >= (small_entropy - entropy_delta)) & (large_entropy <= (small_entropy + entropy_delta))

        high_e_low_a = is_small_entropy_in_range & is_large_entropy_zero
        low_e_high_a = is_small_entropy_in_range & is_large_entropy_near_small

        # Store results
        if high_e_low_a.any():
            by_label["0"][key] = high_e_low_a
        if low_e_high_a.any():
            by_label["1"][key] = low_e_high_a

        small_entropy_dict[key] = small_entropy.mean().item()
        large_entropy_dict[key] = large_entropy.mean().item()

    # Save filtered results and other outputs as needed
    # Example: Save filter dictionary
    filter_path = os.path.join(output_dir, "filter.pickle")
    with open(filter_path, "wb") as f:
        pickle.dump(filt, f)

    print("Filtering complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--precomputed_small_emb_dir', required=True)
    parser.add_argument('--precomputed_large_emb_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--dataset_json_path', required=True)
    parser.add_argument('--small_checkpoint_path', required=True)
    parser.add_argument('--large_checkpoint_path', required=True)
    parser.add_argument('--tokenizer_path', default=None)
    parser.add_argument('--small_model_size', default='small')
    parser.add_argument('--large_model_size', default='large')
    parser.add_argument('--entropy_min', type=float, default=2.0)
    parser.add_argument('--entropy_max', type=float, default=-1)
    parser.add_argument('--entropy_delta', type=float, default=0.1)
    parser.add_argument('--zero_entropy_threshold', type=float, default=0.2)
    parser.add_argument('--balanced_classes', type=bool, default=True)
    parser.add_argument('--shard_output', type=bool, default=False)
    parser.add_argument('--save_zeros', type=bool, default=False)
    parser.add_argument('--shard_size', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    main(**vars(args))
