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

def load_embeddings(file_path):
    """Loads embeddings from a single pickle file."""
    with open(file_path, "rb") as fp:
        embeddings = pickle.load(fp)
    # Ensure that embeddings are torch tensors and placed on the correct device
    if isinstance(embeddings, dict):
        for k, v in embeddings.items():
            if not isinstance(v, torch.Tensor):
                embeddings[k] = torch.tensor(v, dtype=torch.long, device=DEVICE)
            else:
                embeddings[k] = v.to(device=DEVICE)
    return embeddings

def main(
    *,
    precomputed_small_emb_path: str,
    precomputed_large_emb_path: str,
    output_dir: str,
    dataset_json_path: str,
    tokenizer_path: Optional[str] = None,
    entropy_min: float = 2.0,
    entropy_max: float = -1,
    entropy_delta: float = 0.1, 
    zero_entropy_threshold: float = 0.2,
    balanced_classes: bool = True,
    seed: int = 42,
):
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    entropy_max = float("inf") if entropy_max == -1 else entropy_max

    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path if tokenizer_path else 'gpt2')
    small_model = GPT2LMHeadModel.from_pretrained('gpt2').to(DEVICE).eval()
    large_model = GPT2LMHeadModel.from_pretrained('gpt2-large').to(DEVICE).eval()

    small_token_indices = load_embeddings(precomputed_small_emb_path)
    large_token_indices = load_embeddings(precomputed_large_emb_path)

    for key, small_indices in small_token_indices.items():
        if key not in large_token_indices:
            continue

        small_input_ids = small_indices.clone().detach().to(dtype=torch.long, device=DEVICE)
        large_input_ids = large_token_indices[key].clone().detach().to(dtype=torch.long, device=DEVICE)


        with torch.no_grad():
            small_outputs = small_model(input_ids=small_input_ids)
            large_outputs = large_model(input_ids=large_input_ids)

            small_probs = F.softmax(small_outputs.logits, dim=-1)
            large_probs = F.softmax(large_outputs.logits, dim=-1)

            small_entropy = -torch.sum(small_probs * torch.log(small_probs + 1e-10), dim=-1)
            large_entropy = -torch.sum(large_probs * torch.log(large_probs + 1e-10), dim=-1)

        print(f"Small entropy for {key}: {small_entropy.mean().item()}")
        print(f"Large entropy for {key}: {large_entropy.mean().item()}")

    print("Processing complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--precomputed_small_emb_path', required=True)
    parser.add_argument('--precomputed_large_emb_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--dataset_json_path', required=True)
    parser.add_argument('--tokenizer_path', default=None)
    parser.add_argument('--entropy_min', type=float, default=2.0)
    parser.add_argument('--entropy_max', type=float, default=-1)
    parser.add_argument('--entropy_delta', type=float, default=0.1)
    parser.add_argument('--zero_entropy_threshold', type=float, default=0.2)
    parser.add_argument('--balanced_classes', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    main(**vars(args))
