import os
import pickle
import random
import time
from pathlib import Path
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Configuration
DTYPE = torch.float32
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_LEN = 2048

def load_model_and_tokenizer(model_name: str, cache_dir: str = None):
    """ Load GPT-2 model and tokenizer """
    model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir).to(DEVICE)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    return model, tokenizer

def read_shard(shard_path: str):
    """ Load a shard containing precomputed embeddings or other data """
    with open(shard_path, "rb") as f:
        shard = pickle.load(f)
    return shard

def process_embeddings(embeddings, model):
    """ Example function to process embeddings through GPT-2 """
    logits = model(embeddings)['logits']
    return logits

class PrecomputedShardLoader:
    """ Loads precomputed data shards for processing """
    def __init__(self, shard_dirs, dataset_filter_path=None):
        self.shard_dirs = shard_dirs
        self.dataset_filter = self.load_filter(dataset_filter_path) if dataset_filter_path else None
        self.shards = [os.listdir(shard_dir) for shard_dir in shard_dirs]

    def load_filter(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def load_shard(self, shard_id):
        shard_paths = [os.path.join(dir, shard_name) for dir, shard_name in zip(self.shard_dirs, self.shards[shard_id])]
        return [read_shard(path) for path in shard_paths]

    def __iter__(self):
        for shard_id in range(len(self.shards[0])):
            yield self.load_shard(shard_id)

def main(model_name, shard_dirs, output_dir, dataset_json_path):
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load shards
    shard_loader = PrecomputedShardLoader(shard_dirs)
    
    for shards in shard_loader:
        for shard in shards:
            embeddings = shard['embeddings'].to(DEVICE)
            logits = process_embeddings(embeddings, model)
            # Save or further process logits as needed

    print("Completed processing all shards.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 5:
        print("Usage: python script.py <model_name> <shard_dirs> <output_dir> <dataset_json_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
