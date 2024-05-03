import logging
import os
import pickle
import sys
import time
from typing import Optional, Iterator, Tuple, Sequence

import torch
from torch import nn
from transformers import GPT2Model, GPT2Tokenizer


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


class PrecomputedShardLoader:
    def __init__(self, shard_dirs: Sequence[str], use_shard_cache: bool = False):
        self.shard_dirs = shard_dirs
        self.use_shard_cache = use_shard_cache
        self.shard_cache = {}
    
    def load_shard(self, shard_path: str):
        if shard_path in self.shard_cache:
            return self.shard_cache[shard_path]
        
        with open(shard_path, "rb") as file:
            shard = pickle.load(file)
        
        if self.use_shard_cache:
            self.shard_cache[shard_path] = shard
        
        return shard
    
    def __iter__(self):
        for shard_dir in self.shard_dirs:
            for shard_file in os.listdir(shard_dir):
                shard_path = os.path.join(shard_dir, shard_file)
                yield self.load_shard(shard_path)


def load_gpt2_model(model_name: str = 'gpt2'):
    print("Loading GPT-2 model...", file=sys.stderr, end='')
    t0 = time.time()
    model = GPT2Model.from_pretrained(model_name).to(DEVICE)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    print(f"Model loaded in {time.time() - t0:.02f} seconds.", file=sys.stderr)
    return model, tokenizer


def discretize(values: torch.Tensor, no_bins: int, mi: float, ma: float):
    assert mi < ma and no_bins > 0
    boundaries = torch.linspace(mi, ma, no_bins + 1, device=values.device)
    boundaries[..., -1] = float('inf')
    boundaries = boundaries.view(*([1] * len(values.shape)), -1)
    values = values.unsqueeze(-1)
    bin_id = torch.logical_and(boundaries[..., :-1] <= values, boundaries[..., 1:] > values).to(torch.int64).argmax(dim=-1)
    return bin_id


# TODO
def load_embedding_layer(
    checkpoint_path: str, 
    dtype: torch.dtype, 
    device: str, 
    model_type: str,
    model_size: str,
    revision: int = -1,
):
    logging.info(f"Loading model at {checkpoint_path}... ")
    t = time.time()
    if(model_type == "llama"): 
        assert(os.path.isfile(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        embed_layer_weights = checkpoint["transformer.wte.weight"]
        vocab_size, emb_dim = embed_layer_weights.shape
        emb_layer = nn.Embedding(
            vocab_size, emb_dim,
        )
        with torch.no_grad():
            emb_layer.weight.data = embed_layer_weights.to(dtype)
            emb_layer.eval()
            emb_layer = emb_layer.to(device)

        del checkpoint
    elif(model_type == "llama_2"):
        raise NotImplementedError
    elif(model_type == "pythia"):
        raise NotImplementedError
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    logging.info(f"Time: {time.time() - t:.02f} seconds.")

    return emb_layer


# TODO
class DistancePredictionHeadWithLMHead(nn.Module):
    def __init__(self,
        lm_head: nn.Linear,
        no_bins: int,
        hidden_dim: int,
        no_hidden_layers: int,
        dropout: float,
        log_scale: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = lm_head.weight.shape[1]
        self.token_dim = lm_head.weight.shape[0]
        self.no_bins = no_bins
        self.hidden_dim = hidden_dim
        self.no_hidden_layers = no_hidden_layers
        self.dropout = dropout
        self.log_scale = log_scale

        if activation == "relu":
            activation_class = nn.ReLU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.layers = nn.ModuleList()

        has_bias = lm_head.bias is not None
        local_lm_head = nn.Linear(self.input_dim, self.token_dim, bias=has_bias)
        with torch.no_grad():
            local_lm_head.weight.copy_(lm_head.weight)
            if(has_bias):
                local_lm_head.bias.copy_(lm_head.bias)

        self.layers.append(local_lm_head)

        if(no_hidden_layers == 0):
            self.layers.append(nn.Linear(self.token_dim, no_bins))
        else:
            self.layers.append(nn.Linear(self.token_dim, hidden_dim))
            self.layers.append(nn.Dropout(dropout))
            self.layers.append(activation_class())
            for _ in range(no_hidden_layers - 1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(nn.Dropout(dropout))
                self.layers.append(activation_class())

            self.layers.append(nn.Linear(hidden_dim, no_bins))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class DistancePredictionHead(nn.Module):
    def __init__(self,
        input_dim: int,
        no_bins: int,
        hidden_dim: int,
        no_hidden_layers: int,
        dropout: float,
        log_scale: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.no_bins = no_bins
        self.hidden_dim = hidden_dim
        self.no_hidden_layers = no_hidden_layers
        self.dropout = dropout
        self.log_scale = log_scale

        if activation == "relu":
            activation_class = nn.ReLU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.layers = nn.ModuleList()

        if(no_hidden_layers == 0):
            self.layers.append(nn.Linear(input_dim, no_bins))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.layers.append(nn.Dropout(dropout))
            self.layers.append(activation_class())
            for _ in range(no_hidden_layers - 1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(nn.Dropout(dropout))
                self.layers.append(activation_class())

            self.layers.append(nn.Linear(hidden_dim, no_bins))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


if __name__ == "__main__":
    model, tokenizer = load_gpt2_model()
    shard_path = "output/gpt2_small_2023 Bahrain Ministry of Interior Tennis Challenger.xz"
    loader = PrecomputedShardLoader([shard_path], use_shard_cache=True)
    
    for data in loader:
        print(data)
