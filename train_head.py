import os
import sys
import torch
import numpy as np
import random
from pathlib import Path
import pickle
import json
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import wandb

# Configuration
DTYPE = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_wandb(args):
    """ Initialize WandB for tracking experiments """
    wandb.login()
    wandb.init(project=args['wandb_project'], entity=args['wandb_entity'], name=args['wandb_run_name'])

def load_precomputed_embeddings(directory):
    """ Load precomputed embeddings from a specified directory """
    embeddings = {}
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if path.endswith('.pt'):
            key = filename.split('.')[0]
            embeddings[key] = torch.load(path)
    return embeddings

def main(
    precomputed_emb_dir: str,
    output_dir: str,
    model_checkpoint_path: str,
    batch_size: int = 64,
    lr: float = 1e-5,
    epochs: int = 10,
    use_wandb: bool = False,
    wandb_project: str = None,
    wandb_entity: str = None,
    wandb_run_name: str = None,
):
    """ Main function to process embeddings and train a simple model on top of GPT-2 embeddings """
    args = locals()

    # WandB setup
    if use_wandb:
        setup_wandb(args)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load GPT-2 model
    model = GPT2LMHeadModel.from_pretrained(model_checkpoint_path).to(DEVICE)
    model.eval()  # Make sure model is in evaluation mode

    # Load precomputed embeddings
    embeddings = load_precomputed_embeddings(precomputed_emb_dir)

    # Prepare data loader
    data = list(embeddings.values())
    random.shuffle(data)  # Shuffle data

    # Dummy training loop (replace with actual training logic)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(data), batch_size):
            batch = torch.stack(data[i:i + batch_size]).to(DEVICE)

            # Fake target generation for demonstration; replace with real targets
            target = torch.rand_like(batch)

            # Training step
            optimizer.zero_grad()
            outputs = model(batch)
            loss = loss_fn(outputs.logits, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if use_wandb:
                wandb.log({"loss": loss.item()})

        print(f"Epoch {epoch+1}: Average Loss: {total_loss / len(data)}")

    # Save the trained model
    model_path = os.path.join(output_dir, "trained_model.pt")
    torch.save(model.state_dict(), model_path)

    print("Training completed and model saved.")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--precomputed_emb_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_checkpoint_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use_wandb", action='store_true')
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_run_name", type=str)
    args = parser.parse_args()

    main(**vars(args))
