import torch
from pathlib import Path


def save_model_checkpoint(model, file_path):
    """Save a GPT-2 model checkpoint."""
    file_path = Path(file_path)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            torch.save(model.state_dict(), file_path)
        torch.distributed.barrier()
    else:
        torch.save(model.state_dict(), file_path)


def load_model_checkpoint(model, file_path):
    """Load a GPT-2 model checkpoint."""
    model.load_state_dict(torch.load(file_path))
    return model