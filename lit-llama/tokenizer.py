from transformers import GPT2Tokenizer
import torch


class Tokenizer:
    """Tokenizer for GPT-2 using Hugging Face's GPT2Tokenizer."""

    def __init__(self, model_name: str) -> None:
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    def encode(self, string: str, add_special_tokens: bool = True) -> torch.Tensor:
        """Encode a string into input IDs."""
        return torch.tensor(self.tokenizer.encode(string, add_special_tokens=add_special_tokens))

    def decode(self, tokens: torch.Tensor) -> str:
        """Decode input IDs back to a string."""
        return self.tokenizer.decode(tokens.tolist())