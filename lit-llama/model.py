from transformers import GPT2Config, GPT2Model
import torch
from torch import nn

# Configuration for GPT-2
gpt2_configs = {
    "small": GPT2Config(n_embd=768, n_layer=12, n_head=12, vocab_size=50257),
    "large": GPT2Config(n_embd=1280, n_layer=36, n_head=20, vocab_size=50257),  # Updated specs for the large model
}

# Model class using Hugging Face's GPT2Model
class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids):
        outputs = self.model(input_ids)
        last_hidden_states = outputs.last_hidden_state
        logits = self.lm_head(last_hidden_states)
        return logits

    @classmethod
    def from_pretrained(cls, model_name: str):
        config = gpt2_configs[model_name]
        model = cls(config)
        model.model.load_state_dict(torch.load(f"model_path/{model_name}/pytorch_model.bin"))
        return model
