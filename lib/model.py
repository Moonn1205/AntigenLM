import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model


class GPTForFluMultiTask(nn.Module):
    def __init__(self, config: GPT2Config, num_labels: int = 0):
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.use_classification = num_labels > 0
        if self.use_classification:
            self.classification_head = nn.Linear(config.n_embd, num_labels)

    def forward(self, input_ids=None, attention_mask=None):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = out.last_hidden_state
        logits = self.lm_head(hidden)
        cls_logits = None
        if self.use_classification:
            cls_logits = self.classification_head(hidden[:, -1, :])
        return {"logits": logits, "classification_logits": cls_logits}

    @property
    def device(self):
        return next(self.parameters()).device

