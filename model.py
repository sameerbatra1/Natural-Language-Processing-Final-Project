from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import BertModel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    model_name: str = "bert-base-uncased"
    num_labels: int = 6
    dropout: float = 0.3
    freeze_bert_layers: int = 0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class JigsawBERTClassifier(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self._cfg = config
        self.bert = BertModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(p=config.dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.num_labels)

        if config.freeze_bert_layers > 0:
            self._freeze_layers(config.freeze_bert_layers)

    def _freeze_layers(self, n: int) -> None:
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:n]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_output))
