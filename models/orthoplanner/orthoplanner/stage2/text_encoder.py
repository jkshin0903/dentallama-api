"""Frozen diagnosis text encoder (domain-adapted PubMedBERT from Stage 1b)."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class DiagnosisEncoder(nn.Module):
    def __init__(self, model_path: str, max_length: int = 256):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModel.from_pretrained(model_path)
        self.max_length = max_length
        self.hidden_size = self.model.config.hidden_size

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, texts, device):
        enc = self.tokenizer(list(texts), padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt").to(device)
        out = self.model(**enc)
        return out.last_hidden_state, enc["attention_mask"]
