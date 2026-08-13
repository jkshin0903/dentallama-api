"""Frozen Stage-1 cephalometric encoder.

Wraps the from-scratch-trained HRNet and turns its heatmaps into, per landmark:
  - normalized coordinates (x, y) in [0, 1]   (2D soft-argmax)
  - confidence  = peak softmax probability
  - entropy     = normalized heatmap entropy in [0, 1]   (deployment-motivated uncertainty)

This module is FROZEN in Stage 2. It outputs raw per-landmark geometry; projection/tokenization
happens downstream in the GUSR reasoner.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hrnet import HRNet


class CephaloEncoder(nn.Module):
    def __init__(self, num_landmarks: int = 29, tau: float = 0.07):
        super().__init__()
        self.hrnet = HRNet(num_landmarks=num_landmarks)
        self.tau = tau

    @torch.no_grad()
    def load_backbone(self, ckpt_path: str, map_location="cpu"):
        sd = torch.load(ckpt_path, map_location=map_location)
        sd = sd.get("model", sd.get("state_dict", sd)) if isinstance(sd, dict) else sd
        missing, unexpected = self.hrnet.load_state_dict(sd, strict=False)
        return missing, unexpected

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self

    def heatmaps(self, image: torch.Tensor) -> torch.Tensor:
        return self.hrnet(image)  # (B, K, H, W)

    def forward(self, image: torch.Tensor) -> dict:
        """image: (B, 1, S, S) -> dict(landmarks, conf, entropy) with grads only through HRNet
        (disabled when frozen). Coordinates normalized to [0, 1]."""
        hm = self.hrnet(image)                       # (B, K, H, W)
        B, K, H, W = hm.shape
        p = F.softmax(hm.reshape(B, K, H * W) / self.tau, dim=-1)   # (B, K, HW)

        xs = torch.linspace(0, 1, W, device=hm.device, dtype=hm.dtype)
        ys = torch.linspace(0, 1, H, device=hm.device, dtype=hm.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([gx, gy], dim=-1).reshape(H * W, 2)      # (HW, 2)

        coords = torch.einsum("bkn,nd->bkd", p, grid)              # (B, K, 2) in [0,1]
        conf = p.max(dim=-1, keepdim=True).values                 # (B, K, 1)
        ent = -(p * (p + 1e-12).log()).sum(dim=-1, keepdim=True)
        ent = ent / torch.log(torch.tensor(float(H * W), device=hm.device))  # normalize to [0,1]
        return {"landmarks": coords, "conf": conf, "entropy": ent}
