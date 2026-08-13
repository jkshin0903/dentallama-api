"""① Differentiable Cephalometric Measurement Bridge.

Turns frozen Stage-1 landmarks into clinically-named measurements (SNA, SNB, ANB, ...) in closed
form, calibrates them to clinical units with a learnable per-measurement affine, and emits
measurement TOKENS (one per measurement, tagged with a learned type embedding) for the fusion
module. The calibrated measurements are supervised against `measurements_results` in the dataset,
which is what gives the model *verifiable* geometric grounding (rebuts R2-W5, resolves R3-W4).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.cephalometry import NUM_MEASUREMENTS, compute_measurements


class MeasurementBridge(nn.Module):
    def __init__(self, d_model: int = 768):
        super().__init__()
        self.num_meas = NUM_MEASUREMENTS
        # per-measurement affine: calibrated = raw * scale + bias  (maps normalized geometry -> units)
        self.scale = nn.Parameter(torch.ones(NUM_MEASUREMENTS))
        self.bias = nn.Parameter(torch.zeros(NUM_MEASUREMENTS))
        # learned identity per measurement so tokens are distinguishable
        self.type_emb = nn.Parameter(torch.randn(NUM_MEASUREMENTS, d_model) * 0.02)
        # embed each (calibrated) scalar value into a token
        self.value_mlp = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, landmarks: torch.Tensor):
        """landmarks: (B, 29, 2) -> (measurements (B, M), tokens (B, M, d))."""
        raw = compute_measurements(landmarks)                  # (B, M)
        measurements = raw * self.scale + self.bias            # calibrated
        return measurements, self.tokenize(measurements)

    def tokenize(self, measurements: torch.Tensor) -> torch.Tensor:
        """Embed a measurement vector (B, M) in clinical units into tokens (B, M, d).
        Used in measurement-primary mode to tokenize the provided clinical measurements."""
        val_tok = self.value_mlp(measurements.unsqueeze(-1))   # (B, M, d)
        return self.norm(val_tok + self.type_emb.unsqueeze(0))
