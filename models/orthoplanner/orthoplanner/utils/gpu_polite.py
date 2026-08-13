"""Shared-GPU etiquette: cap PyTorch VRAM usage on busy nodes."""
from __future__ import annotations

import os

import torch


def apply_gpu_polite(cuda_id: int) -> None:
    """Honor ORTHO_GPU_MEM_FRACTION (e.g. 0.5) before model weights load."""
    frac_s = os.environ.get("ORTHO_GPU_MEM_FRACTION")
    if not frac_s:
        return
    frac = float(frac_s)
    if not (0.0 < frac < 1.0) or not torch.cuda.is_available():
        return
    torch.cuda.set_per_process_memory_fraction(frac, cuda_id)
    print(f"[gpu_polite] cuda:{cuda_id} memory_fraction={frac:.2f}")
