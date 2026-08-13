"""Stage-1 landmark datasets (trained from scratch).

Primary: Aariz / CEPHA29 (29 landmarks, senior+junior consensus). Self-contained loader that reads
the annotation JSONs directly in canonical CEPHA29 order — no dependency on the original messy
`AarizDataset`/`config` modules. Images are resized to a square with zero-padding (aspect ratio
preserved) and landmarks are tracked into normalized [0, 1] frame coordinates.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..utils.cephalometry import NUM_LANDMARKS

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def resize_with_padding(img: np.ndarray, landmarks: np.ndarray, target: int):
    """Resize (H,W) grayscale to target x target keeping aspect ratio, zero-pad, adjust landmarks."""
    h, w = img.shape[:2]
    scale = target / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_r = np.array(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    pad_top = (target - nh) // 2
    pad_left = (target - nw) // 2
    out = np.zeros((target, target), dtype=img_r.dtype)
    out[pad_top:pad_top + nh, pad_left:pad_left + nw] = img_r
    lm = landmarks.copy().astype(np.float32)
    lm[:, 0] = lm[:, 0] * scale + pad_left
    lm[:, 1] = lm[:, 1] * scale + pad_top
    return out, lm


def make_heatmaps(landmarks: np.ndarray, size: int, sigma: float = 2.0) -> np.ndarray:
    """Gaussian heatmaps (K, size, size) for landmarks given in `size`-frame coordinates."""
    k = landmarks.shape[0]
    hm = np.zeros((k, size, size), dtype=np.float32)
    ys, xs = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    for i in range(k):
        x, y = landmarks[i]
        if x < 0 or y < 0:
            continue
        hm[i] = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2.0 * sigma ** 2))
    return hm


class AarizLandmarkDataset(Dataset):
    def __init__(self, root: str, split: str, input_size: int = 768, heatmap_size: int = 192,
                 sigma: float = 2.0):
        self.root = Path(root)
        self.split = split.lower()
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma

        self.img_dir = self.root / self.split / "Cephalograms"
        ann_base = self.root / self.split / "Annotations" / "Cephalometric Landmarks"
        self.senior_dir = ann_base / "Senior Orthodontists"
        self.junior_dir = ann_base / "Junior Orthodontists"

        self.ids = []
        for f in sorted(os.listdir(self.img_dir)):
            if f.lower().endswith(_IMG_EXTS):
                stem = os.path.splitext(f)[0]
                if (self.senior_dir / f"{stem}.json").exists():
                    self.ids.append(f)
        if not self.ids:
            raise RuntimeError(f"No Aariz samples under {self.img_dir}")

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_landmarks(path: Path) -> np.ndarray:
        d = json.load(open(path))
        pts = [[lm["value"]["x"], lm["value"]["y"]] for lm in d["landmarks"]]
        return np.asarray(pts, dtype=np.float32)

    def __getitem__(self, idx):
        fname = self.ids[idx]
        stem = os.path.splitext(fname)[0]
        img = np.array(Image.open(self.img_dir / fname).convert("L"))

        senior = self._read_landmarks(self.senior_dir / f"{stem}.json")
        junior_path = self.junior_dir / f"{stem}.json"
        lm = np.ceil(0.5 * (senior + self._read_landmarks(junior_path))) \
            if junior_path.exists() else senior
        assert lm.shape == (NUM_LANDMARKS, 2), f"{stem}: expected 29 landmarks, got {lm.shape}"

        img, lm = resize_with_padding(img, lm, self.input_size)
        lm_hm = lm * (self.heatmap_size / self.input_size)

        img = img.astype(np.float32) / 255.0
        img = (img - img.mean()) / (img.std() + 1e-6)
        heatmaps = make_heatmaps(lm_hm, self.heatmap_size, self.sigma)

        return {
            "image": torch.from_numpy(img).unsqueeze(0).float(),          # (1, S, S)
            "heatmaps": torch.from_numpy(heatmaps).float(),               # (29, hm, hm)
            "landmarks_norm": torch.from_numpy(lm / self.input_size).float(),  # (29, 2) in [0,1]
        }
