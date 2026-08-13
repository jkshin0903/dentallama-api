"""Clinical dataset (five-fold-final.json) for Stage-2 training/eval.

- Resolves the clinic-server image paths onto a local root by remapping the
  `pseudonymization_after/...` suffix (all 1357 ceph images resolve locally).
- Composes an English diagnostic summary from the record's structured descriptors + the
  cephalometric measurements (the "D + serialized M" input the paper describes), so the frozen
  English PubMedBERT receives clean clinical text.
- Target = the standardized English `treatment_plan` sentence (the only generation target).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..stage1.landmark_dataset import resize_with_padding
from ..utils.cephalometry import MEASUREMENT_KEYS
from .labels import labels_for

_SERVER_MARK = "pseudonymization_after/"


def remap_image(path: str, clinical_image_root: str):
    if not path:
        return None
    i = path.find(_SERVER_MARK)
    if i < 0:
        return path if os.path.exists(path) else None
    return os.path.join(clinical_image_root, path[i + len(_SERVER_MARK):])


def _measurement_str(rec) -> str:
    meas = rec.get("measurements_results") or {}
    return ", ".join(f"{k} {meas[k]:.1f}" for k in MEASUREMENT_KEYS if isinstance(meas.get(k), (int, float)))


def compose_diagnosis(rec) -> str:
    """Fallback English diagnosis composed from structured descriptors (used when no translation)."""
    parts = []
    field_phrases = [
        ("skeletal", "skeletal {}"), ("dental", "dental {}"), ("vertical", "{} vertical"),
        ("crowding", "{}"), ("spacing", "{}"), ("profile", "{} profile"),
        ("protrusion", "{} protrusion"),
    ]
    for key, tmpl in field_phrases:
        v = rec.get(key)
        if v:
            parts.append(tmpl.format(str(v).strip()))
    diag = "; ".join(parts) if parts else (rec.get("diagnosis") or "").strip()
    mstr = _measurement_str(rec)
    return (f"{diag}. Measurements: {mstr}." if mstr else f"{diag}.").strip()


def diagnosis_text(rec, translations: dict | None, include_measurements: bool = True) -> str:
    """Preferred input text: Gemini-translated English diagnosis (+ serialized measurements).
    Falls back to structured-field composition when no translation is available.
    Set include_measurements=False to force the geometry pathway to supply the measurements."""
    en = (translations or {}).get(rec.get("name"))
    if not en:
        return compose_diagnosis(rec)
    en = en.strip()
    if not include_measurements:
        return en
    mstr = _measurement_str(rec)
    return (f"{en} Measurements: {mstr}." if mstr else en)


class ClinicalDataset(Dataset):
    def __init__(self, json_path, split, eval_fold, clinical_image_root,
                 input_size=768, primary_only_eval=True, translations_path=None,
                 include_measurements=True):
        data = json.load(open(json_path))
        recs = data["records"]
        self.input_size = input_size
        self.root = clinical_image_root
        self.split = split
        self.include_measurements = include_measurements
        self.translations = (json.load(open(translations_path))
                             if translations_path and os.path.exists(translations_path) else {})

        def keep(r):
            if not r.get("treatment_plan"):
                return False
            if remap_image(r.get("ceph_image"), clinical_image_root) is None:
                return False
            if split == "train":
                return r.get("fold") != eval_fold
            return r.get("fold") == eval_fold and (r.get("is_primary", True) or not primary_only_eval)

        self.records = [r for r in recs if keep(r)]
        if not self.records:
            raise RuntimeError(f"No records for split={split} fold={eval_fold}")

    def __len__(self):
        return len(self.records)

    def _load_image(self, rec):
        path = remap_image(rec["ceph_image"], self.root)
        img = np.array(Image.open(path).convert("L"))
        dummy = np.zeros((1, 2), dtype=np.float32)
        img, _ = resize_with_padding(img, dummy, self.input_size)
        img = img.astype(np.float32) / 255.0
        img = (img - img.mean()) / (img.std() + 1e-6)
        return torch.from_numpy(img).unsqueeze(0).float()

    def __getitem__(self, idx):
        rec = self.records[idx]
        item = {
            "image": self._load_image(rec),
            "diagnosis": diagnosis_text(rec, self.translations, self.include_measurements),
            "plan": rec["treatment_plan"].strip(),
            "name": rec.get("name", str(idx)),
        }
        item.update(labels_for(rec))
        return item

    # ---- dataset statistics (computed on the train split) ----
    def duration_stats(self):
        vals = [r["dur_months"] for r in (labels_for(x) for x in self.records)
                if not math.isnan(r["dur_months"])]
        if not vals:
            return 24.0, 12.0
        return float(np.mean(vals)), float(np.std(vals) + 1e-6)

    def measurement_stats(self):
        M = np.array([labels_for(x)["gt_measurements"] for x in self.records], dtype=np.float64)
        mean = np.nanmean(M, axis=0)
        std = np.nanstd(M, axis=0) + 1e-6
        return (torch.tensor(np.nan_to_num(mean), dtype=torch.float32),
                torch.tensor(np.nan_to_num(std, nan=1.0), dtype=torch.float32))

    def measurement_v2_stats(self):
        """Train-split means of the 29-key rule-layer-v2 vector (fill for missing required cols)."""
        M = np.array([labels_for(x)["gt_measurements_v2"] for x in self.records], dtype=np.float64)
        mean = np.nanmean(M, axis=0)
        return torch.tensor(np.nan_to_num(mean), dtype=torch.float32)


def chat_prompt(tokenizer, system, user):
    """Render a chat prompt robustly across backbones.
    - Some templates (Mistral/BioMistral) reject a `system` role -> fold it into the user turn.
    - `enable_thinking=False` disables Qwen3's reasoning mode (ignored by templates without it),
      so the model emits the templated plan directly instead of <think>...</think>."""
    try:
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{system}\n\n{user}"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)


def make_collate(tokenizer, system_prompt, dur_mean, dur_std, max_len=512, meas_fill=None,
                 meas_v2_fill=None):
    # meas_fill: (M,) per-measurement fill values (train means) for MISSING measurements; required in
    # measurement-primary mode so the rule layer never sees NaN (missing -> treated as clinically normal).
    fill = None if meas_fill is None else torch.as_tensor(meas_fill, dtype=torch.float32).cpu()
    fill_v2 = None if meas_v2_fill is None else torch.as_tensor(meas_v2_fill, dtype=torch.float32).cpu()

    def collate(batch):
        images = torch.stack([b["image"] for b in batch])
        diags = [b["diagnosis"] for b in batch]
        names = [b["name"] for b in batch]

        input_ids, labels_list, attn_list = [], [], []
        for b in batch:
            prompt = chat_prompt(tokenizer, system_prompt, f"Diagnosis: {b['diagnosis']}\nPlan:")
            full = prompt + b["plan"] + tokenizer.eos_token
            ids_full = tokenizer(full, truncation=True, max_length=max_len)["input_ids"]
            ids_prompt = tokenizer(prompt, truncation=True, max_length=max_len)["input_ids"]
            lab = list(ids_full)
            for i in range(min(len(ids_prompt), len(lab))):
                lab[i] = -100                                        # mask the prompt
            input_ids.append(torch.tensor(ids_full))
            labels_list.append(torch.tensor(lab))
            attn_list.append(torch.ones(len(ids_full), dtype=torch.long))

        pad = tokenizer.pad_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad)
        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
        attn = torch.nn.utils.rnn.pad_sequence(attn_list, batch_first=True, padding_value=0)

        dur = torch.tensor([b["dur_months"] for b in batch], dtype=torch.float32)
        dur_z = (dur - dur_mean) / dur_std                          # NaN stays NaN -> masked in loss

        gt_meas = torch.tensor([b["gt_measurements"] for b in batch], dtype=torch.float32)  # raw, NaN where missing
        # filled copy for the rule layer (missing -> train mean ~ clinically normal); raw stays NaN for L_measure
        if fill is not None:
            gt_meas_filled = torch.where(torch.isnan(gt_meas), fill.unsqueeze(0).expand_as(gt_meas), gt_meas)
        else:
            gt_meas_filled = torch.nan_to_num(gt_meas)

        gt_meas_v2 = torch.tensor([b["gt_measurements_v2"] for b in batch], dtype=torch.float32)  # 29-key, NaN where missing
        if fill_v2 is not None:
            gt_meas_v2_filled = torch.where(torch.isnan(gt_meas_v2), fill_v2.unsqueeze(0).expand_as(gt_meas_v2), gt_meas_v2)
        else:
            gt_meas_v2_filled = torch.nan_to_num(gt_meas_v2)

        out = {
            "images": images, "diagnosis_texts": diags, "names": names,
            "input_ids": input_ids, "attention_mask": attn, "labels": labels,
            "y_scope": torch.tensor([b["y_scope"] for b in batch], dtype=torch.long),
            "y_extraction": torch.tensor([b["y_extraction"] for b in batch], dtype=torch.long),
            "y_surgery": torch.tensor([b["y_surgery"] for b in batch], dtype=torch.long),
            "y_ttype": torch.tensor([b.get("y_ttype", -100) for b in batch], dtype=torch.long),
            "dur_z": dur_z,
            "gt_measurements": gt_meas,                 # raw (NaN) -> L_measure supervision
            "gt_measurements_filled": gt_meas_filled,   # filled -> rule-layer input (measurement-primary)
            "gt_measurements_v2_filled": gt_meas_v2_filled,  # filled 29-key -> rule-layer-v2 input
            "metadata": torch.tensor([b["metadata"] for b in batch], dtype=torch.float32),  # v2 (B,2)
        }
        for k in batch[0].get("descriptors", {}):                          # v2 descriptor labels
            out[f"y_diag_{k}"] = torch.tensor([b["descriptors"][k] for b in batch], dtype=torch.long)
        return out

    return collate
