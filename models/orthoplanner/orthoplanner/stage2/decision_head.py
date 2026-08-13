"""④ Constrained Structured-Decision Head  (INTERNAL — never decoded to text).

Predicts the interdependent clinical attributes in dependency order (scope -> extraction ->
surgery -> duration) with hard clinical constraints, and turns the predictions into SOFT
conditioning embeddings that are prepended to the LLM. Also emits calibrated per-attribute
confidences used for hedging and for the per-attribute clinical evaluation (R3-W7).

The attributes are auxiliary supervision + conditioning; the model's only *emitted* output remains
the treatment-plan text.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .relational_fusion import DECISION_FACTORS

# attribute -> (factor name used as its anchor, num classes)  ; duration handled as regression
ATTRS = {"scope": ("scope", 3), "extraction": ("extraction", 2), "surgery": ("surgery", 2)}
FACTOR_ORDER = DECISION_FACTORS                                  # single source of truth
FIDX = {f: i for i, f in enumerate(FACTOR_ORDER)}

# v2: diagnostic descriptors derived from geometry (image-primary planning)
DESCRIPTORS = {"skeletal": 3, "dental": 3, "crowding": 2, "spacing": 2,
               "profile": 3, "protrusion": 2, "vertical": 3}


class DiagnosisHead(nn.Module):
    """Derive the clinical diagnosis (descriptors) from the geometry-reasoned factor tokens.
    Makes the system autonomous: it produces the diagnosis instead of receiving it as text."""

    def __init__(self, d_model=768):
        super().__init__()
        self.cls = nn.ModuleDict({k: nn.Linear(2 * d_model, n) for k, n in DESCRIPTORS.items()})
        self.emb = nn.ModuleDict({k: nn.Embedding(n, d_model) for k, n in DESCRIPTORS.items()})
        self.emb_ln = nn.LayerNorm(d_model)

    def forward(self, factor_summary):
        ctx = factor_summary.mean(dim=1)
        logits, soft = {}, []
        for k in DESCRIPTORS:
            # factor-anchored descriptor uses its query token; others (dental/spacing) use global ctx
            anchor = factor_summary[:, FIDX[k]] if k in FIDX else ctx
            logits[k] = self.cls[k](torch.cat([anchor, ctx], dim=-1))
            soft.append(torch.softmax(logits[k], dim=-1) @ self.emb[k].weight)
        return {"diag_logits": logits, "diag_emb": self.emb_ln(torch.stack(soft, dim=1))}  # (B,7,d)


class DecisionHead(nn.Module):
    def __init__(self, d_model=768, dur_mean=24.0, dur_std=12.0, use_diag_context=False):
        super().__init__()
        self.d = d_model
        self.dur_mean, self.dur_std = dur_mean, dur_std
        # v3.1: condition the decision on the rule-derived DIAGNOSIS so it is causally upstream of the
        # plan (diagnosis -> decision -> plan); makes the rule layer controllable/faithful.
        self.use_diag_context = use_diag_context
        in_dim = (3 if use_diag_context else 2) * d_model           # [factor ; ctx ; (diag_ctx)]
        # classifiers conditioned on [factor token ; global context ; (diagnosis context)]
        self.cls = nn.ModuleDict({
            a: nn.Linear(in_dim, n) for a, (_, n) in ATTRS.items()})
        self.conf = nn.ModuleDict({
            a: nn.Linear(in_dim, 1) for a in ATTRS})
        self.temp = nn.ParameterDict({
            a: nn.Parameter(torch.ones(1)) for a in ATTRS})        # calibration temperature
        self.dur = nn.Sequential(nn.Linear(in_dim, d_model), nn.GELU(), nn.Linear(d_model, 1))
        # class embedding tables -> soft decision embeddings for the LLM prefix
        self.cls_emb = nn.ModuleDict({a: nn.Embedding(n, d_model) for a, (_, n) in ATTRS.items()})
        self.dur_emb = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.emb_ln = nn.LayerNorm(d_model)

    def forward(self, factor_summary: torch.Tensor, apply_constraints: bool = True, diag_ctx=None):
        """factor_summary: (B, F, d) -> dict of logits, calibrated confidences, duration, and
        soft decision embeddings (B, n_attrs+1, d). diag_ctx (B, d): pooled diagnosis context that
        makes the decision a function of the (rule-derived) diagnosis when use_diag_context."""
        B = factor_summary.size(0)
        ctx = factor_summary.mean(dim=1)                            # (B, d) global context
        extra = [diag_ctx] if (self.use_diag_context and diag_ctx is not None) else []

        logits, conf, conf_logit, soft_emb = {}, {}, {}, []
        for a, (factor, _) in ATTRS.items():
            h = torch.cat([factor_summary[:, FIDX[factor]], ctx, *extra], dim=-1)
            logits[a] = self.cls[a](h)
            cl = self.conf[a](h).squeeze(-1)                       # raw logit (autocast-safe in BCE)
            conf_logit[a] = cl
            conf[a] = torch.sigmoid(cl)                            # probability for eval/reporting

        # --- hard clinical constraint: orthognathic surgery requires surgical scope ---
        if apply_constraints:
            scope_pred = logits["scope"].argmax(-1)                 # 2 == surgical (see labels.py)
            mask = (scope_pred != 2)
            sl = logits["surgery"].clone()
            sl[mask, 1] = sl[mask, 1] - 1e4                         # forbid surgery=yes when not surgical
            logits["surgery"] = sl

        # calibrated probabilities -> soft decision embeddings
        for a, (_, _) in ATTRS.items():
            probs = torch.softmax(logits[a] / self.temp[a].clamp_min(0.05), dim=-1)
            soft_emb.append(probs @ self.cls_emb[a].weight)         # (B, d)

        h_dur = torch.cat([factor_summary[:, FIDX["duration"]], ctx, *extra], dim=-1)
        dur_z = self.dur(h_dur).squeeze(-1)                         # standardized months
        dur_months = dur_z * self.dur_std + self.dur_mean
        soft_emb.append(self.dur_emb(dur_z.unsqueeze(-1)))         # duration embedding

        decision_emb = self.emb_ln(torch.stack(soft_emb, dim=1))   # (B, 4, d)
        return {"logits": logits, "conf": conf, "conf_logit": conf_logit, "dur_z": dur_z,
                "dur_months": dur_months, "decision_emb": decision_emb}
