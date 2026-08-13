"""Clinical Rule Layer v2 — multi-measure stratification with landmark-derivable auxiliaries.

Extends v1 (ANB-only skeletal/profile) with soft gates on Wits + APDI for skeletal capture,
and an ANB / E-line blend for profile. Designed from cert MNLogit ablation
(``scripts/analyze_rule_layer_v2_features.py``).

Expects measurements in ``RULE_LAYER_V2_KEYS`` order (29 scalars): core 13 from the bridge
plus landmark-extended keys (Wits, A–N perp, …). When only core-13 is available, missing
extended columns are treated as neutral (gate = 1).

Interface matches ``ClinicalRuleLayer`` for drop-in use in ``GUSRReasoner``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .decision_head import DESCRIPTORS
from .rule_layer import LEARNED_DESCRIPTORS, OrdinalRule
from ..utils.cephalometry import RULE_LAYER_V2_KEYS, RULE_V2_MIDX

# v1 specs for descriptors unchanged from single-measure rules
_RULE_V1 = {
    "vertical":   ("FMA",        [22.0, 28.0], [2, 0, 1]),
    "protrusion": ("Eline to L lip", [1.0],    [0, 1]),
}

_SKELETAL_SPEC = ("ANB", [1.0, 3.5], [2, 0, 1])
_WITS_INIT = -2.0
_APDI_INIT = 81.0

# Profile ANB leg uses cert-optimal cuts (≠ skeletal); E-line carries most signal.
_PROFILE_ANB = ("ANB", [-0.5, 3.0], [2, 0, 1])
_PROFILE_ELINE = ("Eline to L lip", [-2.0, 2.0], [2, 0, 1])
# Profile multi-measure evidence (validated offline: blend+ev acc 0.694 ≈ ceiling 0.699).
_PROFILE_EV_KEYS = ["APDI", "Md1 to NB (deg)", "SNA", "FMA"]
_PROFILE_EV_CENTERS = [81.0, 25.0, 82.0, 26.0]

# Dental Overjet leg + evidence (validated offline: OJ+ev acc 0.700 ≈ ceiling 0.706,
# vs Overjet-only 0.638). Evidence: high ANB / forward incisors → II; low → III.
_DENTAL_SPEC = ("Overjet", [2.5, 5.0], [2, 0, 1])
_DENTAL_EV_KEYS = ["ANB", "Md1 to NB (mm)", "ODI"]
_DENTAL_EV_CENTERS_II = [4.0, 0.0, 4.0]
_DENTAL_EV_CENTERS_III = [0.0, 0.0, 82.0]


class GatedSkeletalRule(nn.Module):
    """ANB ordinal + logit evidence from Wits/APDI (neutral at init → identical to v1)."""

    def __init__(self):
        super().__init__()
        _, cuts, perm = _SKELETAL_SPEC
        self.anb = OrdinalRule(3, cuts, perm, sharp_init=12.0)
        self.wits_cut = nn.Parameter(torch.tensor(_WITS_INIT))
        self.apdi_cut = nn.Parameter(torch.tensor(_APDI_INIT))
        self.alpha_wits = nn.Parameter(torch.zeros(1))
        self.alpha_apdi = nn.Parameter(torch.zeros(1))

    def forward(self, meas: torch.Tensor) -> torch.Tensor:
        anb = meas[:, RULE_V2_MIDX["ANB"]]
        log_p = torch.log(self.anb(anb).clamp_min(1e-6))

        wits_idx = RULE_V2_MIDX.get("Wits Appraisal")
        apdi_idx = RULE_V2_MIDX.get("APDI")
        if meas.size(-1) > wits_idx:
            wits = meas[:, wits_idx]
            apdi = meas[:, apdi_idx]
            # low Wits / low APDI → evidence for Class III (label idx 2)
            log_p[:, 2] = log_p[:, 2] + self.alpha_wits * (self.wits_cut - wits)
            log_p[:, 2] = log_p[:, 2] + self.alpha_apdi * (self.apdi_cut - apdi)

        return torch.softmax(log_p, dim=-1)


class GatedDentalRule(nn.Module):
    """Overjet ordinal + logit evidence from ANB / Md1-NB / ODI (neutral at init → v1)."""

    def __init__(self):
        super().__init__()
        _, cuts, perm = _DENTAL_SPEC
        self.overjet = OrdinalRule(3, cuts, perm, sharp_init=12.0)
        self.ev_keys = list(_DENTAL_EV_KEYS)
        self.alpha_II = nn.Parameter(torch.zeros(len(self.ev_keys)))
        self.alpha_III = nn.Parameter(torch.zeros(len(self.ev_keys)))
        self.register_buffer("center_II", torch.tensor(_DENTAL_EV_CENTERS_II))
        self.register_buffer("center_III", torch.tensor(_DENTAL_EV_CENTERS_III))

    def forward(self, meas: torch.Tensor) -> torch.Tensor:
        log_p = torch.log(self.overjet(meas[:, RULE_V2_MIDX["Overjet"]]).clamp_min(1e-6)).clone()
        idx = [RULE_V2_MIDX.get(k) for k in self.ev_keys]
        if all(i is not None and meas.size(-1) > i for i in idx):
            ev = torch.stack([meas[:, i] for i in idx], dim=-1)
            # high ANB / forward incisors → Class II (idx 1); low → Class III (idx 2)
            log_p[:, 1] = log_p[:, 1] + (self.alpha_II * (ev - self.center_II)).sum(-1)
            log_p[:, 2] = log_p[:, 2] + (self.alpha_III * (self.center_III - ev)).sum(-1)
        return torch.softmax(log_p, dim=-1)


class BlendedProfileRule(nn.Module):
    """ANB/E-line blend + logit evidence from APDI / Md1-NB / SNA / FMA (neutral at init)."""

    def __init__(self):
        super().__init__()
        _, anb_cuts, anb_perm = _PROFILE_ANB
        _, el_cuts, el_perm = _PROFILE_ELINE
        self.anb = OrdinalRule(3, anb_cuts, anb_perm, sharp_init=12.0)
        self.eline = OrdinalRule(3, el_cuts, el_perm, sharp_init=12.0)
        self.logit_w_eline = nn.Parameter(torch.tensor(-3.0))  # sigmoid≈0.05 → ANB-dominant at init
        self.ev_keys = list(_PROFILE_EV_KEYS)
        self.beta_convex = nn.Parameter(torch.zeros(len(self.ev_keys)))
        self.beta_concave = nn.Parameter(torch.zeros(len(self.ev_keys)))
        self.register_buffer("center", torch.tensor(_PROFILE_EV_CENTERS))

    def forward(self, meas: torch.Tensor) -> torch.Tensor:
        p_anb = self.anb(meas[:, RULE_V2_MIDX["ANB"]])
        p_el = self.eline(meas[:, RULE_V2_MIDX["Eline to L lip"]])
        w = torch.sigmoid(self.logit_w_eline)
        p = (1 - w) * p_anb + w * p_el
        log_p = torch.log(p.clamp_min(1e-6)).clone()
        idx = [RULE_V2_MIDX.get(k) for k in self.ev_keys]
        if all(i is not None and meas.size(-1) > i for i in idx):
            ev = torch.stack([meas[:, i] for i in idx], dim=-1)
            # low APDI/SNA etc → convex (idx 1); high → concave (idx 2)
            log_p[:, 1] = log_p[:, 1] + (self.beta_convex * (self.center - ev)).sum(-1)
            log_p[:, 2] = log_p[:, 2] + (self.beta_concave * (ev - self.center)).sum(-1)
        return torch.softmax(log_p, dim=-1)


class ClinicalRuleLayerV2(nn.Module):
    """Neuro-symbolic v2: multi-measure skeletal + blended profile."""

    def __init__(self, d_model: int = 768):
        super().__init__()
        self.skeletal = GatedSkeletalRule()
        self.profile = BlendedProfileRule()
        self.dental = GatedDentalRule()
        self.rules = nn.ModuleDict({
            k: OrdinalRule(DESCRIPTORS[k], cuts, perm, sharp_init=12.0)
            for k, (_, cuts, perm) in _RULE_V1.items()
        })
        self.meas_idx = {k: RULE_V2_MIDX[m] for k, (m, _, _) in _RULE_V1.items()}
        self.learned = nn.ModuleDict({
            k: nn.Linear(d_model, DESCRIPTORS[k]) for k in LEARNED_DESCRIPTORS})
        self.emb = nn.ModuleDict({k: nn.Embedding(n, d_model) for k, n in DESCRIPTORS.items()})
        self.emb_ln = nn.LayerNorm(d_model)

    def _rule_probs(self, k: str, meas: torch.Tensor) -> torch.Tensor:
        if k == "skeletal":
            return self.skeletal(meas)
        if k == "profile":
            return self.profile(meas)
        if k == "dental":
            return self.dental(meas)
        return self.rules[k](meas[:, self.meas_idx[k]])

    def forward(self, measurements: torch.Tensor, factor_summary: torch.Tensor, override=None) -> dict:
        ctx = factor_summary.mean(dim=1)
        logits, probs, soft, acts = {}, {}, [], {}
        for k in DESCRIPTORS:
            if k in ("skeletal", "profile", "dental") or k in self.rules:
                p = self._rule_probs(k, measurements)
                acts[k] = p.detach()
            else:
                p = torch.softmax(self.learned[k](ctx), dim=-1)
            if override and k in override:
                spec = override[k]
                if spec == "ablate":
                    p = torch.full_like(p, 1.0 / p.size(-1))
                else:
                    p = torch.zeros_like(p)
                    p[:, int(spec)] = 1.0
            probs[k] = p
            logits[k] = (p.clamp_min(1e-6)).log()
            soft.append(p @ self.emb[k].weight)
        return {
            "diag_logits": logits,
            "diag_probs": probs,
            "diag_emb": self.emb_ln(torch.stack(soft, dim=1)),
            "rule_acts": acts,
        }

    def threshold_report(self) -> dict:
        out = {
            "skeletal": {
                "measurement": "ANB + logit(Wits, APDI)",
                "anb_learned": [round(float(v), 2) for v in self.skeletal.anb.thresholds().tolist()],
                "anb_textbook": [round(float(v), 2) for v in self.skeletal.anb.init_cuts.tolist()],
                "wits_cut": round(float(self.skeletal.wits_cut), 2),
                "apdi_cut": round(float(self.skeletal.apdi_cut), 2),
                "alpha_wits": round(float(self.skeletal.alpha_wits), 3),
                "alpha_apdi": round(float(self.skeletal.alpha_apdi), 3),
            },
            "profile": {
                "measurement": "blend(ANB, Eline to L lip) + logit(APDI, Md1-NB, SNA, FMA)",
                "w_eline": round(float(torch.sigmoid(self.profile.logit_w_eline)), 3),
                "anb_learned": [round(float(v), 2) for v in self.profile.anb.thresholds().tolist()],
                "eline_learned": [round(float(v), 2) for v in self.profile.eline.thresholds().tolist()],
                "beta_convex": [round(float(v), 3) for v in self.profile.beta_convex.tolist()],
                "beta_concave": [round(float(v), 3) for v in self.profile.beta_concave.tolist()],
            },
            "dental": {
                "measurement": "Overjet + logit(ANB, Md1-NB, ODI)",
                "overjet_learned": [round(float(v), 2) for v in self.dental.overjet.thresholds().tolist()],
                "overjet_textbook": [round(float(v), 2) for v in self.dental.overjet.init_cuts.tolist()],
                "alpha_II": [round(float(v), 3) for v in self.dental.alpha_II.tolist()],
                "alpha_III": [round(float(v), 3) for v in self.dental.alpha_III.tolist()],
            },
        }
        for k, rule in self.rules.items():
            m = _RULE_V1[k][0]
            out[k] = {
                "measurement": m,
                "learned": [round(float(v), 2) for v in rule.thresholds().tolist()],
                "textbook": [round(float(v), 2) for v in rule.init_cuts.tolist()],
            }
        return out

    def prior_loss(self) -> torch.Tensor:
        dev = next(self.parameters()).device
        loss = torch.zeros((), device=dev)
        rules = [self.skeletal.anb, self.profile.anb, self.profile.eline,
                 self.dental.overjet, *self.rules.values()]
        for rule in rules:
            loss = loss + ((rule.thresholds() - rule.init_cuts.to(dev)) ** 2).mean()
        return loss / len(rules)
