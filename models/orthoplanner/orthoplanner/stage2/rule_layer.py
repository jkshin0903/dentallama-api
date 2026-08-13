"""Differentiable Clinical Rule Layer  (the v3 novelty — replaces the black-box DiagnosisHead).

Instead of deriving the diagnosis with a hidden MLP, this module computes each diagnostic
descriptor with an EXPLICIT, DIFFERENTIABLE clinical decision rule applied to the bridge-derived
cephalometric measurements (ANB, FMA, overjet, upper-incisor inclination, ...). Each rule is a soft
ordinal threshold whose cut-points are initialised to textbook clinical values but are LEARNABLE,
so the model can refine them from data while the diagnosis stays interpretable: the rule firing
strengths ARE the explanation, and the recovered thresholds-vs-textbook is itself a result.

Why this is the contribution (not BLIP-2's Q-Former, not bare image->plan):
  * the diagnosis is *computed by transparent rules*, auditable per case (which rule fired, how hard);
  * the cut-points are learnable -> the system can recover / refine clinical thresholds from data;
  * a rule can be ablated to test faithfulness (does the plan change when its rule is suppressed?).

Descriptors with a closed-form cephalometric criterion are rule-driven; the two without one in the
13-measurement panel (crowding / spacing — arch-length discrepancy is not measured here) fall back
to a small learned predicate on the geometry context, and we say so.

Output matches DiagnosisHead so it slots into the GUSR prefix unchanged:
    {"diag_logits": {descriptor: (B, n)}, "diag_emb": (B, 7, d)}  (+ interpretability extras)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .decision_head import DESCRIPTORS
from ..utils.cephalometry import MEASUREMENT_KEYS

_MIDX = {k: i for i, k in enumerate(MEASUREMENT_KEYS)}

# descriptor -> (measurement name, textbook cut-points [low..high], ordinal->label permutation)
#   classes are ordered by INCREASING measurement value; perm[ordinal_position] = dataset label index.
#   label indices (labels.py): skeletal/dental 0=I 1=II 2=III ; vertical 0=normal 1=high 2=low ;
#                              profile 0=straight 1=convex 2=concave ; protrusion 0=none 1=protrusion
RULE_SPEC = {
    # ANB: cert-optimal cuts (sweep on certification cohort); textbook often cited as 0–4°.
    "skeletal":   ("ANB",        [1.0, 3.5],  [2, 0, 1]),
    # Overjet: cert sweep optimal 2.5 / 5.0 mm (textbook 0 / 5 → F1 ~0.48 on cert).
    "dental":     ("Overjet",    [2.5, 5.0],  [2, 0, 1]),
    # FMA increasing: low/hypodivergent (<22) -> normal (22..28) -> high/hyperdivergent (>28)
    "vertical":   ("FMA",        [22.0, 28.0], [2, 0, 1]),
    # Profile ANB-only leg is weak; v2 uses E-line blend (see rule_layer_v2).
    "profile":    ("ANB",        [0.0, 4.0],  [2, 0, 1]),
    # lower-lip position vs Ricketts E-line increasing: retruded/normal -> protrusive (lip ahead of line)
    "protrusion": ("Eline to L lip", [1.0],    [0, 1]),
}
LEARNED_DESCRIPTORS = ["crowding", "spacing"]   # no closed-form measurement in the panel


class OrdinalRule(nn.Module):
    """Soft K-class membership from a scalar via K-1 learnable, strictly-ordered cut-points.

    p(class) is built from cumulative sigmoid exceedances, so it is a proper distribution and
    differentiable in both the cut-points and the input measurement. Cut-points stay ordered because
    they are parameterised as (first cut, positive gaps). `perm` reorders ordinal positions (sorted
    by measurement value) into the dataset's label-index space.
    """

    def __init__(self, n_classes: int, init_cuts, perm, sharp_init: float = 1.0):
        super().__init__()
        assert len(init_cuts) == n_classes - 1 and len(perm) == n_classes
        self.k = n_classes
        self.register_buffer("perm", torch.tensor(perm, dtype=torch.long))
        self.register_buffer("init_cuts", torch.tensor(init_cuts, dtype=torch.float32))
        self.cut0 = nn.Parameter(torch.tensor(float(init_cuts[0])))
        gaps = [float(init_cuts[i + 1] - init_cuts[i]) for i in range(len(init_cuts) - 1)]
        self.log_gap = nn.Parameter(torch.log(torch.tensor(gaps).clamp_min(1e-3))) if gaps else None
        self.log_sharp = nn.Parameter(torch.log(torch.tensor([float(sharp_init)])))

    def thresholds(self) -> torch.Tensor:
        cuts = [self.cut0]
        if self.log_gap is not None:
            for g in self.log_gap.exp():
                cuts.append(cuts[-1] + g)
        return torch.stack(cuts)                                          # (k-1,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,) measurement value -> p: (B, k) in dataset label-index order."""
        B = x.size(0)
        s = self.log_sharp.exp().clamp(0.05, 50.0)
        cuts = self.thresholds().to(x.dtype)                             # (k-1,)
        c = torch.sigmoid((x.unsqueeze(-1) - cuts.unsqueeze(0)) * s)     # (B, k-1) exceedances
        ones = torch.ones(B, 1, device=x.device, dtype=x.dtype)
        zeros = torch.zeros(B, 1, device=x.device, dtype=x.dtype)
        c_ext = torch.cat([ones, c, zeros], dim=-1)                      # (B, k+1)
        p_ord = (c_ext[:, :-1] - c_ext[:, 1:]).clamp_min(1e-6)           # (B, k) ordinal-position probs
        p = torch.empty_like(p_ord)
        p[:, self.perm] = p_ord                                          # reorder -> label space
        return p


class ClinicalRuleLayer(nn.Module):
    """Neuro-symbolic diagnosis: explicit clinical rules over measurements + learned fallbacks."""

    def __init__(self, d_model: int = 768):
        super().__init__()
        self.rules = nn.ModuleDict({
            k: OrdinalRule(
                DESCRIPTORS[k], cuts, perm,
                sharp_init=12.0 if k in ("skeletal", "dental", "profile") else 1.0,
            )
            for k, (_, cuts, perm) in RULE_SPEC.items()})
        self.meas_idx = {k: _MIDX[m] for k, (m, _, _) in RULE_SPEC.items()}
        # learned predicates for descriptors with no closed-form measurement
        self.learned = nn.ModuleDict({
            k: nn.Linear(d_model, DESCRIPTORS[k]) for k in LEARNED_DESCRIPTORS})
        # per-descriptor class embedding tables -> soft diagnosis tokens (B,7,d)
        self.emb = nn.ModuleDict({k: nn.Embedding(n, d_model) for k, n in DESCRIPTORS.items()})
        self.emb_ln = nn.LayerNorm(d_model)

    def forward(self, measurements: torch.Tensor, factor_summary: torch.Tensor, override=None) -> dict:
        """measurements: (B, M) calibrated clinical units ; factor_summary: (B, F, d).

        override (interpretability / controllability hook, inference-only): a dict mapping a
        descriptor to either an int class index (force that diagnosis) or the string "ablate"
        (replace the rule's signal with a uniform distribution). Used by the faithfulness experiment
        (ablate a rule -> does the plan change?) and the controllability experiment (force a class ->
        does the plan shift in the clinically-expected direction?)."""
        ctx = factor_summary.mean(dim=1)
        logits, probs, soft, acts = {}, {}, [], {}
        for k in DESCRIPTORS:                                            # keep DESCRIPTORS order -> (B,7,d)
            if k in self.rules:
                p = self.rules[k](measurements[:, self.meas_idx[k]])     # rule-driven soft membership
                acts[k] = p.detach()
            else:
                p = torch.softmax(self.learned[k](ctx), dim=-1)          # learned fallback
            if override and k in override:                               # inference-time intervention
                spec = override[k]
                if spec == "ablate":
                    p = torch.full_like(p, 1.0 / p.size(-1))             # neutralize this rule
                else:
                    p = torch.zeros_like(p)
                    p[:, int(spec)] = 1.0                                # force a specific class
            probs[k] = p
            logits[k] = (p.clamp_min(1e-6)).log()                        # CE-compatible (softmax(log p) == p)
            soft.append(p @ self.emb[k].weight)
        return {"diag_logits": logits, "diag_probs": probs,
                "diag_emb": self.emb_ln(torch.stack(soft, dim=1)),       # (B, 7, d)
                "rule_acts": acts}

    # --- interpretability / regularisation helpers ---
    def threshold_report(self) -> dict:
        """Current learned cut-points vs textbook init (for the faithfulness/interpretability table)."""
        out = {}
        for k, rule in self.rules.items():
            out[k] = {"measurement": RULE_SPEC[k][0],
                      "learned": [round(float(v), 2) for v in rule.thresholds().tolist()],
                      "textbook": [round(float(v), 2) for v in rule.init_cuts.tolist()]}
        return out

    def prior_loss(self) -> torch.Tensor:
        """Light anchor keeping learned cut-points near clinical values (preserves interpretability)."""
        dev = next(self.parameters()).device
        loss = torch.zeros((), device=dev)
        for rule in self.rules.values():
            loss = loss + ((rule.thresholds() - rule.init_cuts.to(dev)) ** 2).mean()
        return loss / max(len(self.rules), 1)
