"""GUSR — Geometry-Grounded, Uncertainty-Aware Structured Reasoner.

Ties together ① measurement bridge, ② anatomy-biased relational fusion, ③ uncertainty gating, and
④ constrained decision head, and produces the LLM conditioning prefix:

    prefix = [ decision-anchored query tokens ; soft decision embeddings ; uncertainty token ]

projected into LLM hidden space. The prefix is prepended (in embedding space) to the LoRA LLM, which
emits the treatment plan — the only output. Auxiliary tensors (measurements, attribute logits,
confidences, duration) are returned for the training losses only.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .decision_head import DecisionHead, DiagnosisHead
from .measurement_bridge import MeasurementBridge
from .relational_fusion import AnatomyBiasedFusion
from .rule_layer import ClinicalRuleLayer
from .rule_layer_v2 import ClinicalRuleLayerV2
from ..utils.cephalometry import compute_rule_v2_measurements

NUM_TREATMENT_TYPES = 4   # comprehensive / growth_modification / growth_plus_comprehensive / surgical


class TreatmentTypeHead(nn.Module):
    """Predict the treatment-type (plan opener) from the derived diagnosis + non-diagnostic
    metadata (age, gender). Emits a soft conditioning token so the LLM picks the right opener
    (e.g. growth-modification vs comprehensive) instead of collapsing to the majority phrase."""

    def __init__(self, d_model: int, n_types: int = NUM_TREATMENT_TYPES, n_meta: int = 2):
        super().__init__()
        self.cls = nn.Sequential(
            nn.Linear(d_model + n_meta, d_model), nn.GELU(), nn.Linear(d_model, n_types))
        self.emb = nn.Embedding(n_types, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, diag_ctx: torch.Tensor, metadata: torch.Tensor):
        h = torch.cat([diag_ctx, metadata], dim=-1)
        logits = self.cls(h)
        soft = torch.softmax(logits, dim=-1) @ self.emb.weight
        return logits, self.ln(soft).unsqueeze(1)   # (B, n_types), (B, 1, d)


class GUSRReasoner(nn.Module):
    def __init__(self, d_model=768, llm_dim=4096, text_dim=768, n_heads=8,
                 n_geom_layers=2, n_cross_layers=2, queries_per_factor=2,
                 dur_mean=24.0, dur_std=12.0, dropout=0.1,
                 use_measurement_bridge=True, use_relation_bias=True,
                 use_decision_anchored_queries=True, use_uncertainty_gating=True,
                 use_decision_head=True, use_diagnosis_derivation=False, use_metadata=False,
                 use_rule_layer=False, measurement_primary=False, diag_drives_decision=False,
                 use_rule_v2=False, use_ttype=False):
        super().__init__()
        self.use_bridge = use_measurement_bridge
        self.use_head = use_decision_head
        self.use_diag = use_diagnosis_derivation
        self.use_metadata = use_metadata
        self.measurement_primary = measurement_primary
        # the rule layer derives diagnosis from measurements -> it requires the bridge
        self.use_rule = use_rule_layer and use_diagnosis_derivation and use_measurement_bridge
        # v3.2: multi-measure rule layer (gated dental/skeletal + evidence-blended profile)
        self.use_rule_v2 = use_rule_v2 and use_diagnosis_derivation and use_measurement_bridge
        # v3.1: feed the derived diagnosis into the decision head (diagnosis -> decision -> plan)
        self.diag_drives_decision = diag_drives_decision and use_diagnosis_derivation and use_decision_head
        # v3.3: predict the treatment-type (opener) from diagnosis + age so the plan opener is not collapsed
        self.use_ttype = use_ttype and use_diagnosis_derivation
        if use_measurement_bridge:
            self.bridge = MeasurementBridge(d_model)
        if use_diagnosis_derivation:
            if self.use_rule_v2:
                self.diag_head = ClinicalRuleLayerV2(d_model)
            elif self.use_rule:
                self.diag_head = ClinicalRuleLayer(d_model)
            else:
                self.diag_head = DiagnosisHead(d_model)
        if use_metadata:
            self.meta_proj = nn.Sequential(nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        if self.use_ttype:
            self.ttype_head = TreatmentTypeHead(d_model)
        self.fusion = AnatomyBiasedFusion(
            d_model=d_model, n_heads=n_heads, n_geom_layers=n_geom_layers,
            n_cross_layers=n_cross_layers, queries_per_factor=queries_per_factor,
            text_dim=text_dim, dropout=dropout, use_relation_bias=use_relation_bias,
            use_decision_anchored_queries=use_decision_anchored_queries,
            use_uncertainty_gating=use_uncertainty_gating, use_measurements=use_measurement_bridge)
        if use_decision_head:
            self.head = DecisionHead(d_model, dur_mean=dur_mean, dur_std=dur_std,
                                     use_diag_context=self.diag_drives_decision)

        # uncertainty token: a learned vector scaled by the aggregate hedge signal
        self.unc_vec = nn.Parameter(torch.randn(d_model) * 0.02)
        self.to_llm = nn.Linear(d_model, llm_dim)
        self.prefix_ln = nn.LayerNorm(llm_dim)

    def forward(self, enc_out: dict, text_tokens: torch.Tensor, text_mask=None,
                apply_constraints: bool = True, metadata=None, gt_measurements=None,
                diag_override=None, gt_measurements_v2=None):
        lm = enc_out["landmarks"]
        conf, ent = enc_out["conf"], enc_out["entropy"]

        if self.use_bridge:
            measurements, meas_tokens = self.bridge(lm)                    # ① geometry-derived (fallback)
        else:
            measurements, meas_tokens = None, None
        # v3.1 measurement-primary: drive the rules + measurement tokens with the accurate clinical
        # measurements; the geometry pathway above is kept (supervised by L_measure) as the fallback.
        rule_meas = measurements
        if self.measurement_primary and gt_measurements is not None:
            rule_meas = gt_measurements.to(lm.dtype)
            if self.use_bridge:
                meas_tokens = self.bridge.tokenize(rule_meas)
        # v3.2: the multi-measure rule layer needs the 29-key RULE_LAYER_V2 vector. Use the
        # GT chart vector in measurement-primary mode; otherwise derive it from predicted landmarks.
        rule_meas_v2 = None
        if self.use_rule_v2:
            if self.measurement_primary and gt_measurements_v2 is not None:
                rule_meas_v2 = gt_measurements_v2.to(lm.dtype)
            else:
                rule_meas_v2 = compute_rule_v2_measurements(lm).to(lm.dtype)
        fused = self.fusion(lm, conf, ent, meas_tokens, text_tokens, text_mask)  # ②③

        out = {"measurements": measurements, "hedge": fused["hedge"], "lm_gate": fused["lm_gate"],
               "attr_logits": None, "attr_conf": None, "attr_conf_logit": None,
               "dur_z": None, "dur_months": None, "diag_logits": None, "ttype_logits": None}

        # derive the diagnosis FIRST so it can drive the decision head (diagnosis -> decision -> plan)
        diag = None
        if self.use_diag:
            if self.use_rule_v2:                                          # v3.2: multi-measure rules
                diag = self.diag_head(rule_meas_v2, fused["factor_summary"], override=diag_override)
                out["rule_acts"] = diag["rule_acts"]
                out["rule_prior"] = self.diag_head.prior_loss()
            elif self.use_rule:                                           # v3: neuro-symbolic rules
                diag = self.diag_head(rule_meas, fused["factor_summary"], override=diag_override)
                out["rule_acts"] = diag["rule_acts"]
                out["rule_prior"] = self.diag_head.prior_loss()
            else:                                                         # v2: black-box head
                diag = self.diag_head(fused["factor_summary"])
            out["diag_logits"] = diag["diag_logits"]
        diag_ctx = diag["diag_emb"].mean(dim=1) if (diag is not None and self.diag_drives_decision) else None

        parts = [fused["query_tokens"]]
        if self.use_head:
            dec = self.head(fused["factor_summary"], apply_constraints, diag_ctx=diag_ctx)  # ④
            parts.append(dec["decision_emb"])
            out.update({"attr_logits": dec["logits"], "attr_conf": dec["conf"],
                        "attr_conf_logit": dec["conf_logit"], "dur_z": dec["dur_z"],
                        "dur_months": dec["dur_months"]})
        if self.use_diag:                                                 # diagnosis tokens in the prefix
            parts.append(diag["diag_emb"])
        # v3.3: treatment-type (opener) token — predicted from diagnosis + age, conditions the plan opener
        if self.use_ttype and diag is not None:
            B = fused["factor_summary"].size(0)
            meta = metadata
            if meta is None:
                meta = fused["factor_summary"].new_zeros(B, 2)
            tt_logits, tt_tok = self.ttype_head(diag["diag_emb"].mean(dim=1), meta)
            out["ttype_logits"] = tt_logits
            parts.append(tt_tok)
        if self.use_metadata and metadata is not None:                    # age + gender token
            parts.append(self.meta_proj(metadata).unsqueeze(1))

        unc_tok = (fused["hedge"] * self.unc_vec.unsqueeze(0)).unsqueeze(1)  # (B,1,d)
        parts.append(unc_tok)
        out["prefix"] = self.prefix_ln(self.to_llm(torch.cat(parts, dim=1)))
        return out
