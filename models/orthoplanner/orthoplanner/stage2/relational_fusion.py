"""② Anatomy-Biased Relational Fusion  +  ③ Uncertainty Gating.

This is the core departure from BLIP-2's Q-Former. Two differences:

  (a) DECISION-ANCHORED queries — one learnable query group per clinical decision factor
      (skeletal / vertical / crowding / profile / protrusion / extraction / scope / surgery /
      duration), not 32 generic tokens.
  (b) GEOMETRIC RELATION BIAS — when landmark tokens attend to each other, the attention logits
      carry a learned bias b_ij = MLP(dist_ij, cos θ_ij, sin θ_ij) derived from the ACTUAL pairwise
      anatomy. Attention is anatomically structured, not free-form self-attention.

Uncertainty gating (③): each landmark key/value is down-weighted by g(conf, entropy), so unreliable
landmarks contribute less; the aggregate gate also drives a hedging signal consumed downstream.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DECISION_FACTORS = [   # base architecture (9 factors) — unchanged so ablations/full-model stay consistent
    "skeletal", "vertical", "crowding", "profile", "protrusion",
    "extraction", "scope", "surgery", "duration",
]


def pairwise_geometry(landmarks: torch.Tensor):
    """(B,29,2) -> dist (B,29,29), cos (B,29,29), sin (B,29,29)."""
    diff = landmarks[:, :, None, :] - landmarks[:, None, :, :]      # (B,29,29,2)
    dist = diff.norm(dim=-1)                                         # (B,29,29)
    ang = torch.atan2(diff[..., 1], diff[..., 0])
    return dist, torch.cos(ang), torch.sin(ang)


class RelationBias(nn.Module):
    """Maps pairwise landmark geometry -> per-head additive attention bias."""

    def __init__(self, n_heads: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, n_heads))

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        dist, cos, sin = pairwise_geometry(landmarks)
        feat = torch.stack([dist, cos, sin], dim=-1)                # (B,29,29,3)
        return self.mlp(feat).permute(0, 3, 1, 2)                   # (B, heads, 29, 29)


class GatedMHA(nn.Module):
    """Multi-head attention with an optional additive attention bias and a per-key gate."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _split(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.h, self.dk).transpose(1, 2)        # (B,h,N,dk)

    def forward(self, query, key, value, attn_bias=None, key_gate=None):
        q, k, v = self._split(self.q(query)), self._split(self.k(key)), self._split(self.v(value))
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)   # (B,h,Nq,Nk)
        if attn_bias is not None:
            logits = logits + attn_bias
        if key_gate is not None:                                    # (B, Nk) in (0,1]
            logits = logits + (key_gate.clamp_min(1e-6).log())[:, None, None, :]
        attn = self.drop(F.softmax(logits, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous()
        return self.o(out.view(out.size(0), out.size(1), -1))


class _FF(nn.Module):
    def __init__(self, d, mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d * mult), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d * mult, d))

    def forward(self, x):
        return self.net(x)


class GeomSelfBlock(nn.Module):
    """Relation-biased, uncertainty-gated self-attention over the 29 landmark tokens."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = GatedMHA(d_model, n_heads, dropout)
        self.ff = _FF(d_model, dropout=dropout)

    def forward(self, x, attn_bias, key_gate):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_bias=attn_bias, key_gate=key_gate)
        x = x + self.ff(self.ln2(x))
        return x


class CrossBlock(nn.Module):
    """Decision-anchored queries cross-attend into the fused key/value set."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln_q, self.ln_kv, self.ln_ff = (nn.LayerNorm(d_model) for _ in range(3))
        self.attn = GatedMHA(d_model, n_heads, dropout)
        self.ff = _FF(d_model, dropout=dropout)

    def forward(self, q, kv, key_gate=None):
        q = q + self.attn(self.ln_q(q), self.ln_kv(kv), self.ln_kv(kv), key_gate=key_gate)
        q = q + self.ff(self.ln_ff(q))
        return q


class AnatomyBiasedFusion(nn.Module):
    def __init__(self, d_model=768, n_heads=8, n_geom_layers=2, n_cross_layers=2,
                 queries_per_factor=2, text_dim=768, dropout=0.1,
                 use_relation_bias=True, use_decision_anchored_queries=True,
                 use_uncertainty_gating=True, use_measurements=True):
        super().__init__()
        self.d = d_model
        self.factors = DECISION_FACTORS
        self.qpf = queries_per_factor
        self.n_queries = len(self.factors) * queries_per_factor
        self.use_relation_bias = use_relation_bias
        self.use_decision_anchored_queries = use_decision_anchored_queries
        self.use_uncertainty_gating = use_uncertainty_gating
        self.use_measurements = use_measurements

        # landmark token = proj([x, y, conf, entropy]) + per-landmark type embedding
        self.lm_proj = nn.Sequential(nn.Linear(4, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.lm_type = nn.Parameter(torch.randn(29, d_model) * 0.02)
        self.uncert_gate = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 1))

        self.rel_bias = RelationBias(n_heads)
        self.geom = nn.ModuleList([GeomSelfBlock(d_model, n_heads, dropout) for _ in range(n_geom_layers)])

        self.text_proj = nn.Linear(text_dim, d_model) if text_dim != d_model else nn.Identity()
        # segment embeddings to tag landmark / measurement / text keys
        self.seg_emb = nn.Parameter(torch.randn(3, d_model) * 0.02)

        self.query = nn.Parameter(torch.randn(self.n_queries, d_model) * 0.02)
        self.factor_emb = nn.Parameter(torch.randn(len(self.factors), d_model) * 0.02)
        self.cross = nn.ModuleList([CrossBlock(d_model, n_heads, dropout) for _ in range(n_cross_layers)])
        self.out_ln = nn.LayerNorm(d_model)

    def init_queries(self, B, device):
        q = self.query.unsqueeze(0).expand(B, -1, -1).clone()
        if self.use_decision_anchored_queries:                       # tag each query with its factor
            fe = self.factor_emb.repeat_interleave(self.qpf, dim=0).unsqueeze(0)
            q = q + fe
        return q

    def forward(self, landmarks, conf, entropy, measurement_tokens, text_tokens, text_mask=None):
        B = landmarks.size(0)

        # --- landmark tokens + uncertainty gate (③) ---
        feat = torch.cat([landmarks, conf, entropy], dim=-1)         # (B,29,4)
        lm_tok = self.lm_proj(feat) + self.lm_type.unsqueeze(0)
        if self.use_uncertainty_gating:
            gate = torch.sigmoid(self.uncert_gate(torch.cat([conf, entropy], dim=-1)).squeeze(-1))
        else:
            gate = torch.ones(B, landmarks.size(1), device=landmarks.device)  # no down-weighting

        # --- geometry-biased self encoding (②) ---
        bias = self.rel_bias(landmarks) if self.use_relation_bias else None  # None -> plain self-attn
        x = lm_tok + self.seg_emb[0]
        for blk in self.geom:
            x = blk(x, attn_bias=bias, key_gate=gate)
        lm_geo = x                                                   # (B,29,d)

        # --- assemble key/value set: [landmarks ; (measurements) ; text] ---
        parts = [lm_geo]
        seg = [gate]                                                 # per-token key gate, segment-aligned
        if self.use_measurements and measurement_tokens is not None:
            parts.append(measurement_tokens + self.seg_emb[1])
            seg.append(torch.ones(B, measurement_tokens.size(1), device=lm_geo.device))
        txt = self.text_proj(text_tokens) + self.seg_emb[2]
        parts.append(txt)
        seg.append(text_mask.float() if text_mask is not None else torch.ones(B, txt.size(1), device=lm_geo.device))
        kv = torch.cat(parts, dim=1)
        key_gate = torch.cat(seg, dim=1)

        # --- decision-anchored cross-attention (②) ---
        q = self.init_queries(B, kv.device)
        for blk in self.cross:
            q = blk(q, kv, key_gate=key_gate)
        q = self.out_ln(q)

        # per-factor pooled summary (for the decision head)
        factor_summary = q.view(B, len(self.factors), self.qpf, self.d).mean(dim=2)  # (B, F, d)
        # scalar hedge signal in [0,1]: high mean landmark uncertainty -> hedge
        hedge = (1.0 - gate).mean(dim=1, keepdim=True)              # (B,1)
        return {"query_tokens": q, "factor_summary": factor_summary, "hedge": hedge, "lm_gate": gate}
