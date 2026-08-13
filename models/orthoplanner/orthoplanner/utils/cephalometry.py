"""Differentiable cephalometric geometry.

The 29-landmark order is the CEPHA29 / Aariz canonical order (the order in which landmarks
appear in every Aariz annotation JSON, verified against the trained encoder). Every measurement
below is a *closed-form, differentiable* function of the predicted landmark coordinates, so it can
be (a) used as a geometry-grounded token and (b) supervised against the clinical
`measurements_results` already present in the dataset.

All landmark coordinates are expected normalized to [0, 1] in a square (padded) image frame, so x
and y share a scale and angles/distances are geometrically valid. Angles are returned in degrees
and are scale-invariant; distances are in normalized units (a learnable per-measurement affine in
`measurement_bridge.py` maps them to clinical mm/°).
"""
from __future__ import annotations

import math

import torch

# --- canonical 29-landmark index map (CEPHA29 order) ---------------------------------------------
LANDMARKS = [
    "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Pn", "R", "S", "Ar", "Co", "Gn", "Go",
    "Po", "LPM", "LIT", "LMT", "UPM", "UIA", "UIT", "UMT", "LIA", "Li", "Ls", "Nprime",
    "Pogprime", "Sn",
]
IDX = {name: i for i, name in enumerate(LANDMARKS)}
NUM_LANDMARKS = len(LANDMARKS)
assert NUM_LANDMARKS == 29

# Core 13: differentiable bridge supervision (landmark geometry -> named scalars).
MEASUREMENT_KEYS = [
    "SNA", "SNB", "ANB", "FMA", "Overjet", "Overbite",
    "Mx1 to SN", "MP to Mn1", "IIA", "Eline to U lip", "Eline to L lip", "APDI",
    "Combination Factor(CF)",
]
NUM_MEASUREMENTS = len(MEASUREMENT_KEYS)

# Full AnalysisChart.csv set (~90%+ coverage + common Bjork rows) for measurements_results.
MEASUREMENT_KEYS_ALL = MEASUREMENT_KEYS + [
    "A to N Perpendicular",
    "Pog to N Perpendicular",
    "Wits Appraisal",
    "ODI",
    "Mx1 to NA (deg)",
    "Mx1 to NA (mm)",
    "Mx1 to A-Pog",
    "Md1 to NB (deg)",
    "Md1 to NB (mm)",
    "Md1 to A-Pog",
    "Body to Ant. Cranial Base Ratio",
    "Facial Height Ratio",
    "Occ Plane",
    "STms-Mx1",
    "SN-GoGn",
    "SN-GoMe",
    "Saddle Angle",
    "Articular Angle",
    "Gonial angle",
    "Bjork Sum",
]

# Subset of MEASUREMENT_KEYS_ALL derivable in closed form from the 29 Aariz landmarks.
# Used by rule_layer_v2 feature analysis and future extended bridge supervision.
LANDMARK_DERIVED_EXTENDED_KEYS = [
    "Wits Appraisal",
    "A to N Perpendicular",
    "Pog to N Perpendicular",
    "ODI",
    "Mx1 to NA (deg)",
    "Mx1 to NA (mm)",
    "Mx1 to A-Pog",
    "Md1 to NB (deg)",
    "Md1 to NB (mm)",
    "Md1 to A-Pog",
    "Saddle Angle",
    "Articular Angle",
    "Gonial angle",
    "Bjork Sum",
    "SN-GoGn",
    "SN-GoMe",
]

RULE_LAYER_V2_KEYS = MEASUREMENT_KEYS + [
    k for k in LANDMARK_DERIVED_EXTENDED_KEYS if k not in MEASUREMENT_KEYS
]
NUM_RULE_V2_MEASUREMENTS = len(RULE_LAYER_V2_KEYS)
RULE_V2_MIDX = {k: i for i, k in enumerate(RULE_LAYER_V2_KEYS)}

# Keys required at inference for ClinicalRuleLayerV2 (high cert coverage).
RULE_LAYER_V2_REQUIRED_KEYS = [
    "ANB", "APDI", "Wits Appraisal", "Eline to L lip", "FMA",
]

# Neutral imputation for optional / missing extended columns (gate ≈ 1).
RULE_LAYER_V2_NEUTRAL: dict[str, float] = {
    "Wits Appraisal": 0.0,
    "APDI": 81.0,
    "A to N Perpendicular": 0.0,
    "Pog to N Perpendicular": 0.0,
    "ODI": 75.0,
}

# Keys in MEASUREMENT_KEYS_ALL with no closed-form landmark formula in this module.
CHART_ONLY_KEYS = [k for k in MEASUREMENT_KEYS_ALL if k not in RULE_LAYER_V2_KEYS]

_EPS = 1e-6


def _foot_on_line(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Foot of the perpendicular from p onto the infinite line through a->b."""
    ab = b - a
    t = ((p - a) * ab).sum(-1, keepdim=True) / (ab.pow(2).sum(-1, keepdim=True) + _EPS)
    return a + t * ab


def compute_landmark_extended(landmarks: torch.Tensor) -> torch.Tensor:
    """Map (B, 29, 2) landmarks -> (B, len(LANDMARK_DERIVED_EXTENDED_KEYS)).

    Raw geometric primitives in normalized units / degrees; bridge affine maps to clinical units.
    """
    g = lambda name: _pt(landmarks, name)  # noqa: E731

    occ_a, occ_b = g("ANS"), g("PNS")
    fa = _foot_on_line(g("A"), occ_a, occ_b)
    fb = _foot_on_line(g("B"), occ_a, occ_b)
    occ = occ_b - occ_a
    occ_n = torch.stack([-occ[..., 1], occ[..., 0]], dim=-1)
    occ_n = occ_n / (occ_n.norm(dim=-1, keepdim=True) + _EPS)
    wits = ((fb - fa) * occ_n).sum(-1)

    fh_a, fh_b = g("Po"), g("Or")
    n_pt = g("N")
    a_to_n = _signed_point_to_line(g("A"), n_pt, fh_b)
    pog_to_n = _signed_point_to_line(g("Pog"), n_pt, fh_b)

    palatal_FH = _line_angle(g("ANS"), g("PNS"), fh_a, fh_b)
    ab_mand = _line_angle(g("A"), g("B"), g("Go"), g("Me"))
    odi = ab_mand + palatal_FH

    mx1_na_deg = _line_angle(n_pt, g("A"), g("UIA"), g("UIT"))
    mx1_na_mm = _signed_point_to_line(g("UIT"), n_pt, g("A"))
    mx1_apog = _signed_point_to_line(g("UIT"), g("A"), g("Pog"))
    md1_nb_deg = _line_angle(n_pt, g("B"), g("LIA"), g("LIT"))
    md1_nb_mm = _signed_point_to_line(g("LIT"), n_pt, g("B"))
    md1_apog = _signed_point_to_line(g("LIT"), g("A"), g("Pog"))

    saddle = _angle_at(g("S"), g("N"), g("Ar"))
    articular = _angle_at(g("S"), g("Ar"), g("Go"))
    gonial = _angle_at(g("Ar"), g("Go"), g("Me"))
    bjork = saddle + articular + gonial
    sn_gogn = _line_angle(g("S"), g("N"), g("Go"), g("Gn"))
    sn_gome = _line_angle(g("S"), g("N"), g("Go"), g("Me"))

    return torch.stack(
        [
            wits, a_to_n, pog_to_n, odi,
            mx1_na_deg, mx1_na_mm, mx1_apog,
            md1_nb_deg, md1_nb_mm, md1_apog,
            saddle, articular, gonial, bjork, sn_gogn, sn_gome,
        ],
        dim=-1,
    )


def compute_rule_v2_measurements(landmarks: torch.Tensor) -> torch.Tensor:
    """Core 13 + landmark-extended scalars in RULE_LAYER_V2_KEYS order."""
    core = compute_measurements(landmarks)
    ext = compute_landmark_extended(landmarks)
    ext_map = {k: ext[..., i] for i, k in enumerate(LANDMARK_DERIVED_EXTENDED_KEYS)}
    cols = []
    for k in RULE_LAYER_V2_KEYS:
        if k in MEASUREMENT_KEYS:
            cols.append(core[..., MEASUREMENT_KEYS.index(k)])
        else:
            cols.append(ext_map[k])
    return torch.stack(cols, dim=-1)


def gt_rule_v2_vector(meas: dict, *, impute: bool = False) -> list[float]:
    """Clinical measurement vector for rule_layer_v2 (missing -> NaN unless impute)."""
    row = _measurement_vector(meas, RULE_LAYER_V2_KEYS)
    if not impute:
        return row
    out = []
    for i, k in enumerate(RULE_LAYER_V2_KEYS):
        v = row[i]
        if math.isnan(v):
            if k in MEASUREMENT_KEYS:
                out.append(float("nan"))
            else:
                out.append(float(RULE_LAYER_V2_NEUTRAL.get(k, 0.0)))
        else:
            out.append(v)
    return out


def _pt(lm: torch.Tensor, name: str) -> torch.Tensor:
    """Select a landmark (..., 2) by name from a (B, 29, 2) tensor."""
    return lm[..., IDX[name], :]


def _angle_at(p: torch.Tensor, vertex: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Interior angle (degrees) at `vertex` between rays vertex->p and vertex->q."""
    v1 = p - vertex
    v2 = q - vertex
    dot = (v1 * v2).sum(-1)
    cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    return torch.rad2deg(torch.atan2(cross.abs() + _EPS, dot))


def _line_angle(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Acute/obtuse angle (degrees, 0..180) between line a->b and line c->d."""
    v1 = b - a
    v2 = d - c
    dot = (v1 * v2).sum(-1)
    cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    return torch.rad2deg(torch.atan2(cross.abs() + _EPS, dot))


def _signed_point_to_line(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Signed perpendicular distance from point p to line a->b (normalized units)."""
    d = b - a
    n = torch.stack([-d[..., 1], d[..., 0]], dim=-1)  # left normal
    n = n / (n.norm(dim=-1, keepdim=True) + _EPS)
    return ((p - a) * n).sum(-1)


def compute_measurements(landmarks: torch.Tensor) -> torch.Tensor:
    """Map (B, 29, 2) normalized landmarks -> (B, NUM_MEASUREMENTS) geometric primitives.

    Output column order == MEASUREMENT_KEYS. Distances are in normalized units; angles in degrees.
    These are raw geometric primitives; `MeasurementBridge` applies a learnable per-measurement
    affine to align them with clinical units before the supervision loss.
    """
    g = lambda name: _pt(landmarks, name)  # noqa: E731

    SNA = _angle_at(g("S"), g("N"), g("A"))
    SNB = _angle_at(g("S"), g("N"), g("B"))
    ANB = SNA - SNB
    FMA = _line_angle(g("Po"), g("Or"), g("Go"), g("Me"))          # FH vs mandibular plane

    # Incisor relationships (horizontal / vertical in the padded square frame).
    overjet = (g("UIT")[..., 0] - g("LIT")[..., 0])
    overbite = (g("LIT")[..., 1] - g("UIT")[..., 1])

    Mx1_SN = _line_angle(g("S"), g("N"), g("UIA"), g("UIT"))        # upper incisor axis vs SN
    IMPA = _line_angle(g("Go"), g("Me"), g("LIA"), g("LIT"))        # lower incisor axis vs mand. plane
    IIA = _line_angle(g("UIA"), g("UIT"), g("LIA"), g("LIT"))       # interincisal angle

    eline_ul = _signed_point_to_line(g("Ls"), g("Pn"), g("Pogprime"))
    eline_ll = _signed_point_to_line(g("Li"), g("Pn"), g("Pogprime"))

    # Composite indices (differentiable proxies; supervised with their own affine).
    facial_FH = _line_angle(g("N"), g("Pog"), g("Po"), g("Or"))    # facial plane vs FH
    ab_facial = _line_angle(g("A"), g("B"), g("N"), g("Pog"))      # A-B plane vs facial plane
    palatal_FH = _line_angle(g("ANS"), g("PNS"), g("Po"), g("Or"))  # palatal plane vs FH
    APDI = facial_FH + ab_facial + palatal_FH
    ab_mand = _line_angle(g("A"), g("B"), g("Go"), g("Me"))        # A-B plane vs mandibular plane
    ODI = ab_mand + palatal_FH                                      # overbite depth indicator proxy
    CF = APDI + ODI                                                 # combination factor proxy

    return torch.stack(
        [SNA, SNB, ANB, FMA, overjet, overbite, Mx1_SN, IMPA, IIA, eline_ul, eline_ll, APDI, CF],
        dim=-1,
    )


def _measurement_vector(meas: dict, keys: list[str]) -> list:
    out = []
    for k in keys:
        v = meas.get(k, None)
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


def gt_measurement_vector(meas: dict) -> list:
    """Bridge supervision vector (13 core keys). Missing -> NaN."""
    return _measurement_vector(meas, MEASUREMENT_KEYS)


def gt_measurement_vector_all(meas: dict) -> list:
    """Full AnalysisChart vector (MEASUREMENT_KEYS_ALL). Missing -> NaN."""
    return _measurement_vector(meas, MEASUREMENT_KEYS_ALL)
