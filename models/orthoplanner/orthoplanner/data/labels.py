"""Structured-label extraction from five-fold-final.json records.

Maps each record to the supervised attributes used by the decision head + measurement bridge:
  scope       : Full=0, Partial=1, Surgical=2          (index 2 must == surgical, see decision_head)
  extraction  : without=0, with=1
  surgery     : no=0, yes=1   (orthognathic)
  duration    : duration_months (float)
  measurements: vector aligned to cephalometry.MEASUREMENT_KEYS  (NaN where missing)
"""
from __future__ import annotations

import math

from ..utils.cephalometry import gt_measurement_vector, gt_rule_v2_vector

SCOPE = {"full": 0, "partial": 1, "surgical": 2}


def scope_label(rec) -> int:
    """Decision-head scope label. v2 phase1 → -100 (masked in loss)."""
    scope_v2 = (rec.get("scope_v2") or "").strip()
    if scope_v2:
        from ..data.scope_type_v2 import scope_v2_to_sc_eval
        if scope_v2 == "surgical" or (rec.get("treatment_type_v2") or "") == "surgical":
            return SCOPE["surgical"]
        mapped = scope_v2_to_sc_eval(scope_v2)
        if mapped == "full":
            return SCOPE["full"]
        if mapped == "partial":
            return SCOPE["partial"]
        return -100
    st = (rec.get("surgery_type") or "").lower()
    if "orthognathic" in st:
        return SCOPE["surgical"]
    fo = (rec.get("full_ortho") or "").strip().lower()
    if fo in ("full",):
        return SCOPE["full"]
    if fo in ("partial",):
        return SCOPE["partial"]
    if fo in ("surgical",):
        return SCOPE["surgical"]
    return -100


def extraction_label(rec) -> int:
    e = rec.get("extraction")
    if e is None:
        return -100
    e = str(e).strip().lower()
    if e == "" or "non" in e or e in ("no", "none"):
        return 0
    if "extraction" in e or any(ch.isdigit() for ch in e):
        return 1
    return -100


def surgery_label(rec) -> int:
    st = (rec.get("surgery_type") or "").lower()
    add = (rec.get("additional_surgery") or "").strip()
    if "orthognathic" in st or len(add) > 0:
        return 1
    if st:  # surgery_type known and not orthognathic
        return 0
    return -100


def duration_months(rec):
    v = rec.get("duration_months")
    try:
        v = float(v)
        return v if v > 0 else math.nan
    except (TypeError, ValueError):
        return math.nan


def measurement_vector(rec):
    return gt_measurement_vector(rec.get("measurements_results") or {})


def measurement_vector_v2(rec):
    """29-key RULE_LAYER_V2 chart vector (optional extended cols neutral-imputed; required stay NaN)."""
    return gt_rule_v2_vector(rec.get("measurements_results") or {}, impute=True)


def _treatment_type_label(rec) -> int:
    from ..data.scope_type_v2 import treatment_type_label
    return treatment_type_label(rec)


# --- v2: diagnostic descriptors derived from geometry (image-primary planning) ---
DESCRIPTORS = {            # field -> (num_classes, mapping fn)
    "skeletal": 3, "dental": 3, "crowding": 2, "spacing": 2,
    "profile": 3, "protrusion": 2, "vertical": 3,
}


def _roman_class(v):
    """I/II/III -> 0/1/2. Check III before II before I (substring-safe)."""
    v = (v or "").strip().lower()
    if "iii" in v or v.startswith("3"):
        return 2
    if "ii" in v or v.startswith("2"):
        return 1
    if "i" in v or v.startswith("1"):
        return 0
    return -100


def descriptor_labels(rec) -> dict:
    """Normalize the messy structured fields into clean class indices (−100 = unknown)."""
    out = {}
    out["skeletal"] = _roman_class(rec.get("skeletal"))
    out["dental"] = _roman_class(rec.get("dental"))
    cr = (rec.get("crowding") or "").strip().lower()
    out["crowding"] = 0 if cr in ("", "none") else 1
    sp = (rec.get("spacing") or "").strip().lower()
    out["spacing"] = 0 if sp in ("", "none") else 1
    pr = (rec.get("profile") or "").strip().lower()
    out["profile"] = 1 if "convex" in pr else (2 if "concave" in pr else
                                               (0 if ("straight" in pr or "normal" in pr) else -100))
    pt = (rec.get("protrusion") or "").strip().lower()
    out["protrusion"] = 0 if pt in ("", "none") else 1
    vt = (rec.get("vertical") or "").strip().lower()
    out["vertical"] = 1 if "high" in vt else (2 if "low" in vt else (0 if "normal" in vt else -100))
    return out


def metadata_vec(rec):
    """Non-diagnostic metadata: [age_years/20 (≈normalized), gender female=1/male=0]."""
    try:
        age = float(rec.get("age_years")) / 20.0
    except (TypeError, ValueError):
        age = 0.7   # ~14y default
    gender = 1.0 if (rec.get("gender") or "").lower().startswith("f") else 0.0
    return [age, gender]


def labels_for(rec) -> dict:
    return {
        "y_scope": scope_label(rec),
        "y_extraction": extraction_label(rec),
        "y_surgery": surgery_label(rec),
        "dur_months": duration_months(rec),
        "gt_measurements": measurement_vector(rec),
        "gt_measurements_v2": measurement_vector_v2(rec),  # v3.2: 29-key rule-layer-v2 vector
        "y_ttype": _treatment_type_label(rec),   # v3.3: treatment-type (opener) class
        "descriptors": descriptor_labels(rec),   # v2: diagnostic descriptors (dict of 7)
        "metadata": metadata_vec(rec),           # v2: [age/20, gender]
    }
