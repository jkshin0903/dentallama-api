"""Scope / treatment-type v2 relabeling (revision cohort).

Canonical spec: OrthoPlanner/data/SCOPE_TYPE_V2_SPEC.md

Axes:
  - ``treatment_type_v2``: EN opener (comprehensive | growth_mod | growth_plus | surgical)
  - ``scope_v2`` / ``full_ortho_v2``: orthodontic extent for *comprehensive* treatment only;
    ``phase1_functional`` marks growth phase (not a full/partial scope label).
"""
from __future__ import annotations

import re
from typing import Any

SCOPE_V2_COMPREHENSIVE = "comprehensive"
SCOPE_V2_LOCALIZED = "localized"
SCOPE_V2_PHASE1 = "phase1_functional"
SCOPE_V2_SURGICAL = "surgical"

TYPE_V2_COMPREHENSIVE = "comprehensive"
TYPE_V2_GROWTH = "growth_modification"
TYPE_V2_GROWTH_PLUS = "growth_plus_comprehensive"
TYPE_V2_SURGICAL = "surgical"

# Stable index map for the treatment-type prediction head (opener selection).
TYPE_V2_INDEX = {
    TYPE_V2_COMPREHENSIVE: 0,
    TYPE_V2_GROWTH: 1,
    TYPE_V2_GROWTH_PLUS: 2,
    TYPE_V2_SURGICAL: 3,
}
NUM_TREATMENT_TYPES = len(TYPE_V2_INDEX)


def treatment_type_label(rec: dict) -> int:
    """treatment_type_v2 -> class index (opener); -100 if missing/unknown."""
    t = (rec.get("treatment_type_v2") or "").strip()
    return TYPE_V2_INDEX.get(t, -100)

FULL_ORTHO_V2_MAP = {
    SCOPE_V2_COMPREHENSIVE: "Full",
    SCOPE_V2_LOCALIZED: "Localized",
    SCOPE_V2_PHASE1: "Phase1",
    SCOPE_V2_SURGICAL: "Surgical",
}

# Chart-primary regex (see SCOPE_TYPE_V2_SPEC.md).
_GROWTH_KO = re.compile(
    r"악정형|약정형|약정항|orthopedic|orthopaedic|facemask|face\s*mask|"
    r"head\s*gear|headgear|activator|twin[\s-]?block|bionator|chin\s*cup|"
    r"\brpe\b|marpe|expansion\s*screw|bite\s*block|schwartz|teuscher",
    re.I,
)
# Palatal/RPE expansion alone is NOT growth if no orthopedic appliance context.
_EXPANSION_ONLY = re.compile(r"palatal\s*expansion|비발치\s*\(\s*palatal", re.I)

_TWO_PHASE_KO = re.compile(
    r"전체\s*교정|전제\s*교정|2\s*차\s*교정|2차교정|2차\s*치아|본교정|"
    r"성장완료\s*후|성장완료후|"
    r"차\s*후\s*발치\s*교정|차\s*후.*교정|"
    r"영구치열기.*치아\s*교정|"
    r"재진단.*치아\s*교정|재평가.*치아\s*교정|이후.*치아\s*교정|"
    r"악정형.*완료.*치아\s*교정|추후\s*2\s*차\s*교정|"
    r"성장패턴.*전체\s*교정|전체\s*교정\s*\(",
    re.I,
)
_LOCALIZED_KO = re.compile(
    r"부분\s*교정|부분교정|\b부분\b|"
    r"전치부|전치\s*부|반대교합|cross\s*bite|crossbite|"
    r"3전치|4전치|2\s*[x×]\s*4|2x4|utility\s*arch",
    re.I,
)
_FULL_COMP_KO = re.compile(
    r"전체\s*교정|전제\s*교정|비발치\s*전체|전악\s*교정",
    re.I,
)
_LOCALIZED_TOOLS = re.compile(
    r"mta|pwb|lingual\s*arch|labiolingual|nha|utility\s*arch|2x4",
    re.I,
)

# EN template slots for treatment_plan regeneration.
V2_EN_OPENER: dict[str, str] = {
    TYPE_V2_COMPREHENSIVE: "Orthodontic treatment,",
    TYPE_V2_GROWTH: "Growth modification orthodontic treatment,",
    TYPE_V2_GROWTH_PLUS: (
        "Growth modification followed by comprehensive orthodontic treatment,"
    ),
    TYPE_V2_SURGICAL: "Orthodontic and orthognathic combined treatment,",
}

V2_EN_SCOPE: dict[str, str] = {
    SCOPE_V2_COMPREHENSIVE: "with full orthodontic scope,",
    SCOPE_V2_LOCALIZED: "with partial orthodontic scope,",
    SCOPE_V2_PHASE1: "with interceptive orthodontic scope,",
    SCOPE_V2_SURGICAL: "with surgical orthodontic scope,",
}

VALID_TYPE_SCOPE: set[tuple[str, str]] = {
    (TYPE_V2_COMPREHENSIVE, SCOPE_V2_COMPREHENSIVE),
    (TYPE_V2_COMPREHENSIVE, SCOPE_V2_LOCALIZED),
    (TYPE_V2_GROWTH, SCOPE_V2_PHASE1),
    (TYPE_V2_GROWTH_PLUS, SCOPE_V2_COMPREHENSIVE),
    (TYPE_V2_GROWTH_PLUS, SCOPE_V2_LOCALIZED),
    (TYPE_V2_SURGICAL, SCOPE_V2_SURGICAL),
}


def _raw_text(rec: dict[str, Any]) -> str:
    return ((rec.get("treatment_plan_raw") or "") + " " + (rec.get("diagnosis") or "")).strip()


def _has_growth_chart(raw: str) -> bool:
    if not _GROWTH_KO.search(raw):
        return False
    if _EXPANSION_ONLY.search(raw) and not re.search(
        r"악정형|약정|orthopedic|facemask|activator|twin|bionator|chin\s*cup|head\s*gear|"
        r"bonded\s*rpe|face\s*mask",
        raw,
        re.I,
    ):
        return False
    return True


def _tags(rec: dict[str, Any]) -> list[str]:
    raw = _raw_text(rec)
    tags: list[str] = []
    if _has_growth_chart(raw):
        tags.append("growth_raw")
    if _TWO_PHASE_KO.search(raw):
        tags.append("two_phase_raw")
    if _LOCALIZED_KO.search(raw):
        tags.append("localized_raw")
    tools = (rec.get("tools") or "")
    if _LOCALIZED_TOOLS.search(tools):
        tags.append("localized_appliance")
    if str(rec.get("extraction") or "").lower().startswith("non"):
        tags.append("non_extraction")
    return tags


def en_plan_prefix(type_v2: str, scope_v2: str) -> str:
    """Canonical opener + scope clause for EN treatment_plan (no extraction/duration)."""
    if (type_v2, scope_v2) not in VALID_TYPE_SCOPE:
        raise ValueError(f"Invalid type/scope pair: {type_v2!r} / {scope_v2!r}")
    return f"{V2_EN_OPENER[type_v2]} {V2_EN_SCOPE[scope_v2]}"


def scope_v2_to_sc_eval(scope_v2: str) -> str | None:
    """Map scope_v2 to binary sc-F1 gold (full/partial) or None if not applicable."""
    return {
        SCOPE_V2_COMPREHENSIVE: "full",
        SCOPE_V2_LOCALIZED: "partial",
        SCOPE_V2_PHASE1: None,
        SCOPE_V2_SURGICAL: None,
    }.get(scope_v2)


def relabel_scope_type_v2(rec: dict[str, Any]) -> dict[str, Any]:
    """Return v2 labels + audit metadata (chart-primary rules)."""
    raw = _raw_text(rec)
    tags = _tags(rec)
    st = (rec.get("surgery_type") or "").strip()
    legacy_fo = (rec.get("full_ortho") or "").strip()

    # 1) Surgical
    if st == "Orthodontic and Orthognathic" or legacy_fo == "Surgical":
        return _pack(
            SCOPE_V2_SURGICAL, TYPE_V2_SURGICAL, "surgical_chart", tags, legacy_fo, rec,
        )

    has_growth = "growth_raw" in tags
    has_two_phase = "two_phase_raw" in tags
    has_localized = ("localized_raw" in tags) or ("localized_appliance" in tags)
    has_full_comp = bool(_FULL_COMP_KO.search(raw))

    # 2) Growth / interceptive phase
    if has_growth:
        if has_two_phase:
            scope = SCOPE_V2_COMPREHENSIVE
            if has_localized and not has_full_comp and re.search(
                r"2차.*(부분|전치부|국소)|전체.*아님", raw, re.I
            ):
                scope = SCOPE_V2_LOCALIZED
            return _pack(
                scope, TYPE_V2_GROWTH_PLUS, "chart_growth_two_phase", tags, legacy_fo, rec,
            )
        return _pack(
            SCOPE_V2_PHASE1, TYPE_V2_GROWTH, "chart_growth_only", tags, legacy_fo, rec,
        )

    # 3) Comprehensive orthodontics only (no growth chart)
    if has_localized and not (has_full_comp and not re.search(r"부분", raw, re.I)):
        # Partial step or localized plan (incl. "1. 부분 … 2. 전체")
        return _pack(
            SCOPE_V2_LOCALIZED, TYPE_V2_COMPREHENSIVE,
            "chart_localized_comprehensive", tags, legacy_fo, rec,
        )

    return _pack(
        SCOPE_V2_COMPREHENSIVE, TYPE_V2_COMPREHENSIVE,
        "chart_comprehensive_default", tags, legacy_fo, rec,
    )


def _pack(
    scope_v2: str,
    type_v2: str,
    rule: str,
    tags: list[str],
    legacy_fo: str,
    rec: dict[str, Any],
    review: bool = False,
) -> dict[str, Any]:
    assert (type_v2, scope_v2) in VALID_TYPE_SCOPE, (type_v2, scope_v2)
    fo_v2 = FULL_ORTHO_V2_MAP[scope_v2]
    legacy_type = _legacy_type(rec)
    changed = (
        legacy_fo != fo_v2
        or legacy_type != type_v2
        or (legacy_fo == "Partial" and scope_v2 in (SCOPE_V2_LOCALIZED, SCOPE_V2_PHASE1))
        or (legacy_fo == "Full" and scope_v2 == SCOPE_V2_PHASE1)
    )
    return {
        "scope_v2": scope_v2,
        "treatment_type_v2": type_v2,
        "full_ortho_v2": fo_v2,
        "relabel_rule": rule,
        "relabel_tags": tags,
        "relabel_review": review,
        "relabel_changed": changed,
        "legacy_full_ortho": legacy_fo,
        "legacy_treatment_type": legacy_type,
    }


def _legacy_type(rec: dict[str, Any]) -> str:
    from ..utils.structured_eval import treatment_type_from_record
    return treatment_type_from_record(rec)
