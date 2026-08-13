"""Candidate scoring / selection for inference-time prompt optimization."""

from __future__ import annotations

import math
import re

from ..data.labels import SCOPE
from ..utils.metrics import parse_plan

_SCOPE_INV = {v: k for k, v in SCOPE.items()}

_SCOPE_CLAUSE = re.compile(
    r"with\s+(?:full|partial|surgical)(?:\s+orthodontic)?\s+scope",
    re.IGNORECASE,
)
_EXTRACTION_SPECIFIC = re.compile(
    r"with\s+the\s+extraction\s+of\s+teeth\s+[^,.]+",
    re.IGNORECASE,
)
_EXTRACTION_WITH = re.compile(r"\bwith\s+extraction\b", re.IGNORECASE)
_EXTRACTION_WITHOUT = re.compile(
    r"\b(?:without\s+extraction|non-extraction)\b",
    re.IGNORECASE,
)
_DURATION_CLAUSE = re.compile(
    r"(with\s+a\s+treatment\s+duration\s+of\s*)\d+(\s*months?)",
    re.IGNORECASE,
)
_DURATION_FALLBACK = re.compile(r"(\d+)\s*(months?)", re.IGNORECASE)

_SCOPE_PHRASE = {
    "full": "with full orthodontic scope",
    "partial": "with partial orthodontic scope",
    "surgical": "with surgical orthodontic scope",
}


def score_plan_vs_head(plan: str, aux: dict) -> float:
    """Higher = better agreement between parsed plan text and decision-head predictions."""
    pp = parse_plan(plan)
    score = 0.0
    if pp["scope"] is not None:
        score += 0.25
    if pp["extraction"] is not None:
        score += 0.25
    if pp["duration"] is not None:
        score += 0.1

    logits = aux.get("attr_logits")
    if not logits:
        return score

    hs = int(logits["scope"].argmax(dim=-1).item())
    he = int(logits["extraction"].argmax(dim=-1).item())
    ps = SCOPE.get(pp["scope"]) if pp["scope"] else None
    pe = pp["extraction"]

    conf = aux.get("attr_conf") or {}
    if ps is not None and ps == hs:
        c = float(conf["scope"][0]) if "scope" in conf else 0.0
        score += 2.0 + c
    if pe is not None and pe == he:
        c = float(conf["extraction"][0]) if "extraction" in conf else 0.0
        score += 2.0 + c
    return score


def rank_candidates(candidates: list[str], aux: dict) -> list[str]:
    """Model rerank order (highest head-agreement first)."""
    if not candidates:
        return []
    return sorted(candidates, key=lambda p: score_plan_vs_head(p, aux), reverse=True)


def select_plan(candidates: list[str], aux: dict, mode: str = "greedy") -> str:
    if not candidates:
        return ""
    if mode not in ("head_rerank", "parse_rerank") or len(candidates) == 1:
        return candidates[0]
    return rank_candidates(candidates, aux)[0]


def _head_scope_label(aux: dict) -> str | None:
    logits = (aux or {}).get("attr_logits") or {}
    if logits.get("scope") is None:
        return None
    return _SCOPE_INV.get(int(logits["scope"].argmax(dim=-1).item()))


def _head_extraction_label(aux: dict) -> int | None:
    """0 = without, 1 = with. None if missing."""
    logits = (aux or {}).get("attr_logits") or {}
    if logits.get("extraction") is None:
        return None
    return int(logits["extraction"].argmax(dim=-1).item())


def _head_duration_months(aux: dict) -> int | None:
    dur = (aux or {}).get("dur_months")
    if dur is None:
        return None
    try:
        val = float(dur.item() if hasattr(dur, "item") else dur)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val) or val <= 0:
        return None
    return max(1, int(round(val)))


def apply_head_to_plan(plan: str, aux: dict) -> str:
    """Hard-write decision-head scope / extraction / duration into the plan sentence.

    Inference-only text edit — does not change model weights or training. Soft prefix
    conditioning can still diverge from logits; this forces the visible sentence to match
    the structured heads (data → attributes → text).
    """
    if not plan or not aux:
        return plan
    text = plan.strip()

    scope = _head_scope_label(aux)
    if scope and scope in _SCOPE_PHRASE:
        phrase = _SCOPE_PHRASE[scope]
        if _SCOPE_CLAUSE.search(text):
            text = _SCOPE_CLAUSE.sub(phrase, text, count=1)
        else:
            text = re.sub(
                r"^(.*?(?:treatment|combined treatment),)\s*",
                rf"\1 {phrase}, ",
                text,
                count=1,
                flags=re.IGNORECASE,
            )

    extr = _head_extraction_label(aux)
    if extr == 0:
        if _EXTRACTION_SPECIFIC.search(text):
            text = _EXTRACTION_SPECIFIC.sub("without extraction", text, count=1)
        elif _EXTRACTION_WITH.search(text):
            text = _EXTRACTION_WITH.sub("without extraction", text, count=1)
        elif not _EXTRACTION_WITHOUT.search(text):
            text = re.sub(
                r"(orthodontic scope),",
                r"\1, without extraction,",
                text,
                count=1,
                flags=re.IGNORECASE,
            )
    elif extr == 1:
        if _EXTRACTION_WITHOUT.search(text):
            text = _EXTRACTION_WITHOUT.sub("with extraction", text, count=1)
        elif not (_EXTRACTION_SPECIFIC.search(text) or _EXTRACTION_WITH.search(text)):
            text = re.sub(
                r"(orthodontic scope),",
                r"\1, with extraction,",
                text,
                count=1,
                flags=re.IGNORECASE,
            )

    months = _head_duration_months(aux)
    if months is not None:
        if _DURATION_CLAUSE.search(text):
            text = _DURATION_CLAUSE.sub(rf"\g<1>{months}\2", text, count=1)
        elif _DURATION_FALLBACK.search(text):
            text = _DURATION_FALLBACK.sub(rf"{months} \2", text, count=1)
        else:
            text = text.rstrip(".") + f", with a treatment duration of {months} months."

    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,+", ", ", text)
    return text.strip()
