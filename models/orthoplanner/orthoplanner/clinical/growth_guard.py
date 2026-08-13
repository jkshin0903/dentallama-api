"""524 growth-modification guard for plan generation (inference-time).

Blocks growth-phase openers and growth-only devices for adults (age >= 19).
ABP is kept when the plan looks like comprehensive/sequential ortho (DBS, TAD, etc.)
or when Korean raw lacks growth/Activator cues — see build_abp_review_sheet logic.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

ADULT_AGE_THRESHOLD = 19

_GROWTH_OPENER = re.compile(
    r"Growth\s+modification(?:\s+followed\s+by\s+comprehensive\s+orthodontic\s+treatment)?\s*,",
    re.I,
)

# Always strip for adults (524 1차 정형/기능성; headgear already removed from adult gold).
_ADULT_FORBIDDEN = [
    r"headgear",
    r"Headgear",
    r"facemask",
    r"Facemask",
    r"chin\s+cup",
    r"Twin-block",
    r"Activator",
    r"Bionator",
    r"Schwarz",
    r"quad\s*helix",
    r"Quad-helix",
    r"tongue\s+crib",
    r"inclined\s+plane",
    r"MARPE",
    r"\bRPE\b",
    r"expansion\s+screw",
    r"three-way\s+expansion",
    r"J-?hook",
    r"high-pull",
    r"Teuscher",
]
_FORBIDDEN_RE = re.compile("|".join(f"(?:{p})" for p in _ADULT_FORBIDDEN), re.I)

_COMP_EN = re.compile(
    r"\bDBS\b|\bBK\b|screw|temporary\s+anchorage|TAD|I-?arch|MEAW|braces|bonding|"
    r"with\s+extraction|orthognathic|surgical\s+scope",
    re.I,
)
_COMP_KO = re.compile(r"전체\s*교정|발치|비발치|술전|술후|SSRO|악교정", re.I)
_GROWTH_KO = re.compile(
    r"악정형\s*치료|약정형|Activator|Bionator|headgear|헤드\s*기어|헤드기어|"
    r"facemask|페이스\s*마스크|Twin-?block|성장\s*교정",
    re.I,
)
_GROWTH_EN = re.compile(r"growth\s+modification|Activator|Bionator", re.I)
_ABP = re.compile(r"\bABP\b", re.I)


def parse_age_years(age=None, *, age_years=None) -> Optional[float]:
    if age_years is not None:
        try:
            return float(age_years)
        except (TypeError, ValueError):
            pass
    if age is None:
        return None
    s = str(age).strip()
    m = re.search(r"(\d+(?:\.\d+)?)\s*Y", s, re.I)
    if m:
        return float(m.group(1))
    try:
        return float(s)
    except ValueError:
        return None


def parse_gender_label(gender=None) -> Optional[str]:
    if not gender:
        return None
    g = str(gender).strip().lower()
    if g.startswith("f") or g in ("여", "여성", "female"):
        return "female"
    if g.startswith("m") or g in ("남", "남성", "male"):
        return "male"
    return gender.strip()


def is_adult(age_years: Optional[float]) -> bool:
    return age_years is not None and age_years >= ADULT_AGE_THRESHOLD


def patient_context_line(
    age=None, gender=None, *, age_years=None,
) -> str:
    yrs = parse_age_years(age, age_years=age_years)
    g = parse_gender_label(gender)
    parts: List[str] = []
    if yrs is not None:
        parts.append(f"{int(yrs) if yrs == int(yrs) else yrs} years old")
    if g:
        parts.append(g)
    if not parts:
        return ""
    return "Patient: " + ", ".join(parts) + "."


def growth_guard_prompt_rules(age_years: Optional[float]) -> str:
    if not is_adult(age_years):
        return ""
    return (
        "\n\nAdult patient rule (age >= 19, MANDATORY):\n"
        "- Use opener 'Orthodontic treatment,' only — NOT 'Growth modification …'.\n"
        "- Do NOT prescribe growth-phase devices: headgear, facemask, chin cup, Twin-block, "
        "Activator, Bionator, RPE, Schwarz, quad helix, tongue crib, J-hook.\n"
        "- ABP may appear only as comprehensive/sequential mechanics (e.g. with DBS or TAD), "
        "not as a growth-modification Activator.\n"
    )


def augment_system_prompt(system_prompt: str, age_years: Optional[float]) -> str:
    rules = growth_guard_prompt_rules(age_years)
    return system_prompt + rules if rules else system_prompt


def abp_is_growth_context(
    plan: str,
    *,
    korean_raw: Optional[str] = None,
) -> bool:
    """True when ABP should be treated as growth/Activator (strip for adults)."""
    if _GROWTH_EN.search(plan):
        return True
    if korean_raw and _GROWTH_KO.search(korean_raw):
        return True
    if _ABP.search(plan) and not _COMP_EN.search(plan):
        if korean_raw and _COMP_KO.search(korean_raw):
            return False
        return True
    return False


def _cleanup_plan_clauses(text: str) -> str:
    out = text
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r";\s*;", ";", out)
    out = re.sub(r",\s*,", ",", out)
    out = re.sub(r"\busing\s*;", "using ", out, flags=re.I)
    out = re.sub(r"\busing\s*,", "using ", out, flags=re.I)
    out = re.sub(r";\s*,", ",", out)
    out = re.sub(r",\s*;", ",", out)
    out = re.sub(r",\s+with\s*,", ",", out, flags=re.I)
    out = re.sub(r"\bwith\s*,", "with ", out, flags=re.I)
    return out.strip(" ,;.")


def apply_growth_guard(
    plan: str,
    age_years: Optional[float],
    *,
    korean_raw: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """Return (sanitized_plan, list of applied actions). No-op for minors or missing age."""
    if not plan or not is_adult(age_years):
        return plan, []

    actions: List[str] = []
    out = plan

    if _GROWTH_OPENER.search(out):
        out = _GROWTH_OPENER.sub("Orthodontic treatment,", out)
        actions.append("growth_opener->orthodontic")

    before = out
    out = _FORBIDDEN_RE.sub("", out)
    if out != before:
        actions.append("removed_growth_devices")

    if _ABP.search(out) and abp_is_growth_context(out, korean_raw=korean_raw):
        out = _ABP.sub("", out)
        actions.append("removed_ABP_growth_context")

    out = _cleanup_plan_clauses(out)
    if re.search(r"\busing\s*[,.]?\s*with\b", out, re.I):
        out = re.sub(r"\busing\s*[,.]?\s*with\b", "with", out, flags=re.I)
        actions.append("fixed_using_clause")

    return out, actions
