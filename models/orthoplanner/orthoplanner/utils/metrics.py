"""Plan-text helpers used by the inference API (eval/NLG suites trimmed for demo)."""
from __future__ import annotations

import re


def parse_plan(text: str) -> dict:
    t = text.lower()
    scope = "surgical" if "surg" in t else ("partial" if "partial" in t else
                                            ("full" if "full" in t else None))
    extraction = 1 if (("with extraction" in t) or ("/" in t and "extraction" in t)) else \
        (0 if "non-extraction" in t or "without extraction" in t else None)
    m = re.search(r"(\d+)\s*month", t)
    dur = float(m.group(1)) if m else None
    return {"scope": scope, "extraction": extraction, "duration": dur}
