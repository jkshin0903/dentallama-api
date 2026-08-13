"""Parse pseudonymization_after AnalysisChart.csv into canonical measurements_results.

Each CSV row: Measure Name, Mean, S.D, normal
  - Mean / S.D  = norm reference (ignored for measurements_results)
  - normal      = patient value (trailing * significance markers stripped)

CSV labels vary by export; aliases map to cephalometry.MEASUREMENT_KEYS_ALL.
Standard labels (priority 0) win over short-form (priority 1).
"""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path

from ..utils.cephalometry import MEASUREMENT_KEYS_ALL

_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# (csv Measure Name, canonical key, priority) — lower priority wins on conflict
MEASURE_ALIASES: list[tuple[str, str, int]] = [
    # --- core 13 ---
    ("SNA (dg)", "SNA", 0),
    ("SNB (dg)", "SNB", 0),
    ("ANB (dg)", "ANB", 0),
    ("APDI (dg)", "APDI", 0),
    ("APDI", "APDI", 1),
    ("FMA (dg)", "FMA", 0),
    ("FMA", "FMA", 1),
    ("Combination Factor (dg)", "Combination Factor(CF)", 0),
    ("Combination factor", "Combination Factor(CF)", 1),
    ("Interincisal Angle (dg)", "IIA", 0),
    ("Interincisal angle", "IIA", 1),
    ("Mx1 to SN (dg)", "Mx1 to SN", 0),
    ("U1 to SN", "Mx1 to SN", 1),
    ("IMPA (dg)", "MP to Mn1", 0),
    ("IMPA", "MP to Mn1", 1),
    ("Upper Lip E-Plane (mm)", "Eline to U lip", 0),
    ("Upper lip to E-plane", "Eline to U lip", 1),
    ("Lower Lip E-Plane (mm)", "Eline to L lip", 0),
    ("Lower lip to E-plane", "Eline to L lip", 1),
    ("Incisor Overjet (mm)", "Overjet", 0),
    ("Overjet", "Overjet", 1),
    ("Incisor Overbite (mm)", "Overbite", 0),
    ("Overbite", "Overbite", 1),
    # --- extended (high coverage in AnalysisChart.csv) ---
    ("A to N Perpendicular (mm)", "A to N Perpendicular", 0),
    ("Pog to N Perpendicular (mm)", "Pog to N Perpendicular", 0),
    ("Wits Appraisal (mm)", "Wits Appraisal", 0),
    ("Wits appraisal", "Wits Appraisal", 1),
    ("ODI (dg)", "ODI", 0),
    ("ODI", "ODI", 1),
    ("Mx1 to NA (dg)", "Mx1 to NA (deg)", 0),
    ("U1 to NA(deg)", "Mx1 to NA (deg)", 1),
    ("Mx1 to NA (mm)", "Mx1 to NA (mm)", 0),
    ("U1 to NA(mm)", "Mx1 to NA (mm)", 1),
    ("Mx1 to A-Pog (mm)", "Mx1 to A-Pog", 0),
    ("U1 to A-Pog(mm)", "Mx1 to A-Pog", 1),
    ("Md1 to NB (dg)", "Md1 to NB (deg)", 0),
    ("L1 to NB(deg)", "Md1 to NB (deg)", 1),
    ("Md1 to NB (mm)", "Md1 to NB (mm)", 0),
    ("L1 to NB(mm)", "Md1 to NB (mm)", 1),
    ("Md1 to A-Pog (mm)", "Md1 to A-Pog", 0),
    ("L1 to A-Pog(mm)", "Md1 to A-Pog", 1),
    ("Body to Ant. Cranial Base Ratio", "Body to Ant. Cranial Base Ratio", 0),
    ("Body to Ant. cranial base ratio", "Body to Ant. Cranial Base Ratio", 1),
    ("Facial Height Ratio", "Facial Height Ratio", 0),
    ("Facial height ratio(PFH/AFH)", "Facial Height Ratio", 1),
    ("Occ Plane (dg)", "Occ Plane", 0),
    ("Occlusal plane to SN angle", "Occ Plane", 1),
    ("STms-Mx1 (mm)", "STms-Mx1", 0),
    ("Stms-Mx1 (mm)", "STms-Mx1", 1),
    ("SN-GoGn (dg)", "SN-GoGn", 0),
    ("SN-GoMe (dg)", "SN-GoMe", 0),
    ("SN-GoMe", "SN-GoMe", 1),
    ("Saddle Angle (dg)", "Saddle Angle", 0),
    ("Saddle angle", "Saddle Angle", 1),
    ("Saddle Angle (dg) Da", "Saddle Angle", 2),
    ("Articular Angle (dg)", "Articular Angle", 0),
    ("Articular angle", "Articular Angle", 1),
    ("Articular Angle (dg) Da", "Articular Angle", 2),
    ("Gonial angle (dg)", "Gonial angle", 0),
    ("Gonial angle", "Gonial angle", 1),
    ("Gonial angle (dg) Da", "Gonial angle", 2),
    ("Bjork Sum (dg)", "Bjork Sum", 0),
    ("Bjork sum", "Bjork Sum", 1),
    ("SUM (dg) da", "Bjork Sum", 2),
]

_ALIAS_LOOKUP: dict[str, tuple[str, int]] = {
    csv_name: (canonical, priority) for csv_name, canonical, priority in MEASURE_ALIASES
}


def _to_float(raw) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not (isinstance(raw, float) and math.isnan(raw)):
        return float(raw)
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "insufficient", "-"}:
        return None
    s = re.sub(r"\*+$", "", s).strip()
    m = _FLOAT_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def analysis_chart_path_from_ceph(ceph_image: str) -> Path:
    return Path(ceph_image).parent / "AnalysisChart.csv"


def parse_analysis_chart_csv(csv_path: str | Path) -> dict[str, float | str]:
    """Return measurements_results: floats when parsed, '' when missing/unparseable."""
    path = Path(csv_path)
    out: dict[str, float | str] = {k: "" for k in MEASUREMENT_KEYS_ALL}
    if not path.is_file():
        return out

    best_priority: dict[str, int] = {k: 999 for k in MEASUREMENT_KEYS_ALL}
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = csv.reader(f)
            header = next(rows, None)
            if not header or "Measure Name" not in header[0]:
                return out
            for row in rows:
                if len(row) < 4:
                    continue
                csv_name = row[0].strip()
                alias = _ALIAS_LOOKUP.get(csv_name)
                if not alias:
                    continue
                canonical, priority = alias
                if priority > best_priority[canonical]:
                    continue
                val = _to_float(row[3])
                if val is None:
                    if priority <= best_priority[canonical]:
                        out[canonical] = ""
                        best_priority[canonical] = priority
                    continue
                out[canonical] = val
                best_priority[canonical] = priority
    except OSError:
        return out
    return out


def parse_from_record(rec: dict) -> dict[str, float | str]:
    ceph = rec.get("ceph_image") or ""
    return parse_analysis_chart_csv(analysis_chart_path_from_ceph(ceph))


def parse_analysis_chart_csv_text(csv_text: str) -> dict[str, float | str]:
    """Parse in-memory AnalysisChart.csv body (same format as parse_analysis_chart_csv)."""
    import io

    out: dict[str, float | str] = {k: "" for k in MEASUREMENT_KEYS_ALL}
    if not (csv_text or "").strip():
        return out

    best_priority: dict[str, int] = {k: 999 for k in MEASUREMENT_KEYS_ALL}
    try:
        rows = csv.reader(io.StringIO(csv_text.strip()))
        header = next(rows, None)
        if not header or "Measure Name" not in header[0]:
            return out
        for row in rows:
            if len(row) < 4:
                continue
            csv_name = row[0].strip()
            alias = _ALIAS_LOOKUP.get(csv_name)
            if not alias:
                continue
            canonical, priority = alias
            if priority > best_priority[canonical]:
                continue
            val = _to_float(row[3])
            if val is None:
                if priority <= best_priority[canonical]:
                    out[canonical] = ""
                    best_priority[canonical] = priority
                continue
            out[canonical] = val
            best_priority[canonical] = priority
    except (OSError, csv.Error):
        return out
    return out


def merge_measurements(*sources: dict | None) -> dict[str, float | str]:
    """Merge measurement dicts; later sources override earlier ones."""
    out: dict[str, float | str] = {k: "" for k in MEASUREMENT_KEYS_ALL}
    for src in sources:
        if not src:
            continue
        for key in MEASUREMENT_KEYS_ALL:
            val = src.get(key)
            if val is None or val == "":
                continue
            if isinstance(val, (int, float)) and not (isinstance(val, float) and math.isnan(val)):
                out[key] = float(val)
            else:
                parsed = _to_float(val)
                if parsed is not None:
                    out[key] = parsed
    return out


def parse_analysis_chart_excel_bytes(data: bytes) -> dict[str, float | str]:
    """Parse first worksheet of an AnalysisChart Excel export."""
    try:
        import openpyxl
    except ImportError as e:
        raise RuntimeError("Excel input requires openpyxl (pip install openpyxl)") from e

    from io import BytesIO

    wb = openpyxl.load_workbook(BytesIO(data), read_only=True, data_only=True)
    ws = wb.active
    rows = [[cell.value for cell in row] for row in ws.iter_rows()]
    if not rows:
        return {k: "" for k in MEASUREMENT_KEYS_ALL}

    lines = []
    for row in rows:
        cells = ["" if c is None else str(c) for c in row]
        while cells and not cells[-1].strip():
            cells.pop()
        if cells:
            lines.append(",".join(cells))
    return parse_analysis_chart_csv_text("\n".join(lines))
