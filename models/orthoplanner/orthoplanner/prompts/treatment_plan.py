"""Llama-oriented treatment-plan prompts (legacy + Prompt++ variants for inference sweeps)."""

from __future__ import annotations

from typing import Literal

PromptMode = Literal["legacy", "promptpp", "promptpp_compact"]

PROMPT_MODES: tuple[str, ...] = ("legacy", "promptpp", "promptpp_compact")

_SENTENCE_TEMPLATE = (
    "Orthodontic treatment, with {full|partial} orthodontic scope, "
    "{with/without extraction clause}, "
    "{optional appliances/procedures}, "
    "with a treatment duration of {N} months."
)

_ANTI_COLLAPSE = (
    "Anti-collapse guidance (IMPORTANT):\n"
    "- Avoid trivial defaults, but do not over-correct.\n"
    "- Scope: choose 'partial orthodontic scope' ONLY when the diagnosis explicitly indicates a limited region.\n"
    "  Common Partial patterns: anterior-only (전치/전치부, 3전치/4전치), localized crossbite, arch narrowing (악궁 협소).\n"
    "  Otherwise choose 'full orthodontic scope'.\n"
    "- Extraction: choose 'with extraction' when crowding/protrusion cues appear; "
    "'without extraction' for spacing/diastema; default 'without extraction' if unclear.\n"
)


def system_prompt(mode: PromptMode = "legacy") -> str:
    if mode == "legacy":
        from ..config import SYSTEM_PROMPT

        return SYSTEM_PROMPT
    if mode == "promptpp_compact":
        return (
            "You are an expert orthodontist. Based on the clinical diagnosis and cephalometric imaging, "
            "write ONE English treatment-plan sentence.\n\n"
            f"Template: {_SENTENCE_TEMPLATE}\n"
            "Scope: full | partial | surgical orthodontic scope. "
            "Extraction: without | with | with the extraction of teeth XX/YY. "
            "End with 'with a treatment duration of N months.' "
            "Output ONLY the sentence.\n"
            "- Use partial scope only for localized anterior/crossbite/narrow-arch cases; else full.\n"
            "- Default to without extraction unless retraction cues are clear.\n"
        )
    if mode == "promptpp":
        return (
            "You are an expert orthodontist. Based on the clinical diagnosis and cephalometric imaging features, "
            "write ONE English treatment-plan sentence for the chart.\n\n"
            "Use EXACTLY this sentence shape (one sentence, no bullet list):\n"
            f"  {_SENTENCE_TEMPLATE}\n\n"
            "Rules:\n"
            "- Treatment type opener (choose ONE):\n"
            "  • 'Orthodontic treatment,' — comprehensive / 본교정\n"
            "  • 'Growth modification orthodontic treatment,' — 악정형\n"
            "  • 'Growth modification followed by comprehensive orthodontic treatment,' — 악정형+본교정\n"
            "  • 'Orthodontic and orthognathic combined treatment,' — 수술교정\n"
            "- Scope: 'full orthodontic scope' OR 'partial orthodontic scope' OR 'surgical scope'.\n"
            "- Extraction: 'without extraction' OR 'with extraction' OR "
            "'with the extraction of teeth XX/YY' (FDI numbering when specific).\n"
            "- Optional middle clause for devices/procedures; omit if none.\n"
            "- End with 'with a treatment duration of N months.' (integer N).\n"
            "- Output ONLY the final sentence. No reasoning, headings, or markdown.\n\n"
            + _ANTI_COLLAPSE
        )
    raise ValueError(f"Unknown prompt mode: {mode}")


def user_message(
    diagnosis: str,
    mode: PromptMode = "legacy",
    *,
    age=None,
    gender=None,
    age_years=None,
) -> str:
    from ..clinical.growth_guard import patient_context_line

    meta = patient_context_line(age, gender, age_years=age_years)
    meta_block = f"{meta}\n" if meta else ""
    if mode == "legacy":
        return f"{meta_block}Diagnosis: {diagnosis}\nPlan:"
    return (
        "Based on the clinical diagnosis and imaging features, determine scope, extraction, "
        "appliances/procedures, and duration.\n\n"
        f"{meta_block}Diagnosis: {diagnosis}\n\n"
        "Write the treatment-plan sentence for this case.\n"
        "Treatment plan:"
    )
