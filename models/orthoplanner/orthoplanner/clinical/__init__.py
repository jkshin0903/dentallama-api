"""Clinical rule helpers (524 growth guard, patient context)."""

from .growth_guard import (
    ADULT_AGE_THRESHOLD,
    apply_growth_guard,
    augment_system_prompt,
    is_adult,
    parse_age_years,
    patient_context_line,
)

__all__ = [
    "ADULT_AGE_THRESHOLD",
    "apply_growth_guard",
    "augment_system_prompt",
    "is_adult",
    "parse_age_years",
    "patient_context_line",
]
