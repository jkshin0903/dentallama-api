from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class TreatmentPlanRequest(BaseModel):
    ceph: str = Field(..., description="Cephalometric image as base64 (data-URL prefix allowed)")
    analysischart: Optional[dict[str, Any]] = Field(
        None, description="AnalysisChart measurements as a JSON dict"
    )
    analysiscsv: Optional[str] = Field(None, description="AnalysisChart.csv body as plain text")
    analysisexcel_b64: Optional[str] = Field(
        None, description="AnalysisChart Excel file as base64 (.xlsx/.xls)"
    )
    gender: Optional[str] = Field(None, description="Male/Female or M/F")
    age: Optional[str] = Field(None, description='Age string e.g. "14Y 3M" or decimal years')
    diagnosis: Optional[str] = Field("", description="Clinical diagnosis text (KO or EN)")
    model: Optional[str] = Field(
        None,
        description="Model alias. Omit for server default (cephstruct_rule).",
    )
    fold: Optional[int] = Field(None, ge=0, le=5, description="Cross-validation fold")
    prompt_mode: Optional[Literal["legacy", "promptpp", "promptpp_compact"]] = None
    gen_select: Optional[Literal["greedy", "head_rerank"]] = None
    hard_reflect_head: Optional[bool] = Field(
        None,
        description=(
            "If true (default), overwrite plan sentence slots from decision-head predictions. "
            "Set false to keep raw LLM wording."
        ),
    )


class TreatmentPlanResponse(BaseModel):
    treatment_plan: str
    parsed: dict[str, Any]
    decision_head: Optional[dict[str, Any]] = None
    hedge: Optional[float] = None
    duration_months_est: Optional[float] = None
    measurements_used: dict[str, Any]
    model: str
    model_fold: int
    model_checkpoint: str
    prompt_mode: str
    gen_select: str
