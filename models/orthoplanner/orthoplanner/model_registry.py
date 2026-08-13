"""Named model aliases -> checkpoint paths for the inference API."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    id: str
    description: str
    run_tag: str
    default_fold: int = 0
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    prompt_mode: str = "promptpp_compact"
    gen_select: str = "head_rerank"
    requires_measurements: bool = True


MODEL_SPECS: dict[str, ModelSpec] = {
    "cephstruct_rule": ModelSpec(
        id="cephstruct_rule",
        description="CephStruct-LMM v3.1 — measurement-primary + neuro-symbolic rule layer (default)",
        run_tag="orthoplanner_v31dd_v2",
        default_fold=0,
        prompt_mode="promptpp_compact",
        gen_select="head_rerank",
        requires_measurements=True,
    ),
    "cephstruct_v31": ModelSpec(
        id="cephstruct_v31",
        description="CephStruct-LMM v3.1 pilot — rule layer + diagnosis derivation + metadata",
        run_tag="orthoplanner_v31dd",
        default_fold=5,
        folds=(5,),
        prompt_mode="promptpp_compact",
        gen_select="head_rerank",
        requires_measurements=True,
    ),
    "cephstruct_v1": ModelSpec(
        id="cephstruct_v1",
        description="GUSR v1 — full geometry+text fusion without rule layer (treatment-plan model, not a classifier)",
        run_tag="orthoplanner_v1_v2",
        default_fold=0,
        prompt_mode="legacy",
        gen_select="greedy",
        requires_measurements=False,
    ),
    "cephstruct_qformer": ModelSpec(
        id="cephstruct_qformer",
        description="Q-Former-style fusion (no relation bias / no decision queries) + rule layer, v2 5-fold",
        run_tag="orthoplanner_qformer_v2",
        default_fold=0,
        prompt_mode="legacy",
        gen_select="greedy",
        requires_measurements=True,
    ),
}

DEFAULT_MODEL_ID = "cephstruct_rule"


def list_models(repo_root: str) -> list[dict]:
    out = []
    for spec in MODEL_SPECS.values():
        folds = []
        for f in spec.folds:
            ckpt = checkpoint_path(repo_root, spec.id, f)
            folds.append({"fold": f, "available": os.path.isdir(ckpt), "checkpoint": ckpt})
        out.append(
            {
                "id": spec.id,
                "description": spec.description,
                "default_fold": spec.default_fold,
                "folds": folds,
                "default_prompt_mode": spec.prompt_mode,
                "default_gen_select": spec.gen_select,
                "requires_measurements": spec.requires_measurements,
            }
        )
    return out


def resolve_model(model: str | None, fold: int | None = None) -> tuple[ModelSpec, int]:
    model_id = (model or DEFAULT_MODEL_ID).strip().lower()
    if model_id not in MODEL_SPECS:
        known = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unknown model '{model_id}'. Available: {known}")
    spec = MODEL_SPECS[model_id]
    f = spec.default_fold if fold is None else fold
    if f not in spec.folds:
        allowed = ", ".join(str(x) for x in spec.folds)
        raise ValueError(f"Model '{model_id}' does not support fold {f}. Allowed folds: {allowed}")
    return spec, f


def checkpoint_path(repo_root: str, model: str | None, fold: int | None = None) -> str:
    spec, f = resolve_model(model, fold)
    return os.path.join(repo_root, "OrthoPlanner/runs", f"{spec.run_tag}_fold{f}", "best")
