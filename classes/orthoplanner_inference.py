"""In-process OrthoPlanner inference (model only; no demo web server)."""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from classes.orthoplanner_schema import TreatmentPlanRequest, TreatmentPlanResponse

_PACKAGE_READY = False


def _ensure_package_on_path(ortho_root: Path) -> None:
    global _PACKAGE_READY
    pkg_root = str(ortho_root)
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    _PACKAGE_READY = True


def _resolve(ortho_root: Path, path: str) -> str:
    if os.path.isabs(path):
        return path
    return str(ortho_root / path)


def _decode_base64_image(b64: str) -> np.ndarray:
    import base64
    import io

    raw = b64.strip()
    if raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        data = base64.b64decode(raw, validate=False)
    except Exception as e:
        raise ValueError(f"Invalid base64 ceph image: {e}") from e
    try:
        return np.array(Image.open(io.BytesIO(data)).convert("L"))
    except Exception as e:
        raise ValueError(f"Cannot decode ceph image: {e}") from e


def _preprocess_ceph_image(b64: str, input_size: int = 768) -> torch.Tensor:
    from orthoplanner.stage1.landmark_dataset import resize_with_padding

    img = _decode_base64_image(b64)
    dummy = np.zeros((1, 2), dtype=np.float32)
    img, _ = resize_with_padding(img, dummy, input_size)
    img = img.astype(np.float32) / 255.0
    img = (img - img.mean()) / (img.std() + 1e-6)
    return torch.from_numpy(img).unsqueeze(0).float()


def _compose_diagnosis_text(diagnosis: str, measurements: dict, include_measurements: bool) -> str:
    from orthoplanner.utils.cephalometry import MEASUREMENT_KEYS

    diag = (diagnosis or "").strip()
    if not include_measurements:
        return diag
    parts = [
        f"{k} {measurements[k]:.1f}"
        for k in MEASUREMENT_KEYS
        if isinstance(measurements.get(k), (int, float))
    ]
    mstr = ", ".join(parts)
    if diag and mstr:
        return f"{diag} Measurements: {mstr}."
    if mstr:
        return f"Measurements: {mstr}."
    return diag


def _build_metadata(age, gender, age_years) -> torch.Tensor:
    from orthoplanner.clinical.growth_guard import parse_age_years, parse_gender_label
    from orthoplanner.data.labels import metadata_vec

    yrs = parse_age_years(age, age_years=age_years)
    g = parse_gender_label(gender) or ""
    rec = {"age_years": yrs if yrs is not None else 14.0, "gender": g}
    return torch.tensor([metadata_vec(rec)], dtype=torch.float32)


def _build_gt_measurements(measurements: dict, meas_fill) -> torch.Tensor:
    from orthoplanner.utils.cephalometry import gt_measurement_vector

    vec = gt_measurement_vector(measurements)
    t = torch.tensor([vec], dtype=torch.float32)
    if meas_fill is not None:
        fill = torch.as_tensor(meas_fill, dtype=torch.float32)
        t = torch.where(torch.isnan(t), fill.unsqueeze(0).expand_as(t), t)
    else:
        t = torch.nan_to_num(t)
    return t


def _head_summary(aux: dict) -> dict[str, Any] | None:
    if aux.get("attr_logits") is None:
        return None
    from orthoplanner.data.labels import SCOPE

    scope_inv = {v: k for k, v in SCOPE.items()}
    out: dict[str, Any] = {}
    for attr in ("scope", "extraction", "surgery"):
        logits = aux["attr_logits"].get(attr)
        conf = aux["attr_conf"].get(attr)
        if logits is None:
            continue
        idx = int(logits.argmax(-1).item())
        entry: dict[str, Any] = {
            "prediction": idx,
            "confidence": float(conf[0].item()) if conf is not None else None,
        }
        if attr == "scope":
            entry["label"] = scope_inv.get(idx)
        else:
            entry["label"] = "yes" if idx == 1 else "no"
        out[attr] = entry
    return out


def _parse_measurements(req: TreatmentPlanRequest) -> dict[str, float | str]:
    from orthoplanner.data.analysis_chart import (
        merge_measurements,
        parse_analysis_chart_csv_text,
        parse_analysis_chart_excel_bytes,
    )

    measurements: dict[str, float | str] = {}
    if req.analysiscsv:
        measurements = merge_measurements(measurements, parse_analysis_chart_csv_text(req.analysiscsv))
    if req.analysischart:
        measurements = merge_measurements(measurements, req.analysischart)
    if req.analysisexcel_b64:
        import base64

        raw = req.analysisexcel_b64.strip()
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[1]
        try:
            xbytes = base64.b64decode(raw, validate=False)
        except Exception as e:
            raise ValueError(f"Invalid analysisexcel_b64: {e}") from e
        measurements = merge_measurements(measurements, parse_analysis_chart_excel_bytes(xbytes))
    return measurements


class OrthoPlannerInference:
    """Loads one OrthoPlanner checkpoint and runs treatment-plan generation."""

    def __init__(
        self,
        ortho_root: Path,
        ckpt: str,
        config_path: str,
        llm_path: Optional[str] = None,
        cuda: int = 0,
        model_id: str = "cephstruct_rule",
        model_fold: int = 0,
    ):
        self.ortho_root = Path(ortho_root)
        self.ckpt = ckpt
        self.config_path = config_path
        self.llm_path = llm_path
        self.cuda = cuda
        self.model_id = model_id
        self.model_fold = model_fold
        self._lock = threading.RLock()
        self._state: dict[str, Any] | None = None

    @property
    def loaded(self) -> bool:
        return self._state is not None and self._state.get("model") is not None

    def load(self) -> None:
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            self._state = self._load_model()

    def _load_model(self) -> dict[str, Any]:
        _ensure_package_on_path(self.ortho_root)
        from orthoplanner.config import load_config
        from orthoplanner.stage2.model import OrthoPlanner
        from peft import PeftModel

        ckpt = self.ckpt
        if not os.path.isdir(ckpt):
            raise FileNotFoundError(f"OrthoPlanner checkpoint not found: {ckpt}")
        stats_path = os.path.join(ckpt, "stats.pt")
        if not os.path.isfile(stats_path):
            raise FileNotFoundError(f"Missing stats.pt in checkpoint: {ckpt}")

        cfg = load_config(self.config_path, cuda=self.cuda)
        device = torch.device(f"cuda:{cfg.cuda}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            from orthoplanner.utils.gpu_polite import apply_gpu_polite

            apply_gpu_polite(cfg.cuda)

        stats = torch.load(stats_path, map_location="cpu")
        cfg.dur_mean, cfg.dur_std = stats["dur_mean"], stats["dur_std"]
        if self.llm_path:
            cfg.llm_path = self.llm_path
        elif stats.get("llm_path"):
            cfg.llm_path = stats["llm_path"]
        if stats.get("ablation"):
            cfg.ablation = stats["ablation"]
        if "include_measurements" in stats:
            cfg.include_measurements = stats["include_measurements"]
        for k, v in (stats.get("gusr") or {}).items():
            setattr(cfg, k, v)

        cfg.ceph_ckpt = _resolve(self.ortho_root, cfg.ceph_ckpt)
        cfg.text_encoder_path = _resolve(self.ortho_root, cfg.text_encoder_path)
        cfg.llm_path = _resolve(self.ortho_root, cfg.llm_path)

        print(f"[OrthoPlanner] loading checkpoint={ckpt} llm={cfg.llm_path} device={device}", flush=True)
        model = OrthoPlanner(cfg).to(device)
        model.reasoner.load_state_dict(
            torch.load(os.path.join(ckpt, "reasoner.pt"), map_location=device)
        )
        model.llm = PeftModel.from_pretrained(model.llm.get_base_model(), ckpt).to(device)
        model.eval()
        print(f"[OrthoPlanner] ready on {device}", flush=True)

        return {
            "model": model,
            "cfg": cfg,
            "device": device,
            "stats": stats,
            "ckpt": ckpt,
            "meas_fill": stats.get("meas_mean"),
        }

    def predict(self, req: TreatmentPlanRequest) -> TreatmentPlanResponse:
        if not self.loaded:
            self.load()
        assert self._state is not None

        from orthoplanner.clinical.growth_guard import parse_age_years
        from orthoplanner.model_registry import resolve_model
        from orthoplanner.prompts.treatment_plan import system_prompt as resolve_system_prompt
        from orthoplanner.utils.metrics import parse_plan

        try:
            spec, fold = resolve_model(req.model or self.model_id, req.fold if req.fold is not None else self.model_fold)
        except ValueError as e:
            raise ValueError(str(e)) from e
        if spec.id != self.model_id or fold != self.model_fold:
            raise ValueError(
                f"This server has loaded {self.model_id} fold {self.model_fold}. "
                f"Requested {spec.id} fold {fold} is not loaded."
            )

        measurements = _parse_measurements(req)
        n_numeric = sum(isinstance(v, (int, float)) for v in measurements.values())
        if spec.requires_measurements and n_numeric == 0:
            raise ValueError(
                f"Model '{spec.id}' requires AnalysisChart measurements "
                "(analysischart, analysiscsv, or analysisexcel_b64)"
            )

        model = self._state["model"]
        cfg = self._state["cfg"]
        device = self._state["device"]
        yrs = parse_age_years(req.age)
        prompt_mode = req.prompt_mode or spec.prompt_mode
        gen_select = req.gen_select or spec.gen_select

        image = _preprocess_ceph_image(req.ceph, cfg.input_size).unsqueeze(0).to(device)
        diag_text = _compose_diagnosis_text(req.diagnosis or "", measurements, cfg.include_measurements)
        gt_meas = _build_gt_measurements(measurements, self._state.get("meas_fill")).to(device)
        metadata = None
        if getattr(cfg, "use_metadata", False):
            metadata = _build_metadata(req.age, req.gender, yrs).to(device)

        sys_prompt = resolve_system_prompt(prompt_mode)
        with self._lock:
            plan, aux = model.generate(
                image,
                diag_text,
                sys_prompt,
                metadata=metadata,
                gt_measurements=gt_meas if n_numeric > 0 else None,
                prompt_mode=prompt_mode,
                gen_select=gen_select,
                hard_reflect_head=req.hard_reflect_head,
                age=req.age,
                gender=req.gender,
                age_years=yrs,
            )

        parsed = parse_plan(plan)
        dur_est = None
        if aux.get("dur_months") is not None:
            dur_est = float(aux["dur_months"].item())
        meas_out = {k: v for k, v in measurements.items() if v != "" and v is not None}

        return TreatmentPlanResponse(
            treatment_plan=plan,
            parsed=parsed,
            decision_head=_head_summary(aux),
            hedge=float(aux["hedge"][0]) if aux.get("hedge") is not None else None,
            duration_months_est=dur_est,
            measurements_used=meas_out,
            model=spec.id,
            model_fold=fold,
            model_checkpoint=self._state["ckpt"],
            prompt_mode=prompt_mode,
            gen_select=gen_select,
        )


def build_from_env(base_dir: Path) -> OrthoPlannerInference:
    ortho_root = Path(os.getenv("ORTHOPLANNER_ROOT") or (base_dir / "models" / "orthoplanner"))
    ckpt = _resolve(ortho_root, os.getenv("ORTHOPLANNER_CKPT") or "fold0/best")
    config_path = _resolve(ortho_root, os.getenv("ORTHOPLANNER_CONFIG") or "config.yaml")
    llm_path = _resolve(
        ortho_root, os.getenv("ORTHOPLANNER_LLM_PATH") or "Llama_3.1_8B-Instruct"
    )
    cuda = int(os.getenv("ORTHOPLANNER_CUDA", "0"))
    return OrthoPlannerInference(
        ortho_root=ortho_root,
        ckpt=ckpt,
        config_path=config_path,
        llm_path=llm_path,
        cuda=cuda,
    )
