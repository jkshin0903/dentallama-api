"""Typed configuration for OrthoPlanner Stage 2 (loadable from YAML)."""
from __future__ import annotations

from dataclasses import dataclass, field, fields


SYSTEM_PROMPT = (
    "You are a specialized orthodontist. Based on the clinical diagnosis and cephalometric imaging "
    "features, generate a single concise English orthodontic treatment plan."
)


@dataclass
class Config:
    # --- paths (relative to models/orthoplanner/) ---
    json_path: str = ""
    clinical_image_root: str = ""
    ceph_ckpt: str = "pretrained/hrnet_scratch.pt"
    text_encoder_path: str = "pretrained/pubmedbert_ortho_scratch"
    translations_path: str = ""
    llm_path: str = "Llama_3.1_8B-Instruct"
    output_dir: str = ""

    # --- data ---
    eval_fold: int = 0
    use_translation: bool = True   # use Gemini KO->EN diagnosis (else structured-field fallback)
    ablation: str = "both"         # both | vision (blank diagnosis text) | text (zero image)
    include_measurements: bool = True   # serialize 13 cephalometric measurements into the text input
    input_size: int = 768
    text_max_length: int = 256
    max_len: int = 512
    system_prompt: str = SYSTEM_PROMPT

    # --- Stage-1 encoder ---
    tau: float = 0.07

    # --- reasoner ---
    d_model: int = 768
    n_heads: int = 8
    n_geom_layers: int = 2
    n_cross_layers: int = 2
    queries_per_factor: int = 2
    dropout: float = 0.1
    # --- GUSR component ablation toggles (all True = full model) ---
    use_measurement_bridge: bool = True       # ① differentiable measurement bridge
    use_relation_bias: bool = True            # ② geometric relation bias in fusion attention
    use_decision_anchored_queries: bool = True  # ② decision-anchored (vs generic) queries
    use_uncertainty_gating: bool = True       # ③ uncertainty down-weighting + hedge
    use_decision_head: bool = True            # ④ constrained structured-decision head
    # --- v2: image-primary autonomous planning ---
    image_primary: bool = False               # plan from the radiograph alone (no diagnosis text)
    use_diagnosis_derivation: bool = False    # derive diagnostic descriptors from geometry
    use_metadata: bool = False                # feed age + gender (non-diagnostic) as a token
    lambda_diag: float = 0.5                  # weight of the descriptor-derivation loss
    # --- v3: neuro-symbolic clinical rule layer (differentiable, learnable-threshold diagnosis) ---
    use_rule_layer: bool = False              # derive diagnosis via explicit clinical rules (needs bridge+diag)
    use_rule_v2: bool = False                 # v3.2: multi-measure rules (gated dental/skeletal + evidence profile)
    use_ttype: bool = False                   # v3.3: predict treatment-type (opener) from diagnosis + age, condition the plan
    lambda_ttype: float = 0.5                 # weight of the treatment-type (opener) prediction loss
    lambda_rule_prior: float = 0.01           # light anchor keeping learned cut-points near textbook values
    # --- v3.1: measurement-primary — feed the rules the standard clinical measurements (accurate),
    #           not the noisy image-only geometry. Image/landmarks remain as context + fallback. ---
    measurement_primary: bool = False         # rules + measurement tokens use GT measurements; text dropped
    diag_drives_decision: bool = False         # feed the derived diagnosis into the decision head (controllability)
    dur_mean: float = 24.0
    dur_std: float = 12.0

    # --- LLM / LoRA ---
    fp16: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_targets: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")

    # --- loss weights ---
    lambda_measure: float = 0.5
    lambda_attr: float = 0.5
    lambda_calib: float = 0.1
    # Upweight minority scope (partial) / extraction (without) in L_attr CE.
    attr_class_balance: bool = False
    scope_ce_weights: tuple = (1.0, 5.0, 1.0)       # full, partial, surgical
    extraction_ce_weights: tuple = (3.0, 1.0)       # without, with

    # --- modality dropout (robustness) ---
    vision_dropout_p: float = 0.1
    text_dropout_p: float = 0.2

    # --- optimization ---
    epochs: int = 12
    batch_size: int = 1
    grad_accum: int = 8
    lr_reasoner: float = 1e-4
    lr_lora: float = 5e-5
    weight_decay: float = 0.05
    warmup_ratio: float = 0.1
    clip_grad: float = 1.0
    num_workers: int = 4
    seed: int = 42
    cuda: int = 0


def load_config(path: str | None = None, **overrides) -> Config:
    cfg = Config()
    if path:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid = {f.name for f in fields(Config)}
        for k, v in data.items():
            if k in valid:
                if k == "lora_targets" and isinstance(v, list):
                    v = tuple(v)
                setattr(cfg, k, v)
    for k, v in overrides.items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
