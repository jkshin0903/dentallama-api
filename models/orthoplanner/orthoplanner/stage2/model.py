"""OrthoPlanner full model (Stage 2).

Frozen Stage-1 cephalo encoder + frozen PubMedBERT diagnosis encoder + trainable GUSR reasoner +
LoRA-adapted Llama-3.1-8B. The reasoner builds a conditioning prefix; the LLM generates the
treatment plan (the only output). Trainable: reasoner + LoRA adapters.
"""
from __future__ import annotations

import os
import random

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..stage1.encoder import CephaloEncoder
from .reasoner import GUSRReasoner
from .text_encoder import DiagnosisEncoder


class OrthoPlanner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # --- frozen Stage-1 encoders ---
        self.ceph = CephaloEncoder(num_landmarks=29, tau=cfg.tau)
        if cfg.ceph_ckpt and os.path.exists(cfg.ceph_ckpt):
            self.ceph.load_backbone(cfg.ceph_ckpt)
        else:
            print(f"[OrthoPlanner] ceph_ckpt not found ({cfg.ceph_ckpt!r}); using random-init HRNet "
                  "(fine for smoke tests; train Stage 1 for real runs).")
        self.ceph.freeze()
        self.text = DiagnosisEncoder(cfg.text_encoder_path, max_length=cfg.text_max_length).freeze()

        # --- LLM backbone + LoRA ---
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            cfg.llm_path, torch_dtype=torch.float16 if cfg.fp16 else torch.float32,
            attn_implementation="eager")
        base.config.use_cache = False
        lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                          target_modules=list(cfg.lora_targets), bias="none", task_type="CAUSAL_LM")
        self.llm = get_peft_model(base, lora)
        llm_dim = base.config.hidden_size

        # --- trainable reasoner ---
        self.reasoner = GUSRReasoner(
            d_model=cfg.d_model, llm_dim=llm_dim, text_dim=self.text.hidden_size,
            n_heads=cfg.n_heads, n_geom_layers=cfg.n_geom_layers, n_cross_layers=cfg.n_cross_layers,
            queries_per_factor=cfg.queries_per_factor, dur_mean=cfg.dur_mean, dur_std=cfg.dur_std,
            dropout=cfg.dropout,
            use_measurement_bridge=cfg.use_measurement_bridge, use_relation_bias=cfg.use_relation_bias,
            use_decision_anchored_queries=cfg.use_decision_anchored_queries,
            use_uncertainty_gating=cfg.use_uncertainty_gating, use_decision_head=cfg.use_decision_head,
            use_diagnosis_derivation=cfg.use_diagnosis_derivation, use_metadata=cfg.use_metadata,
            use_rule_layer=getattr(cfg, "use_rule_layer", False),
            measurement_primary=getattr(cfg, "measurement_primary", False),
            diag_drives_decision=getattr(cfg, "diag_drives_decision", False),
            use_rule_v2=getattr(cfg, "use_rule_v2", False),
            use_ttype=getattr(cfg, "use_ttype", False))

        self.vision_dropout_p = cfg.vision_dropout_p
        self.text_dropout_p = cfg.text_dropout_p

    @property
    def device(self):
        return next(self.llm.parameters()).device

    def build_prefix(self, images, diagnosis_texts, apply_constraints=True, allow_dropout=False,
                     noise_std=0.0, metadata=None, gt_measurements=None, diag_override=None,
                     gt_measurements_v2=None):
        # modality ablations (deterministic, train+eval): isolate the geometry vs text pathway
        ab = getattr(self.cfg, "ablation", "both")
        if ab == "text":            # text-only: remove the image/landmark (geometry) signal
            images = torch.zeros_like(images)
        elif ab == "vision":        # vision-only: remove the diagnosis text (keeps landmark geometry)
            diagnosis_texts = [""] * len(diagnosis_texts)
        # v2 image-primary OR v3.1 measurement-primary: no diagnosis text (avoid conclusion leakage)
        if getattr(self.cfg, "image_primary", False) or getattr(self.cfg, "measurement_primary", False):
            diagnosis_texts = [""] * len(diagnosis_texts)
        enc = self.ceph(images)
        if noise_std > 0.0:
            # robustness study: perturb landmark coords; reflect added uncertainty in entropy
            enc = dict(enc)
            enc["landmarks"] = (enc["landmarks"] + noise_std * torch.randn_like(enc["landmarks"])).clamp(0, 1)
            enc["entropy"] = (enc["entropy"] + noise_std).clamp(0, 1)
            enc["conf"] = (enc["conf"] * (1.0 - noise_std)).clamp(0, 1)
        if allow_dropout and self.training and random.random() < self.vision_dropout_p:
            enc = {k: torch.zeros_like(v) for k, v in enc.items()}

        if allow_dropout and self.training and random.random() < self.text_dropout_p:
            diagnosis_texts = [""] * len(diagnosis_texts)
        text_tokens, text_mask = self.text(diagnosis_texts, self.device)
        text_tokens = text_tokens.to(enc["landmarks"].dtype)

        if metadata is not None:
            metadata = metadata.to(self.device, enc["landmarks"].dtype)
        if gt_measurements is not None:
            gt_measurements = gt_measurements.to(self.device, enc["landmarks"].dtype)
        if gt_measurements_v2 is not None:
            gt_measurements_v2 = gt_measurements_v2.to(self.device, enc["landmarks"].dtype)
        return self.reasoner(enc, text_tokens, text_mask=text_mask,
                             apply_constraints=apply_constraints, metadata=metadata,
                             gt_measurements=gt_measurements, diag_override=diag_override,
                             gt_measurements_v2=gt_measurements_v2)

    def forward(self, images, diagnosis_texts, input_ids, attention_mask, labels, metadata=None,
                gt_measurements=None, gt_measurements_v2=None):
        # apply_constraints=False at TRAIN time: the hard surgery/scope mask (logit -= 1e4) is an
        # inference-only rule; applying it during loss makes CE blow up on constrained classes.
        out = self.build_prefix(images, diagnosis_texts, apply_constraints=False, allow_dropout=True,
                                metadata=metadata, gt_measurements=gt_measurements,
                                gt_measurements_v2=gt_measurements_v2)
        prefix = out["prefix"]                                          # (B, P, llm_dim)
        B, P, _ = prefix.shape

        tok_emb = self.llm.get_input_embeddings()(input_ids)
        prefix = prefix.to(tok_emb.dtype)
        inputs_embeds = torch.cat([prefix, tok_emb], dim=1)
        attn = torch.cat([torch.ones(B, P, device=self.device, dtype=attention_mask.dtype),
                          attention_mask], dim=1)
        lab = torch.cat([torch.full((B, P), -100, device=self.device, dtype=labels.dtype),
                         labels], dim=1)
        lm_out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attn, labels=lab)
        return lm_out, out

    @torch.no_grad()
    def generate(self, image, diagnosis_text, system_prompt, max_new_tokens=160,
                 num_beams=4, noise_std=0.0, metadata=None, gt_measurements=None,
                 gt_measurements_v2=None,
                 diag_override=None, prompt_mode: str = "legacy",
                 gen_select: str = "greedy",
                 return_candidates: bool = False,
                 age=None, gender=None, age_years=None, korean_raw=None,
                 apply_growth_guard: bool = True,
                 augment_growth_rules: bool | None = None,
                 hard_reflect_head: bool | None = None,
                 **gen_kwargs):
        self.eval()
        out = self.build_prefix(image, [diagnosis_text], allow_dropout=False, noise_std=noise_std,
                                metadata=metadata, gt_measurements=gt_measurements,
                                gt_measurements_v2=gt_measurements_v2,
                                diag_override=diag_override)
        prefix = out["prefix"]
        from ..clinical.growth_guard import apply_growth_guard as _guard
        from ..clinical.growth_guard import augment_system_prompt, parse_age_years
        from ..data.clinical_dataset import chat_prompt
        from ..prompts.treatment_plan import user_message
        from ..prompts.generation import apply_head_to_plan, select_plan

        yrs = parse_age_years(age, age_years=age_years)
        use_rules = apply_growth_guard if augment_growth_rules is None else augment_growth_rules
        sys_prompt = augment_system_prompt(system_prompt, yrs) if use_rules else system_prompt
        user_turn = user_message(
            diagnosis_text, prompt_mode, age=age, gender=gender, age_years=yrs,  # type: ignore[arg-type]
        )
        text = chat_prompt(self.tokenizer, sys_prompt, user_turn)
        ids = self.tokenizer(text, return_tensors="pt").to(self.device)
        emb = self.llm.get_input_embeddings()(ids["input_ids"])
        inputs_embeds = torch.cat([prefix.to(emb.dtype), emb], dim=1)  # match LLM (fp16) dtype
        attn = torch.ones(inputs_embeds.shape[:2], device=self.device, dtype=torch.long)

        gen_kw = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.pad_token_id,
            **gen_kwargs,
        )
        rerank_modes = ("head_rerank", "parse_rerank")
        multi = return_candidates or (
            gen_select in rerank_modes and out.get("attr_logits") is not None and num_beams > 1
        )
        if multi:
            gen_kw.update(
                num_beams=num_beams,
                num_return_sequences=num_beams,
                early_stopping=True,
            )
        else:
            gen_kw["num_beams"] = num_beams
        gen = self.llm.generate(**gen_kw)
        candidates = [
            self.tokenizer.decode(row, skip_special_tokens=True).strip()
            for row in gen
        ]
        if apply_growth_guard:
            guarded = [_guard(c, yrs, korean_raw=korean_raw)[0] for c in candidates]
            candidates = guarded
        plan = select_plan(candidates, out, gen_select)
        if hard_reflect_head is None:
            flag = (os.environ.get("ORTHOPLANNER_HARD_REFLECT_HEAD") or "1").strip().lower()
            hard_reflect_head = flag not in ("0", "false", "no", "off")
        if hard_reflect_head and out.get("attr_logits") is not None:
            plan = apply_head_to_plan(plan, out)
        if return_candidates:
            return plan, out, candidates
        return plan, out

    def trainable_parameter_groups(self):
        reasoner_params = [p for p in self.reasoner.parameters() if p.requires_grad]
        lora_params = [p for n, p in self.llm.named_parameters() if p.requires_grad]
        return [
            {"params": reasoner_params, "lr": self.cfg.lr_reasoner},
            {"params": lora_params, "lr": self.cfg.lr_lora},
        ]
