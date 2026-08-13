"""
모델 로딩 유틸리티 모듈

현재는 Gemma 모델만 환경 변수 설정에 따라 로드합니다.
"""
import os
from typing import Optional, Tuple, Any
from pathlib import Path

import torch
from transformers import BitsAndBytesConfig, AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel


def _resolve_processor_path(model_path: str, processor_path: Optional[str] = None) -> str:
    """
    프로세서 경로를 결정합니다.
    환경 변수로 별도 경로가 주어지면 우선 사용하고,
    없으면 모델 경로 내에서 processor 폴더를 탐색합니다.

    Args:
        model_path: 모델 체크포인트 경로
        processor_path: 환경 변수로 지정된 프로세서 경로 (선택)

    Returns:
        프로세서 경로
    """
    if processor_path:
        return processor_path
    candidate = os.path.join(model_path, "processor")
    if os.path.isdir(candidate):
        return candidate
    return model_path


def load_gemma_model(
    model_path: str,
    processor_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    peft_adapter_path: Optional[str] = None,
    load_in_4bit: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Any, Any]:
    """
    Gemma-3 멀티모달 모델과 프로세서를 로드합니다.

    Args:
        model_path: Gemma 체크포인트 디렉터리 경로
        processor_path: 프로세서 리소스 경로 (선택)
        tokenizer_path: 별도 토크나이저 디렉터리 (선택)
        peft_adapter_path: PEFT 어댑터 경로 (선택)
        load_in_4bit: 4bit 양자화 사용 여부
        dtype: 모델 dtype (기본: bfloat16, Gemma 권장)

    Returns:
        (model, processor) 튜플
    """
    # 양자화 설정 (Gemma 권장 설정)
    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # 모델 로드
    print(f"[모델 로드] Gemma-3 from {model_path}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=quant_config,
        dtype=dtype,
        trust_remote_code=True,
    )

    # PEFT 어댑터 로드 (선택)
    if peft_adapter_path:
        print(f"[PEFT 어댑터 로드] {peft_adapter_path}")
        model = PeftModel.from_pretrained(model, peft_adapter_path)

    model.config.use_cache = True
    model.eval()

    # 프로세서 로드
    processor_path_resolved = _resolve_processor_path(model_path, processor_path)
    print(f"[프로세서 로드] {processor_path_resolved}")
    processor = AutoProcessor.from_pretrained(
        processor_path_resolved, 
        trust_remote_code=True,
        use_fast=False,  # slow processor 명시적으로 사용 (경고 방지)
    )

    # 토크나이저 설정 (pad_token 설정)
    tok = getattr(processor, "tokenizer", None)
    if tok:
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        if hasattr(model, "generation_config") and model.generation_config:
            if model.generation_config.pad_token_id is None:
                model.generation_config.pad_token_id = tok.pad_token_id
            if model.generation_config.eos_token_id is None:
                model.generation_config.eos_token_id = tok.eos_token_id

    # 토크나이저 동기화 (선택)
    if tokenizer_path:
        print(f"[토크나이저 동기화] {tokenizer_path}")
        if tok is not None:
            processor.tokenizer = tok.__class__.from_pretrained(tokenizer_path)

    print(f"✅ Gemma-3 ready on {getattr(model, 'device', 'unknown')}")
    return model, processor


def load_model_from_env() -> Tuple[Any, Any]:
    """
    환경 변수 설정에 따라 적절한 모델을 로드합니다.

    환경 변수:
        MODEL_TYPE: "gemma" (현재 Gemma만 지원)
        MODEL_PATH: 모델 체크포인트 경로
        PROCESSOR_PATH: 프로세서 경로 (선택)
        TOKENIZER_PATH: 토크나이저 경로 (선택)
        PEFT_ADAPTER_PATH: PEFT 어댑터 경로 (선택)
        LOAD_IN_4BIT: 4bit 양자화 사용 여부 (기본: "true")

    Returns:
        (model, processor) 튜플

    Raises:
        RuntimeError: 필수 환경 변수가 설정되지 않았거나 모델 타입이 잘못된 경우
    """
    model_type = os.getenv("MODEL_TYPE", "gemma").lower()
    model_path = os.getenv("MODEL_PATH")
    processor_path = os.getenv("PROCESSOR_PATH")
    tokenizer_path = os.getenv("TOKENIZER_PATH")
    peft_adapter_path = os.getenv("PEFT_ADAPTER_PATH")
    load_in_4bit = (
        os.getenv("LOAD_IN_4BIT", "true").lower() not in {"0", "false", "no"}
    )

    if not model_path:
        raise RuntimeError(
            "MODEL_PATH environment variable is required to load the model."
        )

    if model_type == "gemma":
        return load_gemma_model(
            model_path=model_path,
            processor_path=processor_path,
            tokenizer_path=tokenizer_path,
            peft_adapter_path=peft_adapter_path,
            load_in_4bit=load_in_4bit,
            dtype=torch.bfloat16,
        )
    else:
        raise RuntimeError(
            f"Unsupported MODEL_TYPE: {model_type}. Only 'gemma' is supported now."
        )

