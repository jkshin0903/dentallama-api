"""
치과 프로젝트용 경량 FastAPI 서버
"""

from contextlib import asynccontextmanager
import json
import os
import base64
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from dotenv import load_dotenv
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from FlagEmbedding import BGEM3FlagModel
import pymupdf
from pymilvus import connections

from classes.chat import ChatRequest, FileData
from classes.dental_inference import DentalInference
from classes.model_loader import load_model_from_env

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # 0 사용 할시 (~48G 사용)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 다른사람이 실수로 접속해서 메모리 초과 되서 끊기는 것 방지 가능
os.environ["TORCH_USE_CUDA_DSA"] = "1"

device = torch.device("cuda:0")  # VISIBLE DEVICES 중 0번째 사용할 시
torch.cuda.set_device(device)

# Load environment variables
load_dotenv()  # 현재 디렉토리의 .env 파일 로드


BASE_DIR = Path(__file__).resolve().parent

# 모델 타입 및 경로 설정 (현재 Gemma만 지원)
MODEL_TYPE = os.getenv("MODEL_TYPE", "gemma").lower()
if MODEL_TYPE != "gemma":
    raise RuntimeError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}. Only 'gemma' is supported.")

def _get_env(name: str) -> Optional[str]:
    return os.getenv(name) if name else None

MODEL_PATH = _get_env("GEMMA_MODEL_PATH")
PROCESSOR_PATH = _get_env("GEMMA_PROCESSOR_PATH")
TOKENIZER_PATH = _get_env("GEMMA_TOKENIZER_PATH")
PEFT_ADAPTER_PATH = _get_env("GEMMA_PEFT_ADAPTER_PATH")
LOAD_IN_4BIT = (
    (_get_env("GEMMA_LOAD_IN_4BIT") or "true").lower() not in {"0", "false", "no"}
)

# RAG 및 Milvus 설정
RAG_EMBED_MODEL_PATH = os.getenv("RAG_EMBED_MODEL_PATH")  # BGE-M3 임베딩 모델 위치
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")  # Milvus 호스트 주소
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")  # Milvus 포트
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")  # Milvus 인증 토큰 (필요 시)
DEFAULT_USE_RAG = (
    os.getenv("USE_RAG_DEFAULT", "true").lower() not in {"0", "false", "no"}
)  # RAG 기본 활성 여부

inference_system: Optional[DentalInference] = None
_model_loaded: bool = False  # 모델 로드 상태 추적


def _build_inference_system() -> DentalInference:
    """
    DentalInference가 필요로 하는 구성 요소를 순차적으로 초기화합니다.
      1) Gemma 모델 및 프로세서 로드
      2) BGE-M3 임베딩 모델 로드
      3) Milvus 연결
    모델이 이미 로드되어 있으면 재로드하지 않고 기존 인스턴스를 반환합니다.
    """
    global inference_system, _model_loaded
    
    # 이미 로드되어 있으면 재로드하지 않음
    if _model_loaded and inference_system is not None:
        print("[모델 상태] 이미 로드된 모델을 재사용합니다.")
        return inference_system
    
    if not MODEL_PATH:
        model_var_name = (
            "LLAVA_MODEL_PATH" if MODEL_TYPE == "llavanext" else "GEMMA_MODEL_PATH"
        )
        raise RuntimeError(
            f"{model_var_name} environment variable is not set."
        )
    if not RAG_EMBED_MODEL_PATH:
        raise RuntimeError("RAG_EMBED_MODEL_PATH environment variable is not set.")

    print("[모델 로드 시작] 새로운 모델 인스턴스를 로드합니다.")
    
    # 환경 변수 설정 (model_loader가 사용)
    os.environ["MODEL_TYPE"] = MODEL_TYPE
    os.environ["MODEL_PATH"] = MODEL_PATH
    if PROCESSOR_PATH:
        os.environ["PROCESSOR_PATH"] = PROCESSOR_PATH
    if TOKENIZER_PATH:
        os.environ["TOKENIZER_PATH"] = TOKENIZER_PATH
    if PEFT_ADAPTER_PATH:
        os.environ["PEFT_ADAPTER_PATH"] = PEFT_ADAPTER_PATH
    os.environ["LOAD_IN_4BIT"] = "true" if LOAD_IN_4BIT else "false"

    # 모델 및 프로세서 로드
    print(f"[모델 타입] {MODEL_TYPE}")
    model, processor = load_model_from_env()

    # RAG 임베더 로드
    print(f"[RAG 임베더 로드] {RAG_EMBED_MODEL_PATH}")
    rag_encoder = BGEM3FlagModel(RAG_EMBED_MODEL_PATH, use_fp16=True)

    # Milvus 연결 (이미 연결되어 있으면 재연결하지 않음)
    try:
        print(f"[Milvus 연결 시도] {MILVUS_HOST}:{MILVUS_PORT}")
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            token=MILVUS_TOKEN,
        )
    except Exception as e:
        # 이미 연결되어 있으면 무시
        print(f"[Milvus 연결] 이미 연결되어 있거나 연결 실패: {e}")

    inference_system = DentalInference(model=model, processor=processor, rag_encoder=rag_encoder)
    _model_loaded = True
    print("[모델 로드 완료] 모델이 성공적으로 로드되었습니다.")
    
    return inference_system


def _normalize_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        val = value.strip().lower()
        if val in {"0", "false", "no"}:
            return False
        if val in {"1", "true", "yes"}:
            return True
    return default


def _parse_message_payload(content: str) -> Tuple[str, Dict, bool]:
    """
    사용자가 JSON 문자열로 `diagnosis`, `measurements_results`, `use_rag`를 넘겼을 때
    안전하게 파싱하여 DentalInference.run에 그대로 전달합니다.
    단순 텍스트인 경우에는 진단 텍스트만 사용하는 fallback을 제공합니다.
    
    diagnosis가 객체인 경우 (chiefComplain, diagnosis, treatmentPlan, etc 필드 포함):
    각 필드를 조합하여 하나의 문자열로 변환합니다.
    """
    if not content:
        return "", {}, DEFAULT_USE_RAG
    try:
        payload = json.loads(content)
        if isinstance(payload, dict):
            diagnosis_raw = (
                payload.get("diagnosis")
                or payload.get("diagnosis_text")
                or payload.get("text")
                or ""
            )
            
            # diagnosis가 딕셔너리인 경우 (새로운 포맷)
            if isinstance(diagnosis_raw, dict):
                diagnosis_parts = []
                # 필드 순서대로 조합
                if diagnosis_raw.get("chiefComplain"):
                    chief_complain = diagnosis_raw["chiefComplain"]
                    # JSON 문자열 내부의 이스케이프된 따옴표 제거
                    if isinstance(chief_complain, str) and chief_complain.startswith('"') and chief_complain.endswith('"'):
                        chief_complain = chief_complain[1:-1]
                    diagnosis_parts.append(f"주소: {chief_complain}")
                
                if diagnosis_raw.get("diagnosis"):
                    diag = diagnosis_raw["diagnosis"]
                    if isinstance(diag, str) and diag.startswith('"') and diag.endswith('"'):
                        diag = diag[1:-1]
                    diagnosis_parts.append(f"진단: {diag}")
                
                if diagnosis_raw.get("treatmentPlan"):
                    treatment = diagnosis_raw["treatmentPlan"]
                    if isinstance(treatment, str) and treatment.startswith('"') and treatment.endswith('"'):
                        treatment = treatment[1:-1]
                    diagnosis_parts.append(f"치료 계획: {treatment}")
                
                if diagnosis_raw.get("etc") and diagnosis_raw["etc"] != "etc":
                    etc = diagnosis_raw["etc"]
                    if isinstance(etc, str) and etc.startswith('"') and etc.endswith('"'):
                        etc = etc[1:-1]
                    diagnosis_parts.append(f"기타: {etc}")
                
                diagnosis = "\n".join(diagnosis_parts) if diagnosis_parts else ""
            else:
                # 기존 방식: 문자열인 경우
                diagnosis = diagnosis_raw if isinstance(diagnosis_raw, str) else str(diagnosis_raw) if diagnosis_raw else ""
            
            measurements = payload.get("measurements") or payload.get("measurements_results") or {}
            measurements = measurements if isinstance(measurements, dict) else {}
            use_rag = _normalize_bool(payload.get("use_rag", DEFAULT_USE_RAG), DEFAULT_USE_RAG)
            return diagnosis or content, measurements, use_rag
    except json.JSONDecodeError:
        pass
    return content, {}, DEFAULT_USE_RAG


def _select_image_file(files: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """
    DentalInference는 필수적으로 ceph 이미지 경로가 있어야 하므로
    업로드된 파일 중 MIME type이 image인 첫 번째 항목을 선택합니다.
    이미지가 없으면 추론을 수행할 수 없습니다.
    """
    for file_info in files:
        file_type = (file_info.get("type") or "").lower()
        if file_type.startswith("image/"):
            return file_info
    return files[0] if files else None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global inference_system, _model_loaded
    # 모델이 이미 로드되어 있지 않을 때만 로드
    if not _model_loaded:
        inference_system = _build_inference_system()
    try:
        yield
    finally:
        # 서버 종료 시에만 정리 (reload 시에는 유지)
        # 실제 프로세스 종료 시에만 정리되도록 하려면 이 부분을 주석 처리하거나
        # 환경 변수로 제어할 수 있습니다
        # inference_system = None
        # _model_loaded = False
        # try:
        #     connections.disconnect("default")
        # except Exception:
        #     pass
        pass


app = FastAPI(title="DentalLama API", version="1.0.0", lifespan=lifespan)

# CORS settings mirroring flask.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["Content-Type"],
    max_age=3600,
)

def decode_base64_file(file_data: FileData) -> Optional[Dict[str, str]]:
    """
    base64로 인코딩된 파일 데이터를 디코딩하고 임시 파일로 저장합니다.
    
    Args:
        file_data: FileData 객체 (name, type, size, content 포함)
    
    Returns:
        tuple: (파일명, 파일 경로)
    """
    try:
        
        # base64 디코딩
        file_bytes = base64.b64decode(file_data.content)
        
        # tempfile.NamedTemporaryFile 사용 (자동 삭제)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file_data.name}")
        file_path = temp_file.name
        
        with open(file_path, 'wb') as f:
            f.write(file_bytes)
        
        return {
            "name": file_data.name,
            "path": file_path,
            "type": file_data.type,
        }
    
    except Exception as e:
        print(f"[디코딩 실패] {file_data.name}: {e}")
        return None


def cleanup_temp_files(file_paths):
    """
    임시 파일들을 삭제합니다.
    
    Args:
        file_paths: 삭제할 파일 경로 리스트
    """
    print(f"[파일 정리] {len(file_paths)}개 파일 삭제 시작")
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"[파일 정리] 삭제 완료: {file_path}")
            else:
                print(f"[파일 정리] 파일 없음 (이미 삭제됨): {file_path}")
        except Exception as e:
            print(f"[파일 정리 실패] {file_path}: {str(e)}")
    print("[파일 정리] 완료")


@app.post("/api/chat")
def chat(req: ChatRequest):
    print("\n" + "="*80)
    print("[요청 수신] POST /api/chat")
    print("="*80)
    print(f"[요청 정보] ID: {req.id}, 모델: {req.model}, 메시지: {len(req.messages)}, 파일: {len(req.files) if req.files else 0}")
    
    # base64로 인코딩된 파일 디코딩
    print("\n[1단계] 파일 디코딩 시작")
    decoded_files: List[Dict[str, str]] = []
    try:
        if req.files:
            for file_data in req.files:
                print(f"[디코딩] {file_data.name} 처리 중...")
                decoded = decode_base64_file(file_data)
                if decoded:
                    decoded_files.append(decoded)
                    print(f"[디코딩 완료] {decoded['name']}")
        print(f"[1단계 완료] {len(decoded_files)}개 파일 디코딩 성공")
        
        # 마지막 user 메시지 추출
        print("\n[2단계] 메시지 추출")
        if not req.messages:
            print("[에러] 메시지가 비어있음")
            raise HTTPException(status_code=400, detail={"response": "메시지가 비어 있습니다."})
        
        last_message = req.messages[-1]
        print(f"[메시지] Role: {last_message.role}, Content 길이: {len(last_message.content)}")
        
        user_question = (last_message.content or "").strip()
        if not user_question:
            print("[에러] 메시지 내용이 비어있음")
            raise HTTPException(status_code=400, detail={"response": "질문이 비어 있습니다."})
        
        diagnosis_text, measurements, use_rag = _parse_message_payload(user_question)
        
        # 페이로드 정보 로깅
        print(f"[페이로드 파싱 결과]")
        print(f"  - diagnosis 길이: {len(diagnosis_text)} 문자")
        print(f"  - measurements 항목 수: {len(measurements)}")
        if measurements:
            print(f"  - measurements 키: {list(measurements.keys())}")
        print(f"  - use_rag: {use_rag}")
        if diagnosis_text:
            # diagnosis 내용 일부만 출력 (너무 길면 잘라서)
            diagnosis_preview = diagnosis_text[:200] + "..." if len(diagnosis_text) > 200 else diagnosis_text
            print(f"  - diagnosis 미리보기: {diagnosis_preview}")
        
        # DentalInference 실행
        print("\n[3단계] DentalInference 예측")
        if inference_system is None:
            print("[에러] 추론 시스템이 초기화되지 않음")
            raise HTTPException(status_code=503, detail={"response": "추론 시스템이 준비되지 않았습니다."})
        
        image_file = _select_image_file(decoded_files)
        if not image_file:
            raise HTTPException(status_code=400, detail={"response": "이미지 파일이 필요합니다."})
        
        # 추론 전 CUDA 캐시 정리 (메모리 부족 방지)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[메모리 정리] CUDA 캐시 정리 완료")
        
        predicted_plan = inference_system.run(
            image_path=image_file["path"],
            diagnosis_text=diagnosis_text,
            measurements=measurements,
            use_rag=use_rag,
        )
        
        # 추론 후 CUDA 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"[답변] 예측 길이: {len(predicted_plan)} 문자")
        print("\n[응답 전송] 요청 처리 완료")
        print("="*80 + "\n")
        return {"response": predicted_plan}
    finally:
        if decoded_files:
            print("\n[정리] 전달된 파일 삭제")
            cleanup_temp_files([file_info["path"] for file_info in decoded_files])


@app.get("/")
def read_root():
    return {"message": "DentalLama API 서버가 정상적으로 실행 중입니다."}

@app.get("/health")
def health_check():
    """서버 상태 확인 엔드포인트"""
    return {
        "status": "healthy",
        "features": ["pdf_text_extraction", "dental_inference"]
    }