# DentalLama API 서버

치과 교정 시나리오를 위한 FastAPI 기반 추론 서버입니다.  
업로드된 세팔로 측모 이미지와 진단/계측 정보를 입력으로 받아 DentalInference(LlavaNext + RAG)를 실행하고,  
첨부된 PDF에서는 PyMuPDF로 텍스트를 추출한 뒤 결과에 함께 제공합니다.

---

## 핵심 기능

- **다중 모델 지원**  
  - `MODEL_TYPE` 환경 변수로 LlavaNext 또는 Gemma 모델 선택 가능
  - 각 모델별로 독립된 환경 변수 사용 (LLAVA_*, GEMMA_*)
  - 모델 로딩 로직은 `classes/model_loader.py`에 모듈화되어 있음

- **DentalInference**  
  - 선택된 모델 + PEFT 어댑터를 로드해 멀티모달 추론 수행  
  - BGE-M3 임베딩 + Milvus를 통한 RAG 검색 결과를 프롬프트에 삽입  
  - `/api/v1/gemma/treatment-plan` 요청마다 전역 인스턴스를 재사용하므로 서버 부팅 시 한 번만 모델을 초기화
- **파일 파이프라인**  
  - base64로 전달된 이미지·PDF를 임시 파일로 복원  
  - PDF는 PyMuPDF로 텍스트 추출 후 응답 본문에 첨부  
  - 처리 후 모든 임시 파일을 즉시 삭제
- **상태 확인 및 로깅**  
  - `/` 및 `/health`에서 서버 상태 노출  
  - 요청/응답 과정 전체를 한국어 로그로 출력

---

## 설치

```bash
pip install -r requirements.txt
```

> Torch는 CUDA 환경에 맞춰 별도로 설치하세요.

### 필수 패키지
- FastAPI, Uvicorn
- Transformers, PEFT, FlagEmbedding, safetensors
- PyMuPDF, pymilvus

---

## 환경 변수 (.env 예시)

### 모델 타입 선택
```bash
MODEL_TYPE=llavanext  # 또는 "gemma"
```

### LlavaNext 모델 사용 시
```bash
MODEL_TYPE=llavanext
LLAVA_MODEL_PATH=/home/web/projects/dentallama-api/models/llava-next-1016-no_device/checkpoint-300
LLAVA_PROCESSOR_PATH=/home/web/projects/dentallama-api/models/llava-next-1016-no_device/processor
LLAVA_TOKENIZER_PATH=/home/web/projects/dentallama-api/models/llava-next-1016-no_device/checkpoint-300
LLAVA_PEFT_ADAPTER_PATH=/home/web/projects/dentallama-api/models/llava-next-1016-no_device/peft_adapter
LLAVA_LOAD_IN_4BIT=true        # 4bit 양자화 로딩 (BitsAndBytes)
```

### Gemma 모델 사용 시
```bash
MODEL_TYPE=gemma
GEMMA_MODEL_PATH=/home/web/projects/dentallama-api/models/gemma-3-12b-1031/checkpoint-600
GEMMA_PROCESSOR_PATH=/home/web/projects/dentallama-api/models/gemma-3-12b-1031/processor
GEMMA_TOKENIZER_PATH=/home/web/projects/dentallama-api/models/gemma-3-12b-1031/tokenizer
GEMMA_PEFT_ADAPTER_PATH=/home/web/projects/dentallama-api/models/gemma-3-12b-1031/peft_adapter
GEMMA_LOAD_IN_4BIT=true        # 4bit 양자화 로딩 (BitsAndBytes)
```

### 공통 설정 (모든 모델 공통)
```bash
RAG_EMBED_MODEL_PATH=/mnt/nvme01/huggingface/models/BAAI/bge-m3
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_TOKEN=andlab:andlabyonsei
USE_RAG_DEFAULT=true           # 요청 payload에 명시되지 않았을 때의 기본값
```

> `.env`는 `load_dotenv()`로 자동 로드됩니다.  
> `MODEL_TYPE`에 따라 해당 모델의 환경 변수만 읽습니다.

---

## 실행

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

서버 시작 시 `lifespan` 훅이 DentalInference 인스턴스를 준비합니다.

---

## API

### `POST /api/v1/gemma/treatment-plan`
Gemma 멀티모달 추론. 한글 4필드 치료계획. 이전 경로 `/api/chat`도 동일하게 동작합니다.

| 필드 | 설명 |
| --- | --- |
| `messages` | 최소 1개의 사용자 메시지 필요. 마지막 메시지의 `content`가 진단 텍스트 혹은 JSON(payload: `diagnosis`, `measurements`, `use_rag`) |
| `files` | base64 인코딩 파일 배열. 이미지 1개 이상 필수, PDF는 선택 |

#### 응답 예시
```json
{
  "response": "교정(전체)...\n발치(비발치)...\n[파일 텍스트]\n=== 파일: ceph.pdf ===\n..."
}
```

### `POST /api/v1/orthoplanner/treatment-plan`
OrthoPlanner 추론. 영어 치료계획 문장 + decision head. 이전 경로 `/api/v1/treatment-plan`도 동일하게 동작합니다.

### `GET /` · `GET /health`
- 서버 상태, 활성 기능(`dental_inference`, `pdf_text_extraction`, `orthoplanner`) 확인
- OrthoPlanner 로드 상태: `GET /api/v1/orthoplanner/health`

---

## 프로젝트 구조

```
dentallama-api/
├── classes/
│   ├── chat.py              # Pydantic 스키마
│   ├── dental_inference.py  # DentalInference 본체
│   └── model_loader.py       # 모델 로딩 유틸리티 (LlavaNext/Gemma)
├── models/                  # 모델 체크포인트 디렉터리
│   ├── llava-next-1016-no_device/
│   └── gemma-3-12b-1031/
├── scripts/
│   └── check_model_load.sh  # 모델 로딩 테스트 스크립트
├── main.py                  # FastAPI 서버 & 초기화 로직
├── requirements.txt
├── README.md
└── .env                     # 환경 변수 (배포 환경에서 관리)
```

---

## 참고 사항

1. **GPU/메모리**: 모델은 CUDA 4bit 모드로 로드되며 `CUDA_VISIBLE_DEVICES`는 `main.py`에서 5번 GPU로 고정되어 있습니다. 필요 시 수정하세요.
2. **Milvus**: 연결 실패 시 경고만 출력하고 RAG 없이 동작하지만, 정확한 치료 플랜을 위해 서버 가동 전 Milvus 인스턴스를 준비하는 것이 좋습니다.
3. **보안**: 업로드 파일은 `/tmp`에 저장 후 즉시 삭제되지만, 민감 데이터가 포함된 경우 HTTPS 환경에서만 사용하십시오.
