# DentalLama API 서버

치과 교정 시나리오를 위한 FastAPI 추론 서버입니다.  
측모두부방사선(ceph)과 진단/계측을 받아 치료계획을 생성합니다. 모델이 두 개이며 엔드포인트가 다릅니다.

| 모델 | 엔드포인트 | 출력 |
|------|------------|------|
| **Gemma-3** (DentalInference + RAG) | `POST /api/v1/gemma/treatment-plan` | 한글 4필드 텍스트 |
| **OrthoPlanner** (HRNet + GUSR + Llama-3.1-8B LoRA) | `POST /api/v1/orthoplanner/treatment-plan` | 영어 1문장 + 구조화 필드 |

인증 없음. `Content-Type: application/json`. 스트리밍 없음.  
추론은 수십 초~수분 걸릴 수 있으므로 클라이언트 timeout은 300–600초를 권장합니다.

---

## 핵심 기능

- **Gemma 치료계획**  
  - 서버 기동 시 Gemma-3 + PEFT를 한 번 로드하고 요청마다 재사용  
  - BGE-M3 임베딩 + Milvus RAG (`DentalRAG_Correction`, `DentalRAG_Treatment`)  
  - 현재 `MODEL_TYPE=gemma`만 지원 (`llavanext`는 코드에서 거부)
- **OrthoPlanner 치료계획**  
  - `models/orthoplanner`의 코드·가중치를 같은 프로세스에서 직접 호출  
  - 기본은 첫 요청 때 로드 (`ORTHOPLANNER_PRELOAD=false`)  
  - 데모 웹 서버(`:20000`)는 사용하지 않음
- **상태 확인**  
  - `GET /health`, `GET /api/v1/orthoplanner/health`

---

## 설치

```bash
conda activate dentallama-api
pip install -r requirements.txt
```

Torch는 CUDA 환경에 맞춰 별도로 설치하세요.  
`numpy`, `pillow`, `pyyaml`, `accelerate`는 `requirements.txt`에 명시되어 있으며, `dentallama-api` 환경에는 이미 들어 있습니다.

---

## 환경 변수

`.env`는 `load_dotenv()`로 자동 로드됩니다. 예시는 `.env.example`.

### Gemma

```bash
MODEL_TYPE=gemma
GEMMA_MODEL_PATH=/home/web/projects/dentallama-api/models/gemma-3-12b-1031/checkpoint-600
GEMMA_PROCESSOR_PATH=/home/web/projects/dentallama-api/models/gemma-3-12b-1031/processor
GEMMA_TOKENIZER_PATH=/home/web/projects/dentallama-api/models/gemma-3-12b-1031/tokenizer
GEMMA_PEFT_ADAPTER_PATH=/home/web/projects/dentallama-api/models/gemma-3-12b-1031/peft_adapter
GEMMA_LOAD_IN_4BIT=true
```

### RAG / Milvus (Gemma만 사용)

```bash
RAG_EMBED_MODEL_PATH=/mnt/nvme01/huggingface/models/BAAI/bge-m3
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_TOKEN=
USE_RAG_DEFAULT=true
```

검색 대상 컬렉션: `DentalRAG_Correction`, `DentalRAG_Treatment`.  
Milvus는 컬렉션이 있어도 **메모리에 load**되어 있어야 검색됩니다. load되지 않으면 경고만 남기고 RAG 없이 Gemma만 동작합니다.

### OrthoPlanner

경로는 `models/orthoplanner/` 기준 상대경로 또는 절대경로입니다.

```bash
ORTHOPLANNER_CKPT=fold0/best
ORTHOPLANNER_LLM_PATH=Llama_3.1_8B-Instruct
ORTHOPLANNER_CONFIG=config.yaml
ORTHOPLANNER_CUDA=0          # CUDA_VISIBLE_DEVICES 안에서의 인덱스
ORTHOPLANNER_PRELOAD=false   # true면 기동 시 Llama까지 로드 (수분 소요)
```

---

## 실행

```bash
./scripts/run_server.sh
```

또는

```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

- 기본 포트: **8001**
- 로그: `logs/server.log`
- 종료: `./scripts/stop_server.sh`

기동 시 Gemma + BGE-M3를 올립니다. OrthoPlanner는 기본값에서 첫 `POST /api/v1/orthoplanner/treatment-plan` 때 로드합니다.

---

## API

두 엔드포인트의 페이로드는 **호환되지 않습니다.** 섞어 보내지 마세요.

이전 경로도 동일하게 동작합니다.

| 현재 | 이전 (호환) |
|------|-------------|
| `POST /api/v1/gemma/treatment-plan` | `POST /api/chat` |
| `POST /api/v1/orthoplanner/treatment-plan` | `POST /api/v1/treatment-plan` |
| `GET /api/v1/orthoplanner/health` | `GET /api/v1/ortho/health` |

### `POST /api/v1/gemma/treatment-plan`

ceph 이미지 + 진단(선택 계측) → 한글 4줄.

```json
{
  "id": "case-001",
  "model": "gemma-3",
  "messages": [
    {
      "role": "user",
      "content": "{\"diagnosis\":\"골격성 II급\",\"measurements\":{\"SNA\":81.0,\"ANB\":3.0},\"use_rag\":true}"
    }
  ],
  "files": [
    {
      "name": "Ceph.jpg",
      "type": "image/jpeg",
      "size": 123456,
      "content": "<raw base64, data-URL 접두사 없이>"
    }
  ]
}
```

| 필드 | 필수 | 설명 |
|------|------|------|
| `model` | 스키마상 필수 | 값은 사용하지 않음. `"gemma-3"` 등 |
| `messages` | 필수 | **마지막 메시지**만 사용 |
| `messages[].content` | 필수 | 진단 문자열, 또는 JSON 문자열 (`diagnosis` / `measurements` / `use_rag`) |
| `files` | 사실상 필수 | 첫 `image/*`를 ceph로 사용. `content`는 raw base64 |
| `id` | 선택 | 로깅용 |

`diagnosis`가 객체이면 `chiefComplain`, `diagnosis`, `treatmentPlan`, `etc`를 `주소:` / `진단:` 등으로 이어 붙입니다.

성공 응답:

```json
{
  "response": "교정: 전체\n발치: 44/44 발치\n추가시술: null\n치료 기간: 2년 6개월"
}
```

에러 `detail`은 `{"response": "..."}` 형태입니다. (메시지/이미지 없음 → 400, 모델 미준비 → 503)

### `POST /api/v1/orthoplanner/treatment-plan`

ceph + 계측 → 영어 치료계획 문장 + decision head.

```json
{
  "ceph": "data:image/jpeg;base64,/9j/4AAQ...",
  "analysischart": {
    "SNA": 78.42,
    "SNB": 74.51,
    "ANB": 3.91,
    "FMA": 34.52,
    "Overjet": 8.85,
    "Overbite": 3.81,
    "Mx1 to SN": 99.61,
    "MP to Mn1": 79.31,
    "IIA": 134.58,
    "Eline to U lip": -1.27,
    "Eline to L lip": 2.06,
    "APDI": 78.71,
    "Combination Factor(CF)": 140.89
  },
  "age": "22Y 9M",
  "gender": "Male",
  "diagnosis": "Skeletal Class II"
}
```

| 필드 | 필수 | 설명 |
|------|------|------|
| `ceph` | 필수 | 이미지 base64. data-URL 접두사 허용 |
| `analysischart` | 계측 중 하나 | 계측 dict |
| `analysiscsv` | 계측 중 하나 | AnalysisChart.csv 본문. 환자 값은 4번째 열 `normal` |
| `analysisexcel_b64` | 계측 중 하나 | Excel base64 |
| `diagnosis` / `age` / `gender` | 선택 | 한/영 진단, `"22Y 9M"`, `Male`/`Female`/`M`/`F` |
| `model` / `fold` | 선택 | 기본 `cephstruct_rule` fold 0. 현재 서버는 이 체크포인트만 로드 |
| `hard_reflect_head` | 선택 | 기본 `true`. head 예측을 문장에 반영 |

핵심 계측 13키: `SNA`, `SNB`, `ANB`, `FMA`, `Overjet`, `Overbite`, `Mx1 to SN`, `MP to Mn1`, `IIA`, `Eline to U lip`, `Eline to L lip`, `APDI`, `Combination Factor(CF)`

성공 응답:

```json
{
  "treatment_plan": "Orthodontic treatment, with full orthodontic scope, without extraction, with a treatment duration of 24 months.",
  "parsed": { "scope": "full", "extraction": 0, "duration": 24 },
  "decision_head": {
    "scope": { "prediction": 0, "label": "full", "confidence": 0.91 },
    "extraction": { "prediction": 0, "label": "no", "confidence": 0.88 },
    "surgery": { "prediction": 0, "label": "no", "confidence": 0.95 }
  },
  "hedge": 0.12,
  "duration_months_est": 24.3,
  "measurements_used": { "SNA": 78.42 },
  "model": "cephstruct_rule",
  "model_fold": 0,
  "model_checkpoint": ".../models/orthoplanner/fold0/best",
  "prompt_mode": "promptpp_compact",
  "gen_select": "head_rerank"
}
```

에러 `detail`은 문자열입니다. (계측/ceph 문제 → 400, 로드 실패 → 503)

### `GET /health` · `GET /api/v1/orthoplanner/health`

- `/health`: 서버 liveness + OrthoPlanner 로드 여부
- `/api/v1/orthoplanner/health`: 체크포인트 경로, `loaded` (`not_loaded`는 첫 요청 전 정상)

---

## 프로젝트 구조

```
dentallama-api/
├── main.py
├── classes/
│   ├── chat.py                      # Gemma 요청 스키마
│   ├── dental_inference.py          # Gemma 추론 + RAG
│   ├── model_loader.py              # Gemma 로딩
│   ├── orthoplanner_schema.py       # OrthoPlanner 요청/응답 스키마
│   └── orthoplanner_inference.py    # OrthoPlanner 로드·추론
├── models/
│   ├── gemma-3-12b-1031/            # Gemma 가중치 (gitignore)
│   └── orthoplanner/
│       ├── orthoplanner/            # Stage1/2 모델 코드
│       ├── config.yaml
│       ├── fold0/best/              # reasoner + LoRA
│       ├── pretrained/              # HRNet, PubMedBERT
│       └── Llama_3.1_8B-Instruct/
├── scripts/
│   ├── run_server.sh
│   ├── stop_server.sh
│   └── check_model_load.sh
├── logs/
├── requirements.txt
├── .env
└── README.md
```

가중치(수십 GB)는 gitignore입니다. `models/orthoplanner/orthoplanner/` 소스와 `config.yaml`만 추적합니다.

---

## 참고 사항

1. **GPU**: `main.py`가 `CUDA_VISIBLE_DEVICES=4`로 고정합니다. 프로세스 안에서는 `cuda:0`입니다. Gemma와 OrthoPlanner가 **같은 GPU**에 같이 올라갑니다.
2. **Milvus**: 컬렉션이 load되어 있지 않으면 `collection not loaded` 경고 후 RAG 없이 진행됩니다. 검색 전에 `DentalRAG_Correction`, `DentalRAG_Treatment`를 load하세요.
3. **보안**: Gemma 경로는 이미지를 `/tmp`에 풀었다가 요청 종료 시 삭제합니다. 민감 데이터는 HTTPS에서만 사용하세요.
4. **연구/데모용**입니다. 임상 사용을 전제하지 않습니다.
