from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from pymilvus import Collection


CHEAT_SHEET = (
    "You are an orthodontic assistant. Use these compact meanings:\n"
    "- SNA/SNB: Maxilla/Mandible AP position vs SN.\n"
    "- ANB: Sagittal jaw relation (Class II/III tendency).\n"
    "- APDI: AP discrepancy index. CF = APDI + ODI.\n"
    "- FMA: Vertical growth pattern.\n"
    "- Mx1 to SN: Upper incisor inclination to SN.\n"
    "- IIA: Interincisal angle (total incisor inclination).\n"
    "- E-line to U/L lip: Lip protrusion (+) vs retrusion (-).\n"
    "- Nasolabial A.: Nasolabial angle.\n"
    "- MP to Mn1 (≈IMPA): Lower incisor to mandibular plane.\n"
    "- OJ/OB: Horizontal/Vertical overlap.\n"
    "- ALD: Arch length discrepancy (−crowding / +spacing).\n"
    "- Bolton Ant.: Anterior tooth-size ratio."
)

RAG_COLLECTIONS = ["DentalRAG_Correction", "DentalRAG_Treatment"]

__all__ = ["CHEAT_SHEET", "RAG_COLLECTIONS", "DentalInference"]


class DentalInference:
    """
    Wrapper around a vision-language model with optional Milvus-backed RAG lookup.

    Parameters
    ----------
    model: transformers.PreTrainedModel
        LlavaNext (or compatible) model that supports `.generate`.
    processor: transformers.ProcessorMixin
        Processor that can turn multimodal chat conversations into tensors.
    rag_encoder: FlagEmbedding.BGEM3FlagModel
        Text encoder used to embed retrieval queries.
    """

    def __init__(self, model, processor, rag_encoder):
        print("Initializing DentalInference System...")
        self.model = model
        self.processor = processor
        self.rag_encoder = rag_encoder
        self.device = getattr(self.model, "device", torch.device("cpu"))
        print(f"Inference system is set to use device: {self.device}")

    def _encode_for_rag(self, texts: List[str]) -> np.ndarray:
        """Encode text snippets to normalized BGE-M3 embeddings."""
        out = self.rag_encoder.encode(
            texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        vecs = out["dense_vecs"]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return (vecs / norms).astype("float32")

    def _retrieve_rag_content(self, diagnosis: str, top_k: int = 3) -> List[str]:
        """Query Milvus collections for references related to the diagnosis."""
        # Ensure diagnosis is a string
        if not diagnosis:
            return []
        
        # Convert to string if it's not already (handles cases where diagnosis might be a list, dict, etc.)
        if not isinstance(diagnosis, str):
            diagnosis = str(diagnosis)

        print(
            "Searching for RAG content with query: "
            f"'{diagnosis[:50].replace(chr(10), ' ')}...'"
        )

        query_text = (
            f"Retrieve orthodontic treatment strategies for the diagnosis: {diagnosis}. "
            "Extract relevant knowledge from the orthodontic textbook database."
        )
        query_vector = self._encode_for_rag([query_text])[0].tolist()

        all_contents: List[str] = []
        for collection_name in RAG_COLLECTIONS:
            try:
                collection = Collection(name=collection_name)
                res = collection.search(
                    data=[query_vector],
                    anns_field="vector",
                    param={"metric_type": "IP", "params": {"nprobe": 16}},
                    limit=top_k,
                    output_fields=["content"],
                )
                all_contents.extend([hit.entity.get("content") for hit in res[0]])
            except Exception as exc:
                print(
                    f"⚠️ Warning: Failed to search in collection '{collection_name}': {exc}"
                )

        # Preserve original order while deduplicating.
        return list(dict.fromkeys(all_contents))

    def run(
        self,
        image_path: str,
        diagnosis_text: str,
        measurements: Dict,
        use_rag: bool = True,
        max_new_tokens: int = 256,
    ) -> str:
        try:
            # Ensure diagnosis_text is a string
            if not diagnosis_text:
                diagnosis_text = ""
            elif not isinstance(diagnosis_text, str):
                diagnosis_text = str(diagnosis_text)
            measurements = measurements or {}

            try:
                raw_image = Image.open(image_path).convert("RGB")
                images = [raw_image]
            except Exception as exc:
                return (
                    "[Error] Image not found or could not be opened: "
                    f"{image_path} ({exc})"
                )

            rag_block = ""
            if use_rag:
                rag_contents = self._retrieve_rag_content(diagnosis_text)
                rag_block = self.format_rag_references(rag_contents)

            measurements_str = self.format_measurements(measurements)
            user_text = (
                f"{CHEAT_SHEET}\n\n"
                "Based on the attached cephalometric radiograph, the diagnosis, "
                "measurements, and references, produce a concise, actionable "
                "orthodontic treatment plan.\n\n"
                f"Diagnosis:\n{diagnosis_text}\n\n"
                f"Measurements:\n{measurements_str}"
                f"{rag_block}\n\n"
                "The output should only contain these 5 fields, one per line: "
                "교정 (전체/부분), 발치 (발치/비발치), 추가시술, 치료 기간. "
                "If a field is not applicable, output 'null'."
            )

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_text},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
            )

            # 프로세서 파라미터 설정 (모델 타입에 따라 다름)
            processor_kwargs = {
                "text": prompt,
                "images": images,
                "return_tensors": "pt",
                "padding": True,
            }
            # image_aspect_ratio는 LlavaNext 전용 (Gemma는 지원하지 않음)
            processor_type = type(self.processor).__name__
            if "LlavaNext" in processor_type:
                processor_kwargs["image_aspect_ratio"] = "pad"
            
            inputs = self.processor(**processor_kwargs)

            device = getattr(self.model, "device", torch.device("cpu"))
            tensor_inputs = {
                key: (val.to(device) if torch.is_tensor(val) else val)
                for key, val in inputs.items()
            }

            # pad_token_id 가져오기
            pad_token_id = getattr(
                getattr(self.processor, "tokenizer", self.processor),
                "eos_token_id",
                0,
            )
            
            # 생성 파라미터 설정
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "top_p": 1.0,
                "repetition_penalty": 1.08,
                "pad_token_id": pad_token_id,
            }
            # temperature는 do_sample=True일 때만 유효 (do_sample=False면 제거)
            # Gemma 모델의 경우 temperature 파라미터가 무시될 수 있으므로 제거
            
            print(f"[생성 시작] max_new_tokens={max_new_tokens}, do_sample={generation_kwargs['do_sample']}")
            
            # 프롬프트 길이 저장 (에코 제거용)
            prompt_len = tensor_inputs["input_ids"].shape[1]
            
            with torch.inference_mode():
                try:
                    gen_output = self.model.generate(
                        **tensor_inputs,
                        **generation_kwargs,
                        return_dict_in_generate=True,
                    )
                    # 생성된 시퀀스에서 프롬프트 제외한 부분만 추출
                    if hasattr(gen_output, 'sequences'):
                        sequences = gen_output.sequences
                    else:
                        sequences = gen_output
                    
                    gen_ids = sequences[:, prompt_len:]  # 프롬프트 이후 부분만
                    print(f"[생성 완료] 생성된 토큰 수: {gen_ids.shape[1]}")
                except Exception as gen_exc:
                    print(f"[생성 에러] {type(gen_exc).__name__}: {gen_exc}")
                    raise

            # 토크나이저로 디코딩 (프롬프트 에코 제거된 부분만)
            tok = getattr(self.processor, "tokenizer", self.processor)
            if gen_ids.numel() == 0:
                return "[Error] Empty decode."
            
            decoded_text = tok.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
            
            print(f"[디코딩 결과] 원본 텍스트 길이: {len(decoded_text)} 문자")
            if decoded_text:
                print(f"[디코딩 결과] 원본 텍스트 미리보기:\n{decoded_text[:200]}...")
            
            # 응답 포맷팅: 처음 4줄만 선택 (비어있지 않은 라인만)
            if decoded_text:
                lines = decoded_text.split('\n')
                # 비어있지 않은 라인만 골라서 최대 4줄까지 선택
                clean_lines = [line.strip() for line in lines if line.strip()][:4]
                formatted_response = "\n".join(clean_lines)
                
                # 특수 토큰 제거 (<start_of_turn>, <end_of_turn> 등)
                formatted_response = formatted_response.replace("<start_of_turn>", "").replace("<end_of_turn>", "")
                formatted_response = formatted_response.replace("<start_of_turn>model", "").replace("model", "")
                formatted_response = formatted_response.replace("<start_of_turn>user", "").replace("user", "")
                
                # 앞뒤 공백 제거
                formatted_response = formatted_response.strip()
                
                # 최종 응답 로그 출력
                print("\n" + "="*80)
                print("[생성된 답변]")
                print("="*80)
                print(formatted_response)
                print("="*80 + "\n")
                
                return formatted_response if formatted_response else "[Error] Empty decode."
            
            print("[경고] 디코딩된 텍스트가 비어있습니다.")
            return "[Error] Empty decode."

        except Exception as exc:
            import traceback

            traceback.print_exc()
            return f"[Exception] {type(exc).__name__}: {exc}"

    @staticmethod
    def format_measurements(data: dict) -> str:
        if not data or not isinstance(data, dict):
            return "{}"
        items = {key: val for key, val in data.items() if isinstance(val, (int, float))}
        if not items:
            return "{}"
        lines = [f"  {key}: {value}" for key, value in items.items()]
        return "{\n" + "\n".join(lines) + "\n}"

    @staticmethod
    def format_rag_references(
        contents: List[str],
        max_items: int = 2,
        max_chars: int = 200,
    ) -> str:
        if not contents:
            return ""
        out: List[str] = []
        unique_contents = list(dict.fromkeys(contents))
        for text in unique_contents[:max_items]:
            text = (text or "").replace("\n", " ").strip()
            if not text:
                continue
            if len(text) > max_chars:
                text = text[:max_chars].rstrip() + "..."
            out.append(f"- {text}")
        return "\n\nRelevant References:\n" + "\n".join(out)

    @staticmethod
    def format_model_response(text: str) -> str:
        """
        모델 응답을 화면 표시용 형태로 변환합니다.
        예: 0:"교정: 전체\n발치: 44/44 발치\n추가시술: null\n치료 기간: 2년~2년 6개월"
        -> JSON 객체 형태로 변환
        """
        if not text:
            return ""
        
        import re
        import json
        
        # 1. 인덱스와 따옴표 패턴 제거 (예: 0:" 또는 1:" 등)
        text = re.sub(r'^\d+:"', '', text)
        # 2. 마지막 따옴표 제거
        text = re.sub(r'"$', '', text)
        text = text.strip()
        
        # 3. 줄바꿈으로 분리하여 파싱
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        result = {}
        for line in lines:
            # "키: 값" 형태로 파싱
            if ':' in line:
                parts = line.split(':', 1)  # 최대 1번만 분리 (값에 :가 있을 수 있음)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    
                    # null 값 처리
                    if value.lower() in ['null', 'none', '']:
                        value = ""
                    
                    result[key] = value
        
        # JSON 형태로 변환 (한글 키 지원)
        if result:
            return json.dumps(result, ensure_ascii=False, indent=2)
        else:
            # 파싱 실패 시 원본 텍스트 반환 (정리만)
            return text


