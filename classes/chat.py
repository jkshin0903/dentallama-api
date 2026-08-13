from pydantic import BaseModel
from typing import List, Optional


class MessagePart(BaseModel):
    type: str
    text: str


class ChatMessage(BaseModel):
    role: str
    content: str
    # parts: Optional[List[MessagePart]] = None


class FileData(BaseModel):
    name: str  # 파일명
    type: str  # 파일 타입
    size: int  # 파일 크기
    content: str  # base64로 인코딩된 파일 데이터


class ChatRequest(BaseModel):
    id: Optional[str] = None
    messages: List[ChatMessage]
    model: str
    files: Optional[List[FileData]] = None  # base64로 인코딩된 파일 리스트
