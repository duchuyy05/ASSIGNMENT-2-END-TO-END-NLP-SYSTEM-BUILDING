from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)


class SourceDocument(BaseModel):
    rank: int | None = None
    score: float
    chunk_id: str | None = None
    document_id: str | None = None
    chunk_index: int | None = None
    title: str | None = None
    url: str | None = None
    category: str | None = None
    source_language: str | None = None
    qa_language: str | None = None
    word_count: int | None = None
    text: str | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    sources: list[SourceDocument]


class RetrieveResponse(BaseModel):
    question: str
    sources: list[SourceDocument]
