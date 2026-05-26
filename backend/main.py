from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import AskRequest, AskResponse, RetrieveResponse
from backend.service import RAGService


app = FastAPI(
    title="VNU/UET RAG Assignment Backend",
    description="MiniLM + FAISS + DistilBERT QA API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_service: RAGService | None = None


@app.on_event("startup")
def startup_event():
    global rag_service
    rag_service = RAGService()


def get_service() -> RAGService:
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service is not loaded")
    return rag_service


@app.get("/")
def root():
    return {
        "message": "VNU/UET RAG backend is running",
        "pipeline": "Question -> MiniLM -> FAISS -> DistilBERT QA -> Answer",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    return get_service().ask(question=request.question, top_k=request.top_k)


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: AskRequest):
    return get_service().retrieve(question=request.question, top_k=request.top_k)
