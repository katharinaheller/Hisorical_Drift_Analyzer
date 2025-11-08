from __future__ import annotations
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

from src.core.llm.llm_orchestrator import LLMOrchestrator
from src.core.retrieval.retrieval_orchestrator import RetrievalOrchestrator

# ---------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------
app = FastAPI(title="HDA API", version="1.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str
    intent: str | None = "conceptual"
    return_context: bool | None = True

class RetrieveRequest(BaseModel):
    query: str
    intent: str | None = "conceptual"

class ChatResponse(BaseModel):
    answer: str
    retrieved: List[Dict[str, Any]] | None = None

# ---------------------------------------------------------------------
# Singletons (avoid reloading)
# ---------------------------------------------------------------------
_llm_orch: LLMOrchestrator | None = None
_ret_orch: RetrievalOrchestrator | None = None

def _llm() -> LLMOrchestrator:
    global _llm_orch
    if _llm_orch is None:
        _llm_orch = LLMOrchestrator()
        logger.info("LLMOrchestrator initialized.")
    return _llm_orch

def _retriever() -> RetrievalOrchestrator:
    global _ret_orch
    if _ret_orch is None:
        _ret_orch = RetrievalOrchestrator()
        logger.info("RetrievalOrchestrator initialized.")
    return _ret_orch

# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------
@app.post("/api/retrieve", response_model=List[Dict[str, Any]])
def retrieve(req: RetrieveRequest):
    """Retrieve ranked context chunks."""
    try:
        return _retriever().retrieve(req.query, req.intent or "conceptual")
    except Exception as e:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Simple non-streaming chat endpoint."""
    try:
        chunks = _retriever().retrieve(req.query, req.intent or "conceptual")
        if not chunks:
            return ChatResponse(answer="No relevant context found.", retrieved=[])
        answer = _llm().run_with_context(req.query, req.intent or "conceptual", chunks)
        return ChatResponse(answer=answer, retrieved=chunks if req.return_context else None)
    except Exception as e:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main():
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8001, reload=False)

if __name__ == "__main__":
    main()
