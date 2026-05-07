"""Service layer for loading and calling the PetCare RAG system."""

import os
import gc
from typing import Any

from rag.answer import PetRAG


# Global singleton — loaded lazily on first request
_rag_instance: PetRAG | None = None


def get_rag() -> PetRAG:
    """Load RAG once and reuse for all API requests (lazy loading)

    - If ONNX model exists → uses OnnxSearcher (saves ~650 MB RAM)
    - If not → uses SentenceTransformer as usual
    - Loads only on first request (lazy) to save RAM at startup
    """
    global _rag_instance
    if _rag_instance is not None:
        return _rag_instance

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    print("🔄 Loading RAG system (first request)...")
    _rag_instance = PetRAG(llm_provider="gemini", api_key=api_key)

    # Force garbage collection after loading
    gc.collect()

    return _rag_instance


def answer_question(question: str, top_k: int = 3) -> dict[str, Any]:
    """Send question to RAG and return raw result"""
    return get_rag().answer(question, top_k=top_k)
