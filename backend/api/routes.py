"""API routes for the PetCare RAG backend."""

from fastapi import APIRouter, HTTPException

from backend.schemas.ask import AskRequest, AskResponse
from backend.services.rag_service import answer_question

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """รับคำถามจาก frontend แล้วส่งต่อให้ RAG ตอบ"""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="กรุณาพิมพ์คำถามก่อน")

    try:
        result = answer_question(question, top_k=3)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail="ยังไม่ได้ตั้งค่า GEMINI_API_KEY บน backend",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="ระบบตอบคำถามมีปัญหาชั่วคราว กรุณาลองใหม่อีกครั้ง",
        ) from exc

    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
    }
