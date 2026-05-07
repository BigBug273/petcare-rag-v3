"""FastAPI app for the PetCare RAG demo."""

import os

# Memory optimization — set BEFORE importing any ML libraries
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router
from backend.core.config import FRONTEND_STATIC_DIR, INDEX_HTML

app = FastAPI(title="PetCare RAG Assistant")
app.include_router(router)
app.mount("/static", StaticFiles(directory=FRONTEND_STATIC_DIR), name="static")


@app.get("/")
def home():
    """Serve the main web page"""
    return FileResponse(INDEX_HTML)


@app.get("/health")
def health_check():
    """Health check endpoint for Render — lightweight, no model loading"""
    onnx_available = os.path.exists(
        os.path.join(os.path.dirname(__file__), "..", "rag", "model_onnx", "model_int8.onnx")
    )
    return JSONResponse({
        "status": "ok",
        "backend": "onnx" if onnx_available else "pytorch",
    })
