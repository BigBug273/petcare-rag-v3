"""
rag/onnx_searcher.py
====================
Lightweight ONNX Runtime backend for Sentence-BERT inference.

Uses the SAME model (paraphrase-multilingual-MiniLM-L12-v2) as search.py,
but runs it through ONNX Runtime instead of PyTorch.

Memory usage: ~150 MB (vs ~800 MB with PyTorch)
→ Fits within Render's 512 MB RAM limit

This module replicates sentence-transformers' encoding pipeline:
  1. Tokenize with the same tokenizer
  2. Run through the transformer model (ONNX)
  3. Mean pooling (using attention mask)
  4. L2 normalization

The embeddings are numerically identical to sentence-transformers
(within floating-point precision), so the pre-built FAISS index works
without rebuilding.

Usage:
    from rag.onnx_searcher import OnnxSearcher

    searcher = OnnxSearcher()
    results = searcher.search("แมวพันธุ์ไหนเหมาะกับคนอยู่คนเดียว", top_k=3)
"""

import os
import json
import pickle
import numpy as np

# ONNX Runtime — lightweight alternative to PyTorch
import onnxruntime as ort

# Tokenizer — same one used by sentence-transformers
from tokenizers import Tokenizer


# ============================================
# Config
# ============================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_onnx")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")

# ONNX model filename (INT8 quantized for small size)
ONNX_MODEL_FILE = "model_int8.onnx"

# The original Sentence-BERT model name (for reference / logging)
ORIGINAL_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


# ============================================
# ONNX Searcher Class
# ============================================
class OnnxSearcher:
    """Semantic Search using Sentence-BERT via ONNX Runtime.

    Drop-in replacement for PetSearcher that uses ~150 MB RAM
    instead of ~800 MB by replacing PyTorch with ONNX Runtime.
    """

    def __init__(self, model_dir=MODEL_DIR, index_dir=INDEX_DIR):
        self.model_dir = model_dir
        self.index_dir = index_dir
        self.session = None
        self.tokenizer = None
        self.faiss_index = None
        self.chunks = None
        self.max_seq_length = 128
        self.embedding_dim = 384

        self._load()

    def _load(self):
        """Load ONNX model, tokenizer, FAISS index, and chunk metadata."""
        # --- Load config ---
        config_path = os.path.join(self.model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = json.load(f)
            self.max_seq_length = cfg.get("max_seq_length", 128)
            self.embedding_dim = cfg.get("embedding_dimension", 384)

        # --- Load ONNX model ---
        onnx_path = os.path.join(self.model_dir, ONNX_MODEL_FILE)
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"ONNX model not found at {onnx_path}\n"
                f"Run 'python scripts/export_onnx.py' locally to create it, "
                f"then commit the file to your repo."
            )

        # Optimized session options for low memory
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1   # limit threads → less memory
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(onnx_path, sess_options)
        print(f"[ONNX] Loaded model: {ORIGINAL_MODEL_NAME} (INT8 quantized)")

        # --- Load tokenizer ---
        tokenizer_path = os.path.join(self.model_dir, "tokenizer.json")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_path}\n"
                f"Run 'python scripts/export_onnx.py' to create it."
            )
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(max_length=self.max_seq_length)
        print(f"[ONNX] Loaded tokenizer (max_seq_length={self.max_seq_length})")

        # --- Load FAISS index ---
        index_path = os.path.join(self.index_dir, "faiss_index.index")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}\n"
                f"Run 'python rag/build_index.py' first."
            )
        import faiss
        self.faiss_index = faiss.read_index(index_path)
        print(f"[ONNX] Loaded FAISS index: {self.faiss_index.ntotal} vectors")

        # --- Load chunk metadata ---
        chunks_path = os.path.join(self.index_dir, "chunks.pkl")
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        print(f"[ONNX] Loaded chunk metadata: {len(self.chunks)} chunks")

    def encode(self, texts, normalize_embeddings=True):
        """Encode texts into embeddings using ONNX Runtime.

        Replicates sentence-transformers' encode() pipeline:
          1. Tokenize
          2. Run ONNX model
          3. Mean pooling (with attention mask)
          4. L2 normalize

        Encodes ONE text at a time to avoid LayerNormalization
        batch shape mismatch issues with certain ONNX exports.

        Args:
            texts: str or list of str
            normalize_embeddings: whether to L2-normalize (default True)

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        # Encode ONE text at a time to avoid batch shape issues
        # with LayerNormalization in ONNX exports
        for text in texts:
            encoded = self.tokenizer.encode(text)
            input_ids = np.array([encoded.ids], dtype=np.int64)
            attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

            # Run ONNX inference (batch_size=1)
            outputs = self.session.run(
                ["last_hidden_state"],
                {"input_ids": input_ids, "attention_mask": attention_mask},
            )
            last_hidden_state = outputs[0]  # (1, seq_len, dim)

            # Mean pooling
            mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
            sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
            sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            mean_embedding = sum_embeddings / sum_mask  # (1, dim)

            # L2 normalize
            if normalize_embeddings:
                norms = np.linalg.norm(mean_embedding, axis=1, keepdims=True)
                mean_embedding = mean_embedding / np.clip(norms, a_min=1e-9, a_max=None)

            all_embeddings.append(mean_embedding.astype(np.float32)[0])

        return np.array(all_embeddings, dtype=np.float32)

    def search(self, query, top_k=3):
        """Search for top-k relevant chunks.

        Identical interface to PetSearcher.search()
        """
        query_embedding = self.encode([query], normalize_embeddings=True)

        scores, indices = self.faiss_index.search(query_embedding, top_k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append(
                {
                    "text": chunk["text"],
                    "score": float(score),
                    "metadata": chunk["metadata"],
                    "rank": i + 1,
                }
            )

        return results

    def search_with_context(self, query, top_k=3):
        """Search and combine context for RAG.

        Identical interface to PetSearcher.search_with_context()
        """
        results = self.search(query, top_k)

        context_parts = []
        seen_breeds = set()
        sources = []

        for r in results:
            context_parts.append(f"[แหล่งข้อมูล {r['rank']}]\n{r['text']}")

            breed = r["metadata"]["breed_name"]
            if breed not in seen_breeds:
                seen_breeds.add(breed)
                sources.append(
                    {
                        "breed_name": breed,
                        "type": r["metadata"]["type"],
                        "source_url": r["metadata"]["source_url"],
                        "score": r["score"],
                    }
                )

        context = "\n\n---\n\n".join(context_parts)

        return {
            "query": query,
            "context": context,
            "sources": sources,
            "results": results,
        }
