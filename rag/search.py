"""
rag/search.py
=============
ฟังก์ชันค้นหา (Semantic Search) ด้วย FAISS

ขั้นตอน:
1. โหลด FAISS index + chunk metadata
2. แปลงคำถามเป็น embedding
3. ค้นหา top-k chunks ที่ใกล้เคียงที่สุด

วิธีใช้:
    from rag.search import PetSearcher

    searcher = PetSearcher()
    results = searcher.search("แมวพันธุ์ไหนเหมาะกับคนอยู่คนเดียว", top_k=3)

    for r in results:
        print(f"[{r['score']:.3f}] {r['breed_name']} - {r['text'][:100]}...")

Note:
    - ถ้ามี ONNX model files ใน rag/model_onnx/ → ใช้ OnnxSearcher
      (ประหยัด RAM ~650 MB เพราะไม่ต้องโหลด PyTorch)
    - ถ้าไม่มี → ใช้ SentenceTransformer ตามปกติ (สำหรับ dev ในเครื่อง)
    - ทั้งสองแบบใช้ model เดียวกัน: paraphrase-multilingual-MiniLM-L12-v2
      ดังนั้น embeddings จะเหมือนกัน และ FAISS index เดิมใช้ได้เลย
"""

import os
import pickle
import numpy as np

# ============================================
# Auto-detect backend: ONNX (deploy) or SentenceTransformer (dev)
# ============================================
ONNX_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model_onnx')
ONNX_MODEL_FILE = os.path.join(ONNX_MODEL_DIR, 'model_int8.onnx')

# Check if ONNX model exists → use lightweight ONNX backend
_USE_ONNX = os.path.exists(ONNX_MODEL_FILE)


# ============================================
# Config
# ============================================
INDEX_DIR = os.path.join(os.path.dirname(__file__), 'index')
EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'


# ============================================
# PetSearcher Class
# ============================================
class PetSearcher:
    """ระบบค้นหาสายพันธุ์สัตว์เลี้ยงด้วย Semantic Search

    Auto-selects backend:
    - ONNX Runtime (if model_int8.onnx exists) → ~150 MB RAM
    - SentenceTransformer (fallback) → ~800 MB RAM
    """

    def __init__(self, index_dir=INDEX_DIR, model_name=EMBEDDING_MODEL):
        self.index_dir = index_dir
        self.model_name = model_name
        self.index = None
        self.chunks = None
        self.model = None
        self._onnx_searcher = None

        self._load()

    def _load(self):
        """โหลด FAISS index, chunks, และ embedding model"""
        # โหลด FAISS index
        import faiss

        index_path = os.path.join(self.index_dir, 'faiss_index.index')
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"❌ ไม่พบ FAISS index ที่ {index_path}\n"
                f"   กรุณารัน 'python rag/build_index.py' ก่อน"
            )

        self.index = faiss.read_index(index_path)
        print(f"✅ โหลด FAISS index: {self.index.ntotal} vectors")

        # โหลด chunk metadata
        chunks_path = os.path.join(self.index_dir, 'chunks.pkl')
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        print(f"✅ โหลด chunk metadata: {len(self.chunks)} chunks")

        # โหลด embedding model — เลือก backend อัตโนมัติ
        if _USE_ONNX:
            print(f"🔄 กำลังโหลด model (ONNX Runtime): {self.model_name}...")
            from rag.onnx_searcher import OnnxSearcher
            self._onnx_searcher = OnnxSearcher(
                model_dir=ONNX_MODEL_DIR,
                index_dir=self.index_dir,
            )
            # We already loaded FAISS + chunks above, so share them
            self._onnx_searcher.faiss_index = self.index
            self._onnx_searcher.chunks = self.chunks
            print(f"✅ โหลด ONNX model เสร็จ (ประหยัด RAM ~650 MB)")
        else:
            print(f"🔄 กำลังโหลด model (SentenceTransformer): {self.model_name}...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            print(f"✅ โหลด SentenceTransformer model เสร็จ")

    def search(self, query, top_k=3):
        """
        ค้นหา chunks ที่เกี่ยวข้องกับคำถาม

        Args:
            query: คำถาม/คำค้นหา (ภาษาไทยหรืออังกฤษ)
            top_k: จำนวนผลลัพธ์ที่ต้องการ

        Returns:
            list ของ dict:
            {
                "text": chunk text,
                "score": cosine similarity score,
                "metadata": {type, breed_name, source_url, ...}
            }
        """
        if self._onnx_searcher is not None:
            return self._onnx_searcher.search(query, top_k)

        # Fallback: SentenceTransformer
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )
        query_embedding = np.array(query_embedding).astype('float32')

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                continue

            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "score": float(score),
                "metadata": chunk["metadata"],
                "rank": i + 1
            })

        return results

    def search_with_context(self, query, top_k=3):
        """
        ค้นหาและรวม context สำหรับ RAG

        Returns:
            dict:
            {
                "query": คำถาม,
                "context": ข้อความรวมจาก top-k chunks,
                "sources": list ของ {breed_name, type, source_url},
                "results": raw search results
            }
        """
        if self._onnx_searcher is not None:
            return self._onnx_searcher.search_with_context(query, top_k)

        # Fallback: SentenceTransformer
        results = self.search(query, top_k)

        context_parts = []
        seen_breeds = set()
        sources = []

        for r in results:
            context_parts.append(f"[แหล่งข้อมูล {r['rank']}]\n{r['text']}")

            breed = r['metadata']['breed_name']
            if breed not in seen_breeds:
                seen_breeds.add(breed)
                sources.append({
                    "breed_name": breed,
                    "type": r['metadata']['type'],
                    "source_url": r['metadata']['source_url'],
                    "score": r['score']
                })

        context = "\n\n---\n\n".join(context_parts)

        return {
            "query": query,
            "context": context,
            "sources": sources,
            "results": results
        }


# ============================================
# CLI: ทดสอบการค้นหา
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("🔍 PetCare RAG - Semantic Search")
    print("=" * 60)

    if _USE_ONNX:
        print("📦 Backend: ONNX Runtime (lightweight)")
    else:
        print("📦 Backend: SentenceTransformer (full PyTorch)")

    searcher = PetSearcher()

    # คำถามทดสอบจาก assignment
    test_queries = [
        "แมวพันธุ์ไหนเหมาะกับคนอยู่คนเดียว",
        "สุนัขพันธุ์ไหนขนสั้นดูแลง่าย",
        "Persian มีนิสัยยังไง",
        "แมวขนยาวต้องดูแลอะไรบ้าง",
    ]

    for query in test_queries:
        print(f"\n{'─' * 60}")
        print(f"❓ คำถาม: {query}")
        print(f"{'─' * 60}")

        results = searcher.search(query, top_k=3)

        for r in results:
            breed = r['metadata']['breed_name']
            pet_type = r['metadata']['type']
            print(f"  #{r['rank']} [{r['score']:.3f}] {breed} ({pet_type})")
            print(f"     {r['text'][:120]}...")

    print(f"\n{'=' * 60}")
    print("🎉 ทดสอบการค้นหาเสร็จสมบูรณ์!")
