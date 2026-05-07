"""
rag/build_index.py
==================
สร้าง Vector Index จาก pet_breeds.csv

ขั้นตอน:
1. อ่านไฟล์ pet_breeds.csv
2. รวมแต่ละแถวเป็น document (ข้อความเดียว)
3. split document เป็น chunk
4. ทำ embedding ด้วย Sentence-BERT
5. เก็บลง FAISS index

วิธีใช้:
    python rag/build_index.py

ผลลัพธ์:
    rag/index/faiss_index.index     - FAISS index file
    rag/index/chunks.pkl            - chunk metadata (ข้อความ + source)
"""

import os
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# ============================================
# Config
# ============================================
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'pet_breeds.csv')
INDEX_DIR = os.path.join(os.path.dirname(__file__), 'index')
CHUNK_SIZE = 500       # จำนวนตัวอักษรต่อ chunk
CHUNK_OVERLAP = 100    # ตัวอักษรทับซ้อนระหว่าง chunk
EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'  # รองรับภาษาไทย


# ============================================
# Step 1: อ่าน CSV
# ============================================
def load_data(csv_path=CSV_PATH):
    """อ่านไฟล์ pet_breeds.csv"""
    print(f"📂 กำลังอ่านไฟล์: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"✅ อ่านข้อมูลได้ {len(df)} แถว")
    print(f"   - แมว: {len(df[df.type == 'cat'])} สายพันธุ์")
    print(f"   - สุนัข: {len(df[df.type == 'dog'])} สายพันธุ์")

    return df


# ============================================
# Step 2: สร้าง Document จากแต่ละแถว
# ============================================
def create_documents(df):
    """
    รวมแต่ละแถวเป็น document เช่น:
    ประเภท: cat
    สายพันธุ์: Persian
    รายละเอียด: ...
    นิสัย: ...
    การดูแล: ...
    """
    documents = []

    for _, row in df.iterrows():
        doc = (
            f"ประเภท: {row['type']}\n"
            f"สายพันธุ์: {row['breed_name']}\n"
            f"รายละเอียด: {row['full_text']}\n"
            f"แหล่งที่มา: {row['source_url']}"
        )
        documents.append({
            "text": doc,
            "metadata": {
                "type": row['type'],
                "breed_name": row['breed_name'],
                "source_url": row['source_url']
            }
        })

    print(f"📝 สร้าง document ได้ {len(documents)} ฉบับ")
    return documents


# ============================================
# Step 3: Split Document เป็น Chunk
# ============================================
def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    แบ่งข้อความเป็น chunk โดยมี overlap

    Args:
        text: ข้อความต้นฉบับ
        chunk_size: ความยาว chunk (ตัวอักษร)
        overlap: ส่วนทับซ้อน (ตัวอักษร)

    Returns:
        list ของ chunk
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # พยายามตัดที่จุดขึ้นบรรทัดใหม่หรือจุด ถ้าได้
        if end < len(text):
            # หาจุดตัดที่เหมาะสม (ขึ้นบรรทัดใหม่หรือจุดที่ใกล้ที่สุด)
            for sep in ['\n', ' ']:
                last_sep = chunk.rfind(sep)
                if last_sep > chunk_size * 0.5:  # ตัดเฉพาะถ้าไม่สั้นเกินไป
                    chunk = chunk[:last_sep]
                    break

        chunks.append(chunk.strip())
        start += chunk_size - overlap

    return chunks


def create_chunks(documents):
    """
    แบ่งทุก document เป็น chunk พร้อมเก็บ metadata

    Returns:
        list ของ {"text": ..., "metadata": ...}
    """
    all_chunks = []

    for doc in documents:
        text = doc["text"]
        metadata = doc["metadata"]
        chunks = split_text(text)

        for i, chunk in enumerate(chunks):
            if len(chunk) < 30:  # ข้าม chunk ที่สั้นเกินไป
                continue

            all_chunks.append({
                "text": chunk,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })

    print(f"✂️ แบ่งเป็น {len(all_chunks)} chunks (จาก {len(documents)} documents)")
    return all_chunks


# ============================================
# Step 4: Embedding ด้วย Sentence-BERT
# ============================================
def create_embeddings(chunks, model_name=EMBEDDING_MODEL):
    """
    สร้าง embedding vectors จาก chunk texts

    Args:
        chunks: list ของ {"text": ..., "metadata": ...}
        model_name: ชื่อ Sentence-BERT model

    Returns:
        embeddings: numpy array (n_chunks, dim)
        model: loaded model object
    """
    print(f"🔄 กำลังโหลด model: {model_name}")
    print("   (โหลดครั้งแรกจะใช้เวลาดาวน์โหลด ~480MB)")

    model = SentenceTransformer(model_name)

    texts = [chunk["text"] for chunk in chunks]
    print(f"🔄 กำลังสร้าง embeddings สำหรับ {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True  # ทำให้ใช้ cosine similarity ได้ง่าย
    )

    embeddings = np.array(embeddings).astype('float32')
    print(f"✅ สร้าง embeddings เสร็จ: shape = {embeddings.shape}")

    return embeddings, model


# ============================================
# Step 5: เก็บลง FAISS
# ============================================
def build_faiss_index(embeddings):
    """
    สร้าง FAISS index จาก embeddings

    ใช้ IndexFlatIP (Inner Product) เพราะ embeddings ถูก normalize แล้ว
    = cosine similarity
    """
    dimension = embeddings.shape[1]
    print(f"🔨 กำลังสร้าง FAISS index (dimension={dimension})...")

    # IndexFlatIP = cosine similarity (เมื่อ vectors ถูก normalize)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print(f"✅ FAISS index สร้างเสร็จ: {index.ntotal} vectors")
    return index


def save_index(index, chunks, index_dir=INDEX_DIR):
    """บันทึก FAISS index และ chunk metadata"""
    os.makedirs(index_dir, exist_ok=True)

    # บันทึก FAISS index
    index_path = os.path.join(index_dir, 'faiss_index.index')
    faiss.write_index(index, index_path)
    print(f"💾 บันทึก FAISS index ที่: {index_path}")

    # บันทึก chunk metadata
    chunks_path = os.path.join(index_dir, 'chunks.pkl')
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"💾 บันทึก chunk metadata ที่: {chunks_path}")


# ============================================
# Main
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("🐾 PetCare RAG - Build Vector Index")
    print("=" * 60)

    # Step 1: อ่าน CSV
    df = load_data()

    # Step 2: สร้าง Documents
    documents = create_documents(df)

    # Step 3: Split เป็น Chunks
    chunks = create_chunks(documents)

    # Step 4: Embedding
    embeddings, model = create_embeddings(chunks)

    # Step 5: สร้าง FAISS Index
    index = build_faiss_index(embeddings)

    # Step 6: บันทึก
    save_index(index, chunks)

    print("\n" + "=" * 60)
    print("🎉 สร้าง Vector Index เสร็จสมบูรณ์!")
    print(f"   - Documents: {len(documents)}")
    print(f"   - Chunks: {len(chunks)}")
    print(f"   - Embedding dimension: {embeddings.shape[1]}")
    print(f"   - Model: {EMBEDDING_MODEL}")
    print("=" * 60)
