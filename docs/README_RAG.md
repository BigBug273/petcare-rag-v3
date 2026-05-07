# PetCare RAG - Model / AI Logic (คนที่ 2)

> **ส่วนของ:** คนที่ 2 - Model / AI Logic (ซซีาร์)
> **เป้าหมาย:** ทำระบบค้นหา + RAG (Retrieval-Augmented Generation)

---

## ภาพรวมระบบ

```
pet_breeds.csv
      │
      ▼
┌─────────────────┐
│  build_index.py │  สร้าง Vector Index
│  ─────────────  │
│  1. อ่าน CSV     │
│  2. สร้าง doc    │
│  3. Split chunk  │
│  4. Embedding    │
│  5. FAISS index  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   search.py     │  Semantic Search
│  ─────────────  │
│  1. โหลด index  │
│  2. Query embed  │
│  3. FAISS top-k  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   answer.py     │  RAG Q&A
│  ─────────────  │
│  1. ค้นหา ctx   │
│  2. สร้าง prompt │
│  3. เรียก LLM    │
│  4. คืนคำตอบ    │
└─────────────────┘
```

---

## โครงสร้างไฟล์

```
PetCare_Project/
├── data/
│   └── pet_breeds.csv          ← ข้อมูลจากคนที่ 1
├── rag/
│   ├── __init__.py
│   ├── build_index.py          ← สร้าง Vector Index
│   ├── search.py               ← Semantic Search
│   ├── answer.py               ← RAG Answer
│   └── index/                  ← (สร้างอัตโนมัติหลังรัน build_index.py)
│       ├── faiss_index.index
│       └── chunks.pkl
├── test_rag.py                 ← สคริปต์ทดสอบ
├── requirements.txt
└── scraper.py                  ← ของคนที่ 1
```

---

## วิธีติดตั้งและรัน (Step-by-Step)

### Step 1: สร้าง Virtual Environment

```bash
# สร้าง venv
python -m venv venv

# เปิดใช้งาน
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 2: ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

**Package หลักที่ใช้:**

| Package | หน้าที่ |
|---------|---------|
| `sentence-transformers` | สร้าง embedding vectors จากข้อความ |
| `faiss-cpu` | ค้นหา similarity ใน vector space |
| `transformers` | HuggingFace model backend |
| `google-generativeai` | เรียก Gemini API (optional) |
| `openai` | เรียก OpenAI/Typhoon API (optional) |

### Step 3: สร้าง Vector Index

```bash
python rag/build_index.py
```

**ผลลัพธ์ที่คาดหวัง:**
```
🐾 PetCare RAG - Build Vector Index
📂 กำลังอ่านไฟล์: data/pet_breeds.csv
✅ อ่านข้อมูลได้ 53 แถว
📝 สร้าง document ได้ 53 ฉบับ
✂️ แบ่งเป็น ~150 chunks
🔄 กำลังสร้าง embeddings...
✅ สร้าง embeddings เสร็จ: shape = (150, 384)
✅ FAISS index สร้างเสร็จ: 150 vectors
💾 บันทึก FAISS index ที่: rag/index/faiss_index.index
🎉 สร้าง Vector Index เสร็จสมบูรณ์!
```

### Step 4: ทดสอบ Semantic Search

```bash
python rag/search.py
```

**ผลลัพธ์:**
```
❓ คำถาม: แมวพันธุ์ไหนเหมาะกับคนอยู่คนเดียว
  #1 [0.65] บริติช ช็อตแฮร์ (cat)
  #2 [0.58] เอ็กซ์โซติก ช็อตแฮร์ (cat)
  #3 [0.52] แมวเปอร์เซีย (cat)
```

### Step 5: ทดสอบ RAG (พร้อม LLM)

**วิธีที่ 1: ใช้ Gemini**
```bash
set LLM_PROVIDER=gemini
set LLM_API_KEY=your_gemini_api_key
python rag/answer.py
```

**วิธีที่ 2: ใช้ OpenAI**
```bash
set LLM_PROVIDER=openai
set LLM_API_KEY=your_openai_api_key
python rag/answer.py
```

**วิธีที่ 3: ใช้ Typhoon (ภาษาไทยดี)**
```bash
set LLM_PROVIDER=typhoon
set LLM_API_KEY=your_typhoon_api_key
python rag/answer.py
```

**วิธีที่ 4: ใช้ Ollama (local, ฟรี)**
```bash
# ติดตั้ง Ollama ก่อน: https://ollama.ai
ollama pull llama3
python rag/answer.py
```

### Step 6: รัน Full Test

```bash
python test_rag.py
```

---

## อธิบาย Code เป็นรายข้อ

### 📄 `rag/build_index.py`

**หน้าที่:** สร้าง Vector Index จาก CSV

**ขั้นตอนภายใน:**

1. **`load_data()`** - อ่าน `pet_breeds.csv` ด้วย pandas
2. **`create_documents(df)`** - รวมแต่ละแถวเป็น document รูปแบบ:
   ```
   ประเภท: cat
   สายพันธุ์: Persian
   รายละเอียด: ...ข้อความยาว...
   แหล่งที่มา: https://...
   ```
3. **`split_text()`** - แบ่ง document เป็น chunk ขนาด ~500 ตัวอักษร มี overlap 100 ตัวอักษร
   - พยายามตัดที่จุดขึ้นบรรทัดใหม่หรือช่องว่าง
   - ข้าม chunk ที่สั้นกว่า 30 ตัวอักษร
4. **`create_embeddings()`** - แปลง chunk text → vector ด้วย `paraphrase-multilingual-MiniLM-L12-v2`
   - Model นี้รองรับภาษาไทย
   - Normalize embeddings เพื่อใช้ cosine similarity
5. **`build_faiss_index()`** - สร้าง FAISS IndexFlatIP (Inner Product = cosine similarity)
6. **`save_index()`** - บันทึก index + metadata

**Config ที่ปรับได้:**

| Parameter | Default | คำอธิบาย |
|-----------|---------|----------|
| `CHUNK_SIZE` | 500 | ความยาว chunk (ตัวอักษร) |
| `CHUNK_OVERLAP` | 100 | ส่วนทับซ้อน |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Model สำหรับ embedding |

---

### 📄 `rag/search.py`

**หน้าที่:** ค้นหา chunks ที่เกี่ยวข้องกับคำถาม

**Class: `PetSearcher`**

**Methods:**
- **`search(query, top_k=3)`** - ค้นหา top-k chunks
  - แปลงคำถามเป็น embedding
  - ค้นใน FAISS index
  - คืน list ของ {text, score, metadata, rank}

- **`search_with_context(query, top_k=3)`** - ค้นหา + รวม context สำหรับ RAG
  - คืน dict: {query, context, sources, results}
  - ลบสายพันธุ์ซ้ำใน sources

**ตัวอย่างการใช้:**
```python
from rag.search import PetSearcher

searcher = PetSearcher()

# ค้นหาแบบง่าย
results = searcher.search("แมวขนยาวนิสัยเงียบ", top_k=3)

# ค้นหา + รวม context สำหรับ RAG
search_result = searcher.search_with_context("แมวขนยาว", top_k=3)
print(search_result["context"])    # ข้อความรวม
print(search_result["sources"])    # sources ที่ใช้
```

---

### 📄 `rag/answer.py`

**หน้าที่:** ระบบ RAG - ค้นหา context + ถาม LLM

**Class: `PetRAG`**

**Methods:**
- **`answer(query, top_k=3)`** - ตอบคำถามด้วย RAG (เรียก LLM)
- **`answer_search_only(query, top_k=3)`** - ตอบคำถามโดยใช้ search อย่างเดียว (ไม่ต้องใช้ API key)

**LLM Providers ที่รองรับ:**

| Provider | ต้องติดตั้งเพิ่ม | API Key |
|----------|---------------|---------|
| `gemini` | `pip install google-generativeai` | Google AI Studio |
| `openai` | `pip install openai` | OpenAI Platform |
| `typhoon` | `pip install openai` | OpenTyphoon |
| `ollama` | ติดตั้ง Ollama | ไม่ต้อง (local) |

**RAG Prompt Template:**
```
คุณเป็นผู้ช่วยให้ข้อมูลเกี่ยวกับสายพันธุ์สัตว์เลี้ยง

กฎสำคัญ:
1. ตอบคำถามโดยใช้ข้อมูลจาก context เท่านั้น
2. ถ้าไม่มีข้อมูล ให้บอกว่า "ไม่มีข้อมูลในฐานข้อมูล"
3. ห้ามแต่งข้อมูล (no hallucination)
4. ตอบเป็นภาษาไทย
5. อ้างอิงชื่อสายพันธุ์และแหล่งข้อมูล

Context: {context}
คำถาม: {query}
```

---

## คำถามทดสอบ (จาก Assignment)

| # | คำถาม | ประเภทที่คาดหวัง |
|---|-------|------------------|
| 1 | แมวพันธุ์ไหนเหมาะกับคนอยู่คนเดียว | cat |
| 2 | สุนัขพันธุ์ไหนขนสั้นดูแลง่าย | dog |
| 3 | Persian มีนิสัยยังไง | cat |
| 4 | แมวขนยาวต้องดูแลอะไรบ้าง | cat |

---

## ส่งงาน (ไฟล์ที่ต้องส่งให้ทีม)

```
rag/
├── build_index.py     ✅
├── search.py          ✅
└── answer.py          ✅
```

---

## คำอธิบายสำหรับนำเสนอ (Demo Script)

### สไลด์ที่ 1: ภาพรวมระบบ
> "ส่วนของผมรับหน้าที่ทำระบบ Model / AI Logic ครับ คือส่วนที่เอาข้อมูลจากคนที่ 1 มาประมวลผล แล้วทำให้ user ถาม-ตอบได้"

### สไลด์ที่ 2: Data Pipeline
> "เริ่มจากอ่าน CSV → แปลงเป็น Document → Split เป็น Chunk → Embed ด้วย Sentence-BERT → เก็บใน FAISS"

### สไลด์ที่ 3: Semantic Search
> "ตอน user ถาม เราจะแปลงคำถามเป็น embedding แล้วไปค้นหาใน FAISS ว่า chunk ไหนใกล้เคียงที่สุด"

### สไลด์ที่ 4: RAG
> "จากนั้นเอา context ที่ได้ ยัดเข้า prompt แล้วส่งให้ LLM ตอบ โดยบังคับให้ตอบจาก context เท่านั้น ไม่ให้แต่งเอง"

### สไลด์ที่ 5: Demo
> (รัน demo สด ถามคำถามทดสอบ 4 ข้อ)

---

## Troubleshooting

| ปัญหา | วิธีแก้ |
|-------|--------|
| `ModuleNotFoundError: sentence_transformers` | `pip install sentence-transformers` |
| `ModuleNotFoundError: faiss` | `pip install faiss-cpu` |
| โหลด model นานมาก | ครั้งแรกจะดาวน์โหลด ~480MB, ครั้งต่อไปเร็ว |
| FAISS index ไม่พบ | รัน `python rag/build_index.py` ก่อน |
| LLM ไม่ตอบ | เช็ค API key / ลองใช้ search-only mode |
| ภาษาไทย embedding ไม่ดี | ลองเปลี่ยน model เป็น `distiluse-base-multilingual-cased-v2` |

---

## แนวทางปรับปรุง (ถ้ามีเวลา)

1. **เพิ่ม metadata filtering** - ค้นหาเฉพาะ cat หรือ dog
2. **ใช้ ChromaDB แทน FAISS** - มี metadata filtering ในตัว
3. **Hybrid search** - รวม keyword search + semantic search
4. **Reranking** - ใช้ cross-encoder rerank ผลลัพธ์
5. **Conversation memory** - จำ context ของคำถามก่อนหน้า
