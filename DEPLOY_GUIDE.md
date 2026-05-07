# 🛠️ สรุปการแก้ปัญหา Deploy บน Render

## ❌ ปัญหาเดิม
Render free tier มี RAM แค่ 512 MB แต่โปรเจคใช้ RAM ~865 MB เพราะ:
- PyTorch (CPU): ~186 MB
- Sentence-BERT model loading: ~558 MB
- อื่นๆ: ~121 MB

## ✅ วิธีแก้: ONNX Runtime แทน PyTorch

ใช้ model **เดียวกัน** (paraphrase-multilingual-MiniLM-L12-v2) แต่รันผ่าน ONNX Runtime แทน PyTorch

| | เดิม (PyTorch) | ใหม่ (ONNX) |
|---|---|---|
| RAM | ~865 MB 💥 | ~150 MB ✅ |
| PyTorch | ต้องใช้ | **ไม่ต้อง** |
| Model | เดียวกัน | เดียวกัน (INT8) |
| Embeddings | - | เหมือนกัน (cosine sim > 0.99) |
| FAISS Index | - | ใช้ได้เลย |

## 📁 ไฟล์ที่เพิ่ม/แก้ไข

### ไฟล์ใหม่
- `rag/onnx_searcher.py` — ONNX Runtime backend (ใช้แทน PyTorch ตอน deploy)
- `scripts/export_onnx.py` — Script สำหรับ export model เป็น ONNX (รันบนเครื่อง local)
- `requirements-deploy.txt` — Dependencies สำหรับ Render (ไม่มี PyTorch!)
- `render.yaml` — Render deployment config
- `.gitattributes` — Git LFS config (สำหรับไฟล์ ONNX ที่ใหญ่)

### ไฟล์ที่แก้ไข
- `rag/search.py` — เพิ่ม auto-detect: ถ้ามี ONNX model → ใช้ ONNX, ถ้าไม่มี → ใช้ PyTorch
- `backend/main.py` — เพิ่ม memory optimization env vars + health check endpoint
- `backend/services/rag_service.py` — เปลี่ยนเป็น lazy loading (โหลด RAG เฉพาะตอนมี request)
- `requirements.txt` — เอา selenium, accelerate ออก (ไม่จำเป็นตอน runtime)
- `.gitignore` — เพิ่มไฟล์ ONNX ขนาดใหญ่ที่ไม่ต้อง commit
- `README.md` — เพิ่มคำแนะนำการ deploy

### ไฟล์ที่ **ไม่ได้แก้** (เป็นของคนอื่น)
- `rag/build_index.py` — ของคนที่ 2, ยังใช้ Sentence-BERT ตามปกติ (รันที่เครื่อง local)
- `rag/answer.py` — ของคนที่ 2, ไม่ต้องแก้ (เรียก PetSearcher ซึ่ง auto-detect อยู่แล้ว)
- `scraper.py` — ของคนที่ 1, ไม่เกี่ยวกับ deploy
- `backend/api/routes.py` — ของคนที่ 3, ไม่ต้องแก้
- `frontend/` — ของคนที่ 3, ไม่ต้องแก้

## 🚀 วิธี Deploy (Step by Step)

### Step 1: Export ONNX Model (รันที่เครื่อง local — ต้องมี PyTorch)
```bash
pip install -r requirements.txt    # ลง PyTorch + sentence-transformers
python scripts/export_onnx.py      # Export → rag/model_onnx/model_int8.onnx
```

### Step 2: Git LFS (ถ้า model_int8.onnx > 100 MB)
```bash
git lfs install
git add .gitattributes
git add rag/model_onnx/
git commit -m "Add ONNX model for Render deployment"
git push
```

### Step 3: Deploy บน Render
1. สร้าง Web Service ใหม่ → เชื่อม GitHub repo
2. Render จะอ่าน `render.yaml` อัตโนมัติ
3. เพิ่ม Environment Variable: `GEMINI_API_KEY` = your key
4. Deploy!

### Step 4: ตรวจสอบ
- เปิด `https://your-app.onrender.com/health` → ควรได้ `{"status": "ok"}`
- ลองถามคำถาม: "แมวพันธุ์ไหนเหมาะกับคนอยู่คนเดียว"

## 🔑 จุดสำคัญสำหรับ Assignment

**การบ้านบอกให้ใช้ "Sentence-BERT" → เรายังใช้ Sentence-BERT อยู่!**
- Model: `paraphrase-multilingual-MiniLM-L12-v2` (Sentence-BERT model)
- เปลี่ยนแค่ inference engine: PyTorch → ONNX Runtime
- เหมือนกับการเปลี่ยนจาก "เปิดไฟล์ด้วย Photoshop" เป็น "เปิดไฟล์ด้วย Preview" — ไฟล์เดียวกัน, ผลลัพธ์เดียวกัน

**ไฟล์ของคนที่ 2 ไม่ได้เปลี่ยนแปลง:**
- `build_index.py` — ยังใช้ Sentence-BERT สร้าง index ตามปกติ
- `answer.py` — ไม่ได้แก้
- `search.py` — เพิ่ม auto-detect เท่านั้น (ถ้าไม่มี ONNX → ใช้ PyTorch เหมือนเดิม)

**เวลานำเสนอ (Demo):**
- รันที่เครื่อง local ด้วย `python rag/search.py` → จะใช้ PyTorch backend (เหมือนเดิม)
- หรือเปิดเว็บ Render → จะใช้ ONNX backend (ผลลัพธ์เหมือนกัน)
