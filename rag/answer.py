"""
rag/answer.py
=============
ระบบ RAG (Retrieval-Augmented Generation)

ขั้นตอน:
1. รับคำถามจาก user
2. ค้นหา context ที่เกี่ยวข้อง (ผ่าน search.py)
3. สร้าง prompt สำหรับ RAG
4. เรียก LLM (Gemini / Typhoon / OpenAI)
5. คืนคำตอบ + source

วิธีใช้:
    from rag.answer import PetRAG

    rag = PetRAG(llm_provider="gemini", api_key=os.getenv("GEMINI_API_KEY"))
    result = rag.answer("แมวพันธุ์ไหนเหมาะกับคนอยู่คนเดียว")

    print(result["answer"])
    print(result["sources"])
"""

import os
import sys

# แก้ปัญหา import ไม่เจอเวลารันไฟล์ตรงจาก rag/ โฟลเดอร์
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rag.search import PetSearcher


# ============================================
# RAG Prompt Template
# ============================================
RAG_PROMPT_TEMPLATE = """คุณเป็นผู้ช่วยให้ข้อมูลเกี่ยวกับสายพันธุ์สัตว์เลี้ยง (แมวและสุนัข) จากฐานข้อมูล Purina

**กฎสำคัญ:**
1. ตอบคำถามโดยใช้ข้อมูลจาก context ด้านล่างเท่านั้น
2. ถ้าไม่มีข้อมูลใน context ให้บอกว่า "ไม่มีข้อมูลในฐานข้อมูล"
3. ห้ามแต่งข้อมูลขึ้นมาเอง (no hallucination)
4. ตอบเป็นภาษาไทย สุภาพและเป็นมิตร
5. หากมีหลายสายพันธุ์ที่เกี่ยวข้อง ให้กล่าวถึงทุกสายพันธุ์
6. อ้างอิงชื่อสายพันธุ์และแหล่งข้อมูลที่มาด้วย

---

**Context จากฐานข้อมูล:**

{context}

---

**คำถาม:** {query}

**คำตอบ:**"""


# ============================================
# LLM Providers
# ============================================
def call_gemini(api_key, prompt, model="gemini-2.5-flash"):
    """เรียก Google Gemini API (ใช้ google-genai ใหม่, fallback เป็นตัวเก่า)"""
    # ลองใช้ google-genai (ใหม่) ก่อน
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )
        return response.text
    except ImportError:
        pass

    # Fallback: ใช้ google.generativeai (เก่า)
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        genai_model = genai.GenerativeModel(model)
        response = genai_model.generate_content(prompt)
        return response.text
    except Exception as e:
        # ถ้า quota เต็ม ลองเปลี่ยน model
        if "429" in str(e) or "quota" in str(e).lower():
            print(f"  ⚠️  model '{model}' quota เต็ม กำลังลองใช้ gemini-1.5-flash...")
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            genai_model = genai.GenerativeModel("gemini-1.5-flash")
            response = genai_model.generate_content(prompt)
            return response.text
        raise


def call_openai(api_key, prompt, model="gpt-4o-mini", base_url=None):
    """เรียก OpenAI-compatible API (รองรับ Typhoon และอื่นๆ)"""
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=base_url  # None = OpenAI ปกติ
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "คุณเป็นผู้ช่วยให้ข้อมูลสายพันธุ์สัตว์เลี้ยง"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # ต่ำเพื่อลด hallucination
        max_tokens=1024
    )

    return response.choices[0].message.content


def call_typhoon(api_key, prompt, model="typhoon-v2-70b"):
    """เรียก Typhoon API (OpenAI-compatible)"""
    return call_openai(
        api_key=api_key,
        prompt=prompt,
        model=model,
        base_url="https://api.opentyphoon.ai/v1"
    )


def call_ollama(prompt, model="llama3", base_url="http://localhost:11434"):
    """เรียก Ollama local LLM (ไม่ต้องใช้ API key)"""
    import requests

    response = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]


# ============================================
# PetRAG Class
# ============================================
class PetRAG:
    """ระบบ RAG สำหรับ PetCare Chatbot"""

    def __init__(self, llm_provider="gemini", api_key=None, model=None):
        """
        Args:
            llm_provider: "gemini" | "openai" | "typhoon" | "ollama"
            api_key: API key สำหรับ LLM provider
            model: ชื่อ model (optional)
        """
        self.llm_provider = llm_provider
        if llm_provider == "gemini":
            self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        else:
            self.api_key = api_key or os.getenv(f"{llm_provider.upper()}_API_KEY", "")
        self.model = model

        # โหลด Searcher
        print("🔄 กำลังโหลด RAG system...")
        self.searcher = PetSearcher()
        print("✅ RAG system พร้อมใช้งาน")

    def _call_llm(self, prompt):
        """เรียก LLM ตาม provider ที่เลือก"""
        if self.llm_provider == "gemini":
            model = self.model or "gemini-2.5-flash"
            return call_gemini(self.api_key, prompt, model)

        elif self.llm_provider == "openai":
            model = self.model or "gpt-4o-mini"
            return call_openai(self.api_key, prompt, model)

        elif self.llm_provider == "typhoon":
            model = self.model or "typhoon-v2-70b"
            return call_typhoon(self.api_key, prompt, model)

        elif self.llm_provider == "ollama":
            model = self.model or "llama3"
            return call_ollama(prompt, model)

        else:
            raise ValueError(f"❌ ไม่รองรับ provider: {self.llm_provider}")

    def answer(self, query, top_k=3):
        """
        ตอบคำถามด้วย RAG

        Args:
            query: คำถามจาก user
            top_k: จำนวน context chunks ที่จะดึงมา

        Returns:
            dict:
            {
                "query": คำถาม,
                "answer": คำตอบจาก LLM,
                "sources": list ของ {breed_name, type, source_url, score},
                "context_used": context ที่ส่งให้ LLM
            }
        """
        # Step 1: ค้นหา context
        search_result = self.searcher.search_with_context(query, top_k)

        # Step 2: สร้าง RAG prompt
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=search_result["context"],
            query=query
        )

        # Step 3: เรียก LLM
        print(f"🔄 กำลังเรียก {self.llm_provider} LLM...")
        answer_text = self._call_llm(prompt)

        return {
            "query": query,
            "answer": answer_text,
            "sources": search_result["sources"],
            "context_used": search_result["context"]
        }

    def answer_search_only(self, query, top_k=3):
        """
        ตอบคำถามโดยใช้เฉพาะ search (ไม่เรียก LLM)
        เหมาะสำหรับทดสอบโดยไม่ต้องใช้ API key

        Returns:
            dict เช่นเดียวกับ answer() แต่ answer เป็น context รวม
        """
        search_result = self.searcher.search_with_context(query, top_k)

        # สร้างคำตอบจาก context โดยไม่ผ่าน LLM
        answer_parts = []
        for r in search_result["results"]:
            breed = r['metadata']['breed_name']
            answer_parts.append(f"📖 {breed} (ความเกี่ยวข้อง: {r['score']:.1%}):\n{r['text'][:300]}...")

        answer_text = "\n\n".join(answer_parts)

        return {
            "query": query,
            "answer": f"[Search-only mode - ไม่ได้ใช้ LLM]\n\n{answer_text}",
            "sources": search_result["sources"],
            "context_used": search_result["context"]
        }


# ============================================
# CLI: ทดสอบ RAG
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("🐾 PetCare RAG - Q&A System")
    print("=" * 60)

    # ตรวจสอบ API key
    provider = "gemini"
    api_key = os.getenv("GEMINI_API_KEY", "")

    print(f"\n📋 LLM Provider: {provider}")
    print(f"📋 API Key: {'✅ มี' if api_key else '❌ ไม่มี (ใช้ search-only mode)'}")

    if api_key:
        rag = PetRAG(llm_provider=provider, api_key=api_key)
    else:
        print("⚠️  ไม่พบ API key → ใช้ search-only mode (ไม่เรียก LLM)")
        rag = PetRAG(llm_provider="ollama")

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

        if api_key:
            result = rag.answer(query)
        else:
            result = rag.answer_search_only(query)

        print(f"\n💬 คำตอบ:\n{result['answer']}")

        print(f"\n📚 Sources:")
        for src in result['sources']:
            print(f"   - {src['breed_name']} ({src['type']}) | score: {src['score']:.3f}")
            print(f"     {src['source_url']}")

    print(f"\n{'=' * 60}")
    print("🎉 ทดสอบเสร็จสมบูรณ์!")
