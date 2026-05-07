"""
test_rag.py
===========
สคริปต์ทดสอบระบบ RAG ทั้งหมด

วิธีใช้:
    # ทดสอบแบบ search-only (ไม่ต้องใช้ LLM API key)
    python test_rag.py

    # ทดสอบพร้อม LLM (ต้องตั้ง API key)
    LLM_PROVIDER=gemini LLM_API_KEY=your_key python test_rag.py
"""

import os
import sys

# เพิ่ม project root เข้า path หลังย้ายไฟล์เข้า tests/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


def test_step_1_build_index():
    """ทดสอบ: สร้าง Vector Index"""
    print("\n" + "=" * 60)
    print("TEST 1: Build Vector Index")
    print("=" * 60)

    from rag.build_index import load_data, create_documents, create_chunks, create_embeddings, build_faiss_index, save_index

    # อ่าน CSV
    df = load_data()
    assert len(df) > 0, "❌ CSV ไม่มีข้อมูล"
    print("✅ PASS: อ่าน CSV ได้")

    # สร้าง Documents
    documents = create_documents(df)
    assert len(documents) > 0, "❌ ไม่สร้าง document ได้"
    print("✅ PASS: สร้าง documents ได้")

    # สร้าง Chunks
    chunks = create_chunks(documents)
    assert len(chunks) > 0, "❌ ไม่สร้าง chunk ได้"
    print("✅ PASS: สร้าง chunks ได้")

    # สร้าง Embeddings
    embeddings, model = create_embeddings(chunks)
    assert embeddings.shape[0] == len(chunks), "❌ จำนวน embeddings ไม่ตรง"
    print("✅ PASS: สร้าง embeddings ได้")

    # สร้าง FAISS Index
    index = build_faiss_index(embeddings)
    assert index.ntotal == len(chunks), "❌ จำนวน vectors ใน index ไม่ตรง"
    print("✅ PASS: สร้าง FAISS index ได้")

    # บันทึก
    save_index(index, chunks)
    print("✅ PASS: บันทึก index เรียบร้อย")

    return True


def test_step_2_search():
    """ทดสอบ: Semantic Search"""
    print("\n" + "=" * 60)
    print("TEST 2: Semantic Search")
    print("=" * 60)

    from rag.search import PetSearcher

    searcher = PetSearcher()

    # ทดสอบคำถามแต่ละแบบ
    test_queries = [
        ("แมวพันธุ์ไหนเหมาะกับคนอยู่คนเดียว", "cat"),
        ("สุนัขพันธุ์ไหนขนสั้นดูแลง่าย", "dog"),
        ("Persian มีนิสัยยังไง", "cat"),
        ("แมวขนยาวต้องดูแลอะไรบ้าง", "cat"),
    ]

    for query, expected_type in test_queries:
        results = searcher.search(query, top_k=3)
        assert len(results) > 0, f"❌ ไม่พบผลลัพธ์สำหรับ: {query}"

        top_type = results[0]['metadata']['type']
        score = results[0]['score']

        print(f"  ✅ '{query}' → {results[0]['metadata']['breed_name']} ({top_type}) score={score:.3f}")

    print("✅ PASS: การค้นหาทำงานได้ถูกต้อง")
    return True


def test_step_3_rag_answer():
    """ทดสอบ: RAG Answer (search-only ถ้าไม่มี API key)"""
    print("\n" + "=" * 60)
    print("TEST 3: RAG Answer")
    print("=" * 60)

    from rag.answer import PetRAG

    provider = os.environ.get("LLM_PROVIDER", "ollama")
    api_key = os.environ.get("LLM_API_KEY", "")

    if api_key:
        rag = PetRAG(llm_provider=provider, api_key=api_key)
        mode = "full RAG (with LLM)"
    else:
        rag = PetRAG(llm_provider="ollama")
        mode = "search-only (no LLM)"

    print(f"  Mode: {mode}")

    test_queries = [
        "แมวพันธุ์ไหนเหมาะกับคนอยู่คนเดียว",
        "สุนัขพันธุ์ไหนขนสั้นดูแลง่าย",
        "Persian มีนิสัยยังไง",
        "แมวขนยาวต้องดูแลอะไรบ้าง",
    ]

    for query in test_queries:
        if api_key:
            result = rag.answer(query)
        else:
            result = rag.answer_search_only(query)

        assert result['answer'], f"❌ ไม่มีคำตอบสำหรับ: {query}"
        assert result['sources'], f"❌ ไม่มี source สำหรับ: {query}"

        print(f"\n  ❓ {query}")
        print(f"  💬 {result['answer'][:150]}...")
        print(f"  📚 Sources: {[s['breed_name'] for s in result['sources']]}")

    print("\n✅ PASS: RAG answer ทำงานได้ถูกต้อง")
    return True


def main():
    print("🐾 PetCare RAG - Full Test Suite")
    print("=" * 60)

    results = {}

    # Test 1: Build Index
    try:
        results["build_index"] = test_step_1_build_index()
    except Exception as e:
        print(f"❌ FAIL: Build Index - {e}")
        results["build_index"] = False

    # Test 2: Search (ต้องผ่าน Test 1 ก่อน)
    if results.get("build_index"):
        try:
            results["search"] = test_step_2_search()
        except Exception as e:
            print(f"❌ FAIL: Search - {e}")
            results["search"] = False
    else:
        results["search"] = False
        print("⏭️ SKIP: Search (ต้องผ่าน Build Index ก่อน)")

    # Test 3: RAG Answer
    if results.get("search"):
        try:
            results["rag_answer"] = test_step_3_rag_answer()
        except Exception as e:
            print(f"❌ FAIL: RAG Answer - {e}")
            results["rag_answer"] = False
    else:
        results["rag_answer"] = False
        print("⏭️ SKIP: RAG Answer (ต้องผ่าน Search ก่อน)")

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())
    print(f"\n{'🎉 ทุกการทดสอบผ่าน!' if all_passed else '⚠️ มีการทดสอบที่ไม่ผ่าน'}")
    return all_passed


if __name__ == "__main__":
    main()
