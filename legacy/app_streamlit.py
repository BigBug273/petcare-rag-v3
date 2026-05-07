"""
app.py
======
PetCare RAG Chatbot - Streamlit Web Demo
ส่วนของคนที่ 3: Backend / UI

วิธีรัน:
    streamlit run app.py

หน้าเว็บจะมี:
- Title: PetCare RAG Chatbot
- ช่องพิมพ์คำถาม
- ปุ่ม Ask
- แสดงคำตอบ AI + Source + สายพันธุ์ที่เกี่ยวข้อง
"""

import os
import sys

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
from dotenv import load_dotenv
from html import escape

load_dotenv()

# ============================================
# Path setup - ให้หา rag/ package เจอ
# ============================================
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rag.answer import PetRAG


# ============================================
# Page Config
# ============================================
st.set_page_config(
    page_title="PetCare RAG Chatbot",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

if not os.getenv("GEMINI_API_KEY"):
    st.error(
        "Missing GEMINI_API_KEY. Create a local .env file or set GEMINI_API_KEY "
        "in your Render environment variables, then restart the app."
    )
    st.stop()


# ============================================
# UI Theme - ธีมแมวโทนพาสเทลสำหรับหน้าเว็บ
# ============================================
st.markdown("""
<style>
    :root {
        --cream: #fff8ef;
        --surface: #ffffff;
        --soft-orange: #f6a96d;
        --soft-pink: #ffd9df;
        --light-brown: #8a5a44;
        --muted-brown: #a9785f;
        --warm-border: #f1d7c5;
        --cat-shadow: 0 10px 28px rgba(138, 90, 68, 0.12);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 217, 223, 0.55), transparent 30%),
            linear-gradient(135deg, #fff8ef 0%, #fffdf9 48%, #fff1e2 100%);
        color: #4c342b;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fff4e7 0%, #ffffff 100%);
        border-right: 1px solid var(--warm-border);
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--light-brown);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        max-width: 1160px;
    }

    .cat-hero {
        position: relative;
        background: linear-gradient(135deg, #ffffff 0%, #fff1e2 55%, #ffd9df 100%);
        border: 1px solid var(--warm-border);
        border-radius: 8px;
        box-shadow: var(--cat-shadow);
        padding: 34px 34px 30px;
        margin-bottom: 24px;
        overflow: hidden;
    }

    .cat-hero:after {
        content: "🐾";
        position: absolute;
        right: 28px;
        top: 20px;
        font-size: 74px;
        opacity: 0.16;
        transform: rotate(-12deg);
    }

    .cat-hero h1 {
        color: #6e4434;
        font-size: 2.5rem;
        line-height: 1.12;
        margin: 0 0 12px;
        letter-spacing: 0;
    }

    .cat-hero p {
        color: #765344;
        font-size: 1.05rem;
        line-height: 1.75;
        max-width: 760px;
        margin: 0;
    }

    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 20px;
    }

    .cat-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(241, 215, 197, 0.9);
        border-radius: 999px;
        color: #744631;
        font-size: 0.86rem;
        font-weight: 700;
        padding: 7px 12px;
    }

    .answer-card,
    .source-card,
    .input-card {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid var(--warm-border);
        border-radius: 8px;
        box-shadow: var(--cat-shadow);
    }

    .input-card {
        padding: 20px 20px 10px;
        margin-bottom: 18px;
    }

    .input-card h3,
    .answer-title {
        color: var(--light-brown);
        margin: 0 0 8px;
    }

    .input-card p {
        color: var(--muted-brown);
        margin: 0 0 14px;
    }

    div[data-testid="stTextInput"] input {
        border: 1px solid #edcdb8;
        border-radius: 8px;
        background: #fffdf9;
        color: #4c342b;
        min-height: 48px;
        font-size: 1rem;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: var(--soft-orange);
        box-shadow: 0 0 0 3px rgba(246, 169, 109, 0.18);
    }

    .stButton > button {
        border-radius: 8px;
        border: 1px solid #efc7ad;
        background: #ffffff;
        color: #704936;
        font-weight: 700;
        transition: all 0.15s ease;
    }

    .stButton > button:hover {
        border-color: var(--soft-orange);
        color: #5b3326;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(138, 90, 68, 0.12);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #f6a96d 0%, #f48b9a 100%);
        border: 0;
        color: white;
    }

    .answer-card {
        padding: 22px 24px;
        border-left: 6px solid var(--soft-orange);
        line-height: 1.85;
        font-size: 1rem;
        color: #4c342b;
        margin-bottom: 20px;
    }

    .question-card {
        background: #fff4e7;
        border: 1px dashed #e9b98f;
        border-radius: 8px;
        color: #6e4434;
        padding: 14px 16px;
        margin: 8px 0 18px;
    }

    .source-card {
        padding: 16px;
        min-height: 168px;
        margin-bottom: 12px;
    }

    .source-card strong {
        color: #6e4434;
        font-size: 1.02rem;
    }

    .source-meta {
        color: #856352;
        font-size: 0.86rem;
        margin: 8px 0;
    }

    .source-link {
        color: #b85d4c;
        font-size: 0.8rem;
        word-break: break-all;
        text-decoration: none;
    }

    .breed-tag {
        display: inline-block;
        background: #fff0f2;
        border: 1px solid #ffd2d9;
        border-radius: 999px;
        color: #7a4638;
        padding: 6px 12px;
        margin: 4px 5px 4px 0;
        font-size: 0.9rem;
        font-weight: 700;
    }

    .sidebar-note {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid var(--warm-border);
        border-radius: 8px;
        padding: 12px 14px;
        color: #765344;
        line-height: 1.65;
        margin-bottom: 12px;
    }

    .footer-note {
        text-align: center;
        color: #9b7664;
        font-size: 0.82rem;
        padding: 20px 0 4px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# Load RAG System (cached - โหลดครั้งเดียว)
# ============================================
@st.cache_resource
def load_rag():
    """โหลด RAG system ครั้งเดียว แล้ว cache ไว้"""
    provider = "gemini"
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        st.error("❌ ไม่พบ GEMINI_API_KEY กรุณาตั้งค่า Environment Variable ก่อนใช้งาน")
        st.stop()

    rag = PetRAG(llm_provider=provider, api_key=api_key)
    return rag


# ============================================
# Sidebar - ข้อมูลโปรเจกต์ วิธีใช้ ตัวอย่างคำถาม และ Tech stack
# ============================================
with st.sidebar:
    st.markdown("## 🐱 PetCare RAG")
    st.markdown(
        """
        <div class="sidebar-note">
            ผู้ช่วย AI สำหรับถามตอบเรื่องสายพันธุ์แมวและสุนัข
            จากข้อมูลสัตว์เลี้ยงที่เตรียมไว้ในโปรเจกต์
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 📌 About this project")
    st.markdown(
        """
        <div class="sidebar-note">
            โปรเจกต์เดโม Streamlit ที่ใช้ RAG ช่วยค้นข้อมูลที่เกี่ยวข้อง
            แล้วให้ Gemini สรุปคำตอบเป็นภาษาไทยแบบอ่านง่าย
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🐾 How to use")
    st.markdown(
        """
        <div class="sidebar-note">
            1. พิมพ์คำถามเกี่ยวกับแมวหรือสุนัข<br>
            2. เลือกจำนวนข้อมูลอ้างอิงที่อยากใช้<br>
            3. กด Ask แล้วรอคำตอบจากผู้ช่วย
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 💡 Example questions")
    example_queries = [
        "แมวพันธุ์ไหนเหมาะกับคนอยู่คนเดียว",
        "สุนัขพันธุ์ไหนขนสั้นดูแลง่าย",
        "Persian มีนิสัยยังไง",
        "แมวขนยาวต้องดูแลอะไรบ้าง",
        "สุนัขเล็กๆ ที่นิสัยอ่อนโยน",
        "แมวที่ชอบเล่นและขี้เล่น",
    ]
    for q in example_queries:
        if st.button(q, key=q, use_container_width=True):
            st.session_state["user_query"] = q

    st.markdown("### ⚙️ Search settings")
    top_k = st.slider("จำนวนข้อมูลอ้างอิงที่ใช้ค้นหา", 1, 5, 3, key="top_k")

    st.markdown("### 🧰 Tech stack")
    st.markdown(
        """
        <div class="sidebar-note">
            Streamlit · Sentence-BERT · FAISS · Gemini AI · Python
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption("ปลอดภัยด้วย Environment Variable")


# ============================================
# Main Content - Hero และพื้นที่ถามคำถาม
# ============================================

# Hero section - ส่วนต้อนรับธีมแมว
st.markdown("""
<section class="cat-hero">
    <h1>🐱 PetCare RAG Assistant</h1>
    <p>
        ผู้ช่วย AI สำหรับตอบคำถามเรื่องการดูแลสัตว์เลี้ยงและสายพันธุ์แมว-สุนัข
        โดยค้นข้อมูลที่เกี่ยวข้องด้วยระบบ RAG แล้วสรุปคำตอบให้อ่านง่าย เหมาะสำหรับเดโมโปรเจกต์นักศึกษา
    </p>
    <div class="badge-row">
        <span class="cat-badge">🐾 Cat Care</span>
        <span class="cat-badge">📚 RAG System</span>
        <span class="cat-badge">✨ Gemini AI</span>
        <span class="cat-badge">🎓 Student Project</span>
    </div>
</section>
""", unsafe_allow_html=True)

# Question input - กล่องพิมพ์คำถาม
st.markdown("""
<div class="input-card">
    <h3>ถามน้องผู้ช่วยได้เลย</h3>
    <p>พิมพ์คำถามเกี่ยวกับนิสัย การดูแล หรือสายพันธุ์สัตว์เลี้ยงที่อยากรู้</p>
</div>
""", unsafe_allow_html=True)
col_input1, col_input2 = st.columns([5, 1])

with col_input1:
    user_query = st.text_input(
        "คำถามเกี่ยวกับสัตว์เลี้ยง",
        placeholder="เช่น แมวพันธุ์ไหนเหมาะกับคนอยู่คนเดียว?",
        label_visibility="collapsed",
        value=st.session_state.get("user_query", "")
    )

with col_input2:
    ask_clicked = st.button("🐾 Ask", type="primary", use_container_width=True)

# Clear the session state after using it
if "user_query" in st.session_state:
    if st.session_state["user_query"] != user_query:
        pass  # user typed something different
    # Don't clear immediately so the query stays

# ============================================
# Process Query
# ============================================
if ask_clicked and user_query.strip():
    # โหลด RAG
    with st.spinner("🔄 กำลังโหลดระบบ RAG..."):
        rag = load_rag()

    # ถามคำถาม
    with st.spinner("🔍 กำลังค้นหาและตอบคำถาม..."):
        try:
            result = rag.answer(user_query, top_k=top_k)
        except Exception as e:
            # ถ้า LLM มีปัญหา ใช้ search-only mode
            st.warning(f"⚠️ LLM มีปัญหา: {e}")
            st.info("🔄 ใช้ Search-only mode แทน...")
            rag = load_rag()
            result = rag.answer_search_only(user_query, top_k=top_k)

    # Store result in session state for display
    st.session_state["last_result"] = result
    st.session_state["last_query"] = user_query

# ============================================
# Display Results - แสดงคำตอบและแหล่งอ้างอิงเป็น cat cards
# ============================================
if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    query = st.session_state.get("last_query", "")
    safe_query = escape(query)
    answer_html = escape(str(result.get("answer", ""))).replace("\n", "<br>")

    # Question echo - แสดงคำถามล่าสุด
    st.markdown(f"""
    <div class="question-card">
        <strong>คำถามของคุณ:</strong> {safe_query}
    </div>
    """, unsafe_allow_html=True)

    # Answer card - กล่องคำตอบธีมแมว
    st.markdown('<h3 class="answer-title">💬 คำตอบจากน้องผู้ช่วย</h3>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="answer-card">
        {answer_html}
    </div>
    """, unsafe_allow_html=True)

    # Sources - การ์ดข้อมูลอ้างอิง
    st.markdown('<h3 class="answer-title">📚 ข้อมูลอ้างอิงที่ใช้ตอบ</h3>', unsafe_allow_html=True)
    sources = result.get("sources", [])

    if sources:
        cols = st.columns(min(len(sources), 3))
        for i, src in enumerate(sources):
            col = cols[i % len(cols)]
            with col:
                pet_emoji = "🐱" if src["type"] == "cat" else "🐶"
                score_pct = src["score"] * 100

                # Color based on score
                if score_pct >= 70:
                    score_color = "#4CAF50"  # green
                    score_label = "เกี่ยวข้องมาก"
                elif score_pct >= 50:
                    score_color = "#FF9800"  # orange
                    score_label = "เกี่ยวข้อง"
                else:
                    score_color = "#9E9E9E"  # grey
                    score_label = "เกี่ยวข้องบางส่วน"

                breed_name = escape(str(src["breed_name"]))
                pet_type = escape(str(src["type"]).upper())
                source_url = escape(str(src["source_url"]))

                st.markdown(f"""
                <div class="source-card">
                    <strong>{pet_emoji} {breed_name}</strong>
                    <div class="source-meta">
                        ประเภท: {pet_type}<br>
                        <span style="color: {score_color}; font-weight: 800;">
                            {score_label} ({score_pct:.0f}%)
                        </span>
                    </div>
                    <a href="{source_url}" target="_blank" class="source-link">
                        🔗 {source_url}
                    </a>
                </div>
                """, unsafe_allow_html=True)

    # Related breeds section - แท็กสายพันธุ์ที่เกี่ยวข้อง
    st.markdown('<h3 class="answer-title">🏷️ สายพันธุ์ที่เกี่ยวข้อง</h3>', unsafe_allow_html=True)
    breed_tags = " ".join([
        f'<span class="breed-tag">'
        f'{"🐱" if src["type"]=="cat" else "🐶"} {escape(str(src["breed_name"]))}</span>'
        for src in sources
    ])
    st.markdown(f'<div style="margin-top: 10px;">{breed_tags}</div>', unsafe_allow_html=True)

    # Expandable: Raw context used
    with st.expander("🔧 ดู Context ที่ใช้ตอบ (สำหรับ debug)"):
        st.text(result.get("context_used", "ไม่มีข้อมูล"))

# ============================================
# Footer
# ============================================
st.markdown("""
<div class="footer-note">
    PetCare RAG Chatbot | Data from <a href="https://www.purina.co.th" target="_blank">Purina Thailand</a>
    | Powered by Sentence-BERT + FAISS + Gemini
</div>
""", unsafe_allow_html=True)
