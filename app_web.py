import streamlit as st
import os
import pickle
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Import các thành phần cấu hình từ gói src của bạn
from src.config import embeddings, PERSIST_DIRECTORY, BM25_INDEX_PATH
from src.retrievers import CustomHybridRetriever
from src.prompts import LEGAL_PROMPT

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN CHUẨN XÁC NÉN GỌN
# ==========================================
st.set_page_config(
    page_title="Tra cứu Pháp luật",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Khối CSS xử lý triệt để lỗi lệch ô, đổi màu tag sang xanh lá/xanh dương và căn đều lề chữ
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Thiết lập nền xám nhạt cho Sidebar giống mẫu */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0;
    }
    
    [data-testid="stSidebarUserContent"] {
        padding-top: 1.5rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    /* Tiêu đề phẳng không dùng ô bọc thô kệch */
    .flat-header {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 15px;
        margin-bottom: 8px;
    }
    .flat-header i {
        color: #2563EB;
        font-size: 13px;
    }

    /* Đổi màu và làm gọn các Tag lựa chọn văn bản sang Màu Xanh Chủ Đạo của hệ thống */
    div[data-baseweb="select"] span[data-baseweb="tag"] {
        background-color: #2563EB !important; /* Xanh dương SaaS cao cấp, có thể đổi thành #16A34A nếu thích xanh lá hoàn toàn */
        border-radius: 4px !important;
        padding-top: 2px !important;
        padding-bottom: 2px !important;
    }
    div[data-baseweb="select"] span[data-baseweb="tag"] span {
        color: #FFFFFF !important;
        font-size: 12px !important;
    }

    /* Thu nhỏ khoảng cách các ô selectbox của bộ lọc */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 6px !important;
        min-height: 34px !important;
        height: 34px !important;
        font-size: 13px !important;
    }
    
    .filter-caption {
        font-size: 12px;
        color: #64748B;
        font-weight: 500;
        margin-top: 8px;
        margin-bottom: 4px;
    }

    /* Khối trạng thái LLM giữ nguyên độ đẹp và có viền bo tròn gọn gàng */
    .status-card-clean {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        margin-top: 15px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.01);
    }
    .status-badge-green {
        background-color: #DCFCE7;
        color: #16A34A;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 700;
        border: 1px solid #BBF7D0;
    }

    /* KHU VỰC CHAT TRUNG TÂM & CĂN THẲNG HÀNG 2 BÊN RÌA */
    .block-container {
        padding-top: 1.5rem !important;
        max-width: 900px !important;
    }
    
    .chat-header-line {
        font-size: 16px;
        font-weight: 600;
        color: #0F172A;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 12px;
        margin-bottom: 24px;
    }

    /* Căn đều hai bên lề cho phần chữ hiển thị trong ô Chat Assistant */
    [data-testid="stChatMessageAssistant"] {
        background-color: #FFFFFF !important;
        border-radius: 12px !important;
        padding: 20px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
    }
    
    /* Ép văn bản text và danh sách bullet trong ô phản hồi phải thẳng hàng lề phải và lề trái */
    [data-testid="stChatMessageAssistant"] .stMarkdown {
        text-align: justify !important;
        text-justify: inter-word !important;
        font-size: 14.5px;
        line-height: 1.6;
        color: #1E293B;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. KHỞI TẠO ENGINE DỮ LIỆU RAG
# ==========================================
@st.cache_resource
def load_rag_system():
    vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_docs = pickle.load(f)
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    sources = list(set([doc.metadata.get('source', 'Chưa rõ') for doc in bm25_docs]))
    return vector_db, bm25_retriever, sources

vector_db, bm25_retriever, all_sources = load_rag_system()

# ==========================================
# 3. SIDEBAR PHẲNG - KHÔNG Ô THỪA - KHÔNG LỆCH VỊ TRÍ
# ==========================================
with st.sidebar:
    # Logo thương hiệu ứng dụng
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px; padding-left: 2px;">
            <div style="background-color: #2563EB; color: white; padding: 6px 8px; border-radius: 6px;">
                <i class="fa-solid fa-scale-balanced" style="font-size: 16px;"></i>
            </div>
            <div>
                <div style="font-size: 16px; font-weight: 700; color: #0F172A; line-height: 1.1;">Tra cứu Pháp luật</div>
                <div style="font-size: 11px; color: #64748B;">An toàn giao thông đường bộ</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 3.1 Phạm vi tra cứu (Tiêu đề phẳng phẳng lỳ)
    st.markdown('<div class="flat-header"><i class="fa-regular fa-folder-open"></i> Phạm vi tra cứu</div>', unsafe_allow_html=True)
    selected_docs = st.multiselect(
        "Chọn văn bản pháp luật",
        options=all_sources,
        default=all_sources,
        label_visibility="collapsed"
    )

    # 3.2 Bộ lọc nâng cao (Xếp gọn gàng ngay bên dưới)
    st.markdown('<div class="flat-header"><i class="fa-solid fa-sliders"></i> Bộ lọc nâng cao</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="filter-caption">Loại phương tiện</div>', unsafe_allow_html=True)
    st.selectbox("sb_vehicle", ["Tất cả", "Ô tô", "Xe máy", "Xe thô sơ"], label_visibility="collapsed")
    
    st.markdown('<div class="filter-caption">Hành vi / Lỗi</div>', unsafe_allow_html=True)
    st.selectbox("sb_violation", ["Tất cả", "Tốc độ", "Nồng độ cồn", "Tín hiệu đèn"], label_visibility="collapsed")
    
    st.markdown('<div class="filter-caption">Mức xử phạt</div>', unsafe_allow_html=True)
    st.selectbox("sb_fine", ["Tất cả", "Dưới 1 triệu", "1 - 5 triệu", "Trên 5 triệu"], label_visibility="collapsed")

    # 3.3 Trạng thái LLM Chuẩn màu xanh lá cây
    st.markdown('<div class="flat-header"><i class="fa-solid fa-chart-line"></i> Trạng thái LLM</div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="status-card-clean">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 13px; font-weight: 600; color: #16A34A;">
                        <span style="margin-right: 4px; color: #16A34A;">●</span> LLM đang hoạt động
                    </div>
                    <div style="font-size: 11px; color: #64748B; margin-top: 2px;">Mô hình: GPT-4o-mini</div>
                </div>
                <div class="status-badge-green">100%</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 3.4 Lịch sử Chat tinh gọn
    st.markdown('<div class="flat-header"><i class="fa-solid fa-clock-rotate-left"></i> Lịch sử chat</div>', unsafe_allow_html=True)
    st.markdown("""
        <div style="background-color: #2563EB; color: white; padding: 8px 10px; border-radius: 6px; font-size: 12.5px; display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span>Vượt đèn đỏ ô tô phạt bao nhiêu?</span>
            <span style="font-size: 10px; opacity: 0.8;">10:24</span>
        </div>
        <div style="color: #475569; padding: 6px 10px; font-size: 12.5px; display: flex; justify-content: space-between;">
            <span><i class="fa-regular fa-circle-check" style="color: #94A3B8; margin-right: 6px;"></i> Chạy quá tốc độ 20km/h...</span>
            <span style="font-size: 10px; color: #94A3B8;">Hôm qua</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# 4. KHÔNG GIAN CHAT CHÍNH VÀ TUYÊN BỐ PHÁP LÝ
# ==========================================
st.markdown('<div class="chat-header-line">Chat tra cứu pháp luật</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Giao diện chào sân kèm Disclaimer an toàn pháp lý đặt ở giữa
if not st.session_state.messages:
    st.markdown("""
        <div style='text-align: center; margin-top: 5rem; margin-bottom: 2rem;'>
            <div style='color: #2563EB; font-size: 36px; margin-bottom: 12px;'><i class="fa-solid fa-scale-balanced"></i></div>
            <h2 style='color: #0F172A; font-weight: 600; font-size: 22px; margin-bottom: 8px;'>Hệ thống Trợ lý Trí tuệ Nhân tạo Pháp luật Giao thông</h2>
            <p style='color: #64748B; font-size: 14px; max-width: 580px; margin: 0 auto 16px auto;'>
                Hỗ trợ tra cứu nhanh chóng, trích dẫn chính xác các quy định, nghị định xử phạt hành chính đường bộ.
            </p>
            <div style='background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px 16px; max-width: 600px; margin: 0 auto; text-align: left;'>
                <p style='color: #64748B; font-size: 12px; margin: 0; line-height: 1.5;'>
                    💡 <b>Thông báo miễn trừ trách nhiệm:</b> Thông tin do trợ lý AI cung cấp chỉ nhằm mục đích trích dẫn tham khảo văn bản quy phạm pháp luật nhanh và không thể thay thế cho văn bản chính thức của cơ quan nhà nước hoặc ý kiến tư vấn chuyên môn từ luật sư.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Hiển thị các tin nhắn cũ
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Logic kết nối cơ sở dữ liệu RAG
def get_response(user_input):
    search_kwargs = {"k": 6}
    if len(selected_docs) < len(all_sources):
        search_kwargs["filter"] = {"source": {"$in": selected_docs}}

    vector_retriever = vector_db.as_retriever(search_kwargs=search_kwargs)
    bm25_retriever.k = 6
    hybrid_retriever = CustomHybridRetriever(vector_retriever=vector_retriever, bm25_retriever=bm25_retriever)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

    def format_docs_for_llm(docs):
        return "\n\n".join([f"NGUỒN: {doc.metadata.get('source')}\nĐIỀU LUẬT: {doc.metadata.get('dieu')}\nNỘI DUNG: {doc.page_content}" for doc in docs])

    chain = (
        {"context": hybrid_retriever | format_docs_for_llm, "question": RunnablePassthrough()}
        | LEGAL_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain.invoke(user_input)

# Ô gõ chat chân trang
if prompt := st.chat_input("Nhập nội dung bạn cần tra cứu luật tại đây..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(""):
            response = get_response(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})