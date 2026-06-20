import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Import chính xác bộ Retriever lai và cấu hình đã tối ưu của bạn
from src.retriever import HybridVectorGraphRetriever

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN CHUẨN XÁC NÉN GỌN
# ==========================================
st.set_page_config(
    page_title="Tra cứu Pháp luật",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ==========================================
# KHỞI TẠO STATE CHO CHATBOT (BẮT BUỘC)
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "langchain_history" not in st.session_state:
    st.session_state.langchain_history = []
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #F8FAFC !important; border-right: 1px solid #E2E8F0; }
    [data-testid="stSidebarUserContent"] { padding-top: 1.5rem !important; padding-left: 1rem !important; padding-right: 1rem !important; }
    .flat-header { display: flex; align-items: center; gap: 8px; font-size: 12px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 15px; margin-bottom: 8px; }
    .flat-header i { color: #2563EB; font-size: 13px; }
    div[data-baseweb="select"] span[data-baseweb="tag"] { background-color: #2563EB !important; border-radius: 4px !important; padding-top: 2px !important; padding-bottom: 2px !important; }
    div[data-baseweb="select"] span[data-baseweb="tag"] span { color: #FFFFFF !important; font-size: 12px !important; }
    div[data-baseweb="select"] > div { background-color: #FFFFFF !important; border: 1px solid #E2E8F0 !important; border-radius: 6px !important; min-height: 34px !important; height: 34px !important; font-size: 13px !important; }
    .filter-caption { font-size: 12px; color: #64748B; font-weight: 500; margin-top: 8px; margin-bottom: 4px; }
    .status-card-clean { background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; margin-top: 15px; box-shadow: 0 1px 2px rgba(0,0,0,0.01); }
    .status-badge-green { background-color: #DCFCE7; color: #16A34A; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 700; border: 1px solid #BBF7D0; }
    .block-container { padding-top: 1.5rem !important; max-width: 900px !important; }
    .chat-header-line { font-size: 16px; font-weight: 600; color: #0F172A; border-bottom: 1px solid #E2E8F0; padding-bottom: 12px; margin-bottom: 24px; }
    [data-testid="stChatMessageAssistant"] { background-color: #FFFFFF !important; border-radius: 12px !important; padding: 20px !important; border: 1px solid #E2E8F0 !important; box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important; }
    [data-testid="stChatMessageAssistant"] .stMarkdown { text-align: justify !important; text-justify: inter-word !important; font-size: 14.5px; line-height: 1.6; color: #1E293B; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. KHỞI TẠO CORE AGENT VỚI CACHE
# ==========================================
@st.cache_resource
def init_legal_agent():
    # Khởi tạo bộ sinh và bộ truy vấn lai đã sửa lỗi Neo4j của bạn
    retreiver = HybridVectorGraphRetriever()
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Prompt hỗ trợ bộ nhớ hội thoại liên tục
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """Bạn là một trợ lý ảo chuyên gia về Luật Giao thông Đường bộ Việt Nam. 
Nhiệm vụ của bạn là trả lời các câu hỏi của người dùng một cách chính xác, minh bạch dựa ĐÚNG và ĐỦ theo ngữ cảnh pháp lý được cung cấp dưới đây.

LƯU Ý QUAN TRỌNG:
- Chỉ trả lời dựa trên thông tin có trong 'Ngữ cảnh pháp lý'. Không tự bịa đặt điều luật.
- Hãy trích dẫn rõ ràng tên văn bản, Chương mấy, Điều mấy khi trả lời để tạo sự uy tín.
- Nếu ngữ cảnh không chứa thông tin để trả lời, hãy lịch sự báo rằng hệ thống dữ liệu hiện tại chưa cập nhật hành vi này.

Ngữ cảnh pháp lý được cung cấp từ hệ thống Vector-Graph DB:
----------------------
{context}
----------------------"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])
    return retreiver, llm, prompt_template

retriever, llm, prompt_template = init_legal_agent()

# Thay vì khai báo mảng cứng, hãy quét động thư mục tài liệu của bạn
DOCS_DIR = "docs"

if os.path.exists(DOCS_DIR):
    # Lấy tất cả file .docx, bỏ đi phần mở rộng và dấu gạch dưới ở đầu (nếu có)
    all_sources = [
        f.replace(".docx", "").strip() 
        for f in os.listdir(DOCS_DIR) 
        if f.endswith(".docx")
    ]
else:
    # Phương án dự phòng nếu không tìm thấy thư mục docs
    all_sources = ["Luật Trật tự và an toàn GTĐB", "Nghị định xử phạt", "Luật Đường bộ", "Luật Giao thông đường bộ"]

# ==========================================
# 3. SIDEBAR PHẲNG - HIỂN THỊ ĐỘNG DANH SÁCH CHECKBOX
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

    # 3.1 Phạm vi tra cứu - Render danh sách checkbox động hàng dọc cực kỳ dễ nhìn
    st.markdown('<div class="flat-header"><i class="fa-regular fa-folder-open"></i> Phạm vi tra cứu</div>', unsafe_allow_html=True)
    
    selected_docs = []
    # Duyệt qua danh sách file thực tế quét được để tạo checkbox
    for source in all_sources:
        if st.checkbox(source, value=True, key=f"cb_{source}"):
            selected_docs.append(source)

    # Nếu người dùng lỡ tay bỏ chọn tất cả, mặc định lấy toàn bộ phạm vi để tránh lỗi RAG
    if not selected_docs:
        selected_docs = all_sources

    # 3.2 Bộ lọc nâng cao (Giữ nguyên phần dưới của bạn)
    st.markdown('<div class="flat-header"><i class="fa-solid fa-sliders"></i> Bộ lọc nâng cao</div>', unsafe_allow_html=True)
   
    st.markdown('<div class="filter-caption">Loại phương tiện</div>', unsafe_allow_html=True)
    st.selectbox("sb_vehicle", ["Tất cả", "Ô tô", "Xe máy", "Xe máy chuyên dùng"], label_visibility="collapsed", key="selected_vehicle")

    st.markdown('<div class="filter-caption">Hành vi / Lỗi</div>', unsafe_allow_html=True)
    st.selectbox("sb_violation", ["Tất cả", "Tốc độ", "Nồng độ cồn", "Tín hiệu đèn"], label_visibility="collapsed", key="selected_violation")

    st.markdown('<div class="flat-header"><i class="fa-solid fa-chart-line"></i> Trạng thái LLM</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="status-card-clean">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 13px; font-weight: 600; color: #16A34A;">● LLM đang hoạt động</div>
                    <div style="font-size: 11px; color: #64748B; margin-top: 2px;">Mô hình: Gemini Flash</div>
                </div>
                <div class="status-badge-green">100%</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.langchain_history = []
        st.rerun()

    st.markdown('<div class="flat-header"><i class="fa-solid fa-clock-rotate-left"></i> Lịch sử chat</div>', unsafe_allow_html=True)
    
    # Lấy danh sách các câu hỏi của User (role == "user") trong session_state
    user_questions = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]
    
    if user_questions:
        # Hiển thị tối đa 5 câu hỏi gần nhất trên sidebar cho gọn
        for q in user_questions[-5:]:
            # Cắt ngắn câu hỏi nếu quá dài để tránh vỡ giao diện sidebar
            short_q = q[:30] + "..." if len(q) > 30 else q
            st.markdown(f"""
                <div style="background-color: #E2E8F0; color: #1E293B; padding: 6px 10px; border-radius: 6px; font-size: 12.5px; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
                    <i class="fa-regular fa-comment-dots" style="color: #64748B;"></i>
                    <span style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%;">{short_q}</span>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size: 12px; color: #94A3B8; font-style: italic; padding-left: 4px;">Chưa có cuộc hội thoại nào</div>', unsafe_allow_html=True)


# ==========================================
# 4. QUẢN LÝ TRẠNG THÁI BỘ NHỚ TRONG STREAMLIT
# ==========================================
st.markdown('<div class="chat-header-line">Chat tra cứu pháp luật</div>', unsafe_allow_html=True)

    # if "messages" not in st.session_state:
    #     st.session_state.messages = []

# Lịch sử dạng đối tượng LangChain để truyền trực tiếp vào Prompt template
if "langchain_history" not in st.session_state:
    st.session_state.langchain_history = []

if not st.session_state.messages:
    st.markdown("""
        <div style='text-align: center; margin-top: 3rem; margin-bottom: 2rem;'>
            <div style='color: #2563EB; font-size: 36px; margin-bottom: 12px;'><i class="fa-solid fa-scale-balanced"></i></div>
            <h2 style='color: #0F172A; font-weight: 600; font-size: 22px; margin-bottom: 8px;'>Hệ thống Trợ lý Trí tuệ Nhân tạo Pháp luật Giao thông</h2>
            <p style='color: #64748B; font-size: 14px; max-width: 580px; margin: 0 auto 16px auto;'>
                Hỗ trợ tra cứu nhanh chóng, trích dẫn chính xác các quy định, nghị định xử phạt hành chính đường bộ.
            </p>
            <div style='background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px 16px; max-width: 600px; margin: 0 auto; text-align: left;'>
                <p style='color: #64748B; font-size: 12px; margin: 0; line-height: 1.5;'>
                    💡 <b>Thông báo miễn trừ trách nhiệm:</b> Thông tin do trợ lý AI cung cấp chỉ nhằm mục đích trích dẫn tham khảo văn bản quy phạm pháp luật nhanh và không thể thay thế cho văn bản chính thức của cơ quan nhà nước.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Hiển thị hội thoại lịch sử dữ liệu cũ
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 5. LOGIC XỬ LÝ TRUY VẤN LAI + TRÍ NHỚ ĐỘNG
# ==========================================
if prompt := st.chat_input("Nhập nội dung bạn cần tra cứu luật tại đây..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(""):
            try:
                # --- BƯỚC CẢI TIẾN: QUERY REWRITING ---
                # Dùng LLM để chuyển câu hỏi khẩu ngữ thành từ khóa pháp luật chuẩn xác
                rewrite_prompt = f"""Bạn là một chuyên gia pháp lý. Hãy chuyển câu hỏi khẩu ngữ sau đây của người dùng thành các thuật ngữ hoặc hành vi chính xác dùng trong văn bản luật giao thông để phục vụ tìm kiếm database (chỉ trả về câu văn/từ khóa sau khi tối ưu, không giải thích gì thêm).
Câu hỏi: {prompt}"""
                
                # Gọi nhanh LLM để đổi "vượt đèn đỏ" -> "không chấp hành hiệu lệnh của đèn tín hiệu giao thông"
                optimized_query = llm.invoke(rewrite_prompt).content.strip()
                
                # Tăng top_k lên 5 hoặc 6 để bao quát dữ liệu tốt hơn
                retrieved_docs = retriever.retrieve(optimized_query, top_k=5)
                
                # --- ĐỊNH DẠNG NGỮ CẢNH HOÀN TOÀN THÔNG SUỐT ĐỂ KIỂM TRA ---
                formatted_context = ""
                
                print(f"\n===== DEBUG: HỆ THỐNG TRẢ VỀ {len(retrieved_docs)} MẢNH VĂN BẢN =====")

                for idx, doc in enumerate(retrieved_docs):
                    content, source, dieu = "", "", ""
                    
                    # 1. Bóc tách dữ liệu an toàn từ Dict (Neo4j) hoặc Object (Chroma)
                    if isinstance(doc, dict):
                        if 'node' in doc and isinstance(doc['node'], dict):
                            node_data = doc['node']
                        elif 'properties' in doc and isinstance(doc['properties'], dict):
                            node_data = doc['properties']
                        else:
                            node_data = doc
                            
                        content = node_data.get('content', node_data.get('page_content', ''))
                        source = node_data.get('source', '')
                        dieu = node_data.get('dieu', 'Không rõ')
                    else:
                        content = getattr(doc, 'page_content', '')
                        metadata = getattr(doc, 'metadata', {})
                        source = metadata.get('source', '')
                        dieu = metadata.get('dieu', 'Không rõ')
                    
                    if not content:
                        continue

                    # 2. In log ra terminal để bạn kiểm tra xem mảnh text chứa nội dung gì
                    print(f"--> Mảnh số {idx+1} | Nguồn: {source} | Điều: {dieu}")
                    print(f"    Nội dung xem trước: {content[:100]}...")

                    # 3. TẠM THỜI BỎ QUA TẤT CẢ BỘ LỌC (CẢ CHECKBOX LẪN SELECTBOX)
                    # Mục đích: Ép toàn bộ 5 mảnh tìm được từ Terminal phải đổ thẳng vào Web
                    
                    # 4. Đóng gói trực tiếp gửi cho Gemini
                    formatted_context += f"\n[VĂN BẢN PHÁP LUẬT SỐ {idx+1}]\n"
                    formatted_context += f"- Nguồn gốc: {source if source else 'Nghị định xử phạt'}\n"
                    formatted_context += f"- Điều khoản quy định: {dieu}\n"
                    formatted_context += f"- Nội dung chi tiết: {content}\n"
                    formatted_context += "----------------------------------------\n"

                print(f"========================================================\n")

            

                # [ĐOẠN KIỂM TRA NHANH ĐỂ CHẨN ĐOÁN]
                # Nếu sau khi lọc mà context trống rỗng, in log ra terminal để bạn biết ngay lý do
                if not formatted_context.strip():
                    print("CẢNH BÁO: Tìm thấy tài liệu trong DB nhưng đã bị bộ lọc Sidebar loại bỏ sạch!")
                    print(f"Tên nguồn từ DB: {[d.get('source') if isinstance(d, dict) else d.metadata.get('source') for d in retrieved_docs]}")
                    print(f"Danh sách bạn đang chọn trên Sidebar: {selected_docs}")
                # --- SINH PHẢN HỒI ---
                chain = prompt_template | llm
                response = chain.invoke({
                    "context": formatted_context,
                    "history": st.session_state.langchain_history,
                    "question": prompt # Vẫn giữ câu hỏi gốc của user để Bot trả lời tự nhiên
                })
                
                st.markdown(response.content)
                
                # Lưu lịch sử
                st.session_state.messages.append({"role": "assistant", "content": response.content})
                st.session_state.langchain_history.append(HumanMessage(content=prompt))
                st.session_state.langchain_history.append(AIMessage(content=response.content))
                
                if len(st.session_state.langchain_history) > 6:
                    st.session_state.langchain_history = st.session_state.langchain_history[-6:]
                    
            except Exception as e:
                st.error(f" Có lỗi kết nối hệ thống dữ liệu: {e}")