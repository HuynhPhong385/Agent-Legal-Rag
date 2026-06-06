import streamlit as tf
import os
import pickle
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.config import embeddings, PERSIST_DIRECTORY, BM25_INDEX_PATH
from src.retrievers import CustomHybridRetriever
from src.prompts import LEGAL_PROMPT

# --- CẤU HÌNH TRANG WEB ---
tf.set_page_config(
    page_title="Trợ lý Luật Giao thông Đường bộ",
    page_icon="🤖",
    layout="centered"
)

tf.title("⚖️ Trợ lý Luật sư Ảo - Giao Thông Việt Nam")
tf.caption("Hệ thống Agentic RAG hỗ trợ tra cứu Nghị định 100 & Luật Đường bộ mới nhất")

# --- KHỞI TẠO BACKEND (Caching để không bị load lại mỗi lần chat) ---
@tf.cache_resource
def init_rag_chain():
    # 1. Khởi tạo Dense Retriever
    vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 8})

    # 2. Khởi tạo Sparse Retriever
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_docs = pickle.load(f)
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = 8

    # 3. Khởi tạo Hybrid
    hybrid_retriever = CustomHybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever
    )

    # 4. Khởi tạo LLM và Chain
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    
    def format_docs(docs):
        formatted = []
        for doc in docs:
            source = doc.metadata.get('source', 'Không rõ nguồn')
            dieu = doc.metadata.get('dieu', 'Không rõ điều')
            formatted.append(f"[{source} - {dieu}]\n{doc.page_content}")
        return "\n\n====================\n\n".join(formatted)

    rag_chain = (
        {"context": hybrid_retriever | format_docs, "question": RunnablePassthrough()}
        | LEGAL_PROMPT
        | llm
        | StrOutputParser()
    )
    return rag_chain

# Khởi động hệ thống RAG
try:
    rag_chain = init_rag_chain()
except Exception as e:
    tf.error(f"❌ Không thể khởi tạo hệ thống dữ liệu. Hãy đảm bảo đã chạy ingest_data.py! Lỗi: {e}")
    tf.stop()

# --- QUẢN LÝ LỊCH SỬ CHAT (Session State) ---
if "messages" not in tf.session_state:
    tf.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Tôi là trợ lý luật sư ảo. Bạn cần tra cứu mức phạt hay quy định giao thông nào hôm nay?"}
    ]

# Hiển thị các tin nhắn cũ trong lịch sử
for message in tf.session_state.messages:
    with tf.chat_message(message["role"]):
        tf.markdown(message["content"])

# --- XỬ LÝ NHẬP LIỆU TỪ NGƯỜI DÙNG ---
if user_query := tf.chat_input("Nhập câu hỏi của bạn (Ví dụ: Mức phạt nồng độ cồn xe máy)..."):
    
    # 1. Hiển thị tin nhắn của người dùng trên UI
    with tf.chat_message("user"):
        tf.markdown(user_query)
    tf.session_state.messages.append({"role": "user", "content": user_query})

    # 2. Gọi RAG Chain để lấy câu trả lời từ Gemini
    with tf.chat_message("assistant"):
        message_placeholder = tf.empty()
        with tf.spinner("🧠 Đang tra cứu luật và suy luận luật..."):
            try:
                response = rag_chain.invoke(user_query)
                message_placeholder.markdown(response)
                # Lưu câu trả lời vào lịch sử
                tf.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                tf.error(f"Có lỗi xảy ra: {e}")