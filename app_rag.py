import os
import pickle
from src.config import embeddings, PERSIST_DIRECTORY, BM25_INDEX_PATH
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- IMPORT TỪ CÁC FILE MODULE RIÊNG BIỆT VỪA TÁCH ---
from src.retrievers import CustomHybridRetriever
from src.prompts import LEGAL_PROMPT

print(" Đang kết nối vào kho dữ liệu Legal Vector DB...")
if not os.path.exists(PERSIST_DIRECTORY):
    raise FileNotFoundError(f"Không tìm thấy thư mục DB {PERSIST_DIRECTORY}. Hãy chạy ingest_data.py trước!")

# Khởi tạo Dense Retriever (Vector) - top 8
vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
vector_retriever = vector_db.as_retriever(search_kwargs={"k": 8})

print("⚡ Đang tải bộ tra cứu từ khóa BM25 từ ổ cứng...")
if os.path.exists(BM25_INDEX_PATH):
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_docs = pickle.load(f)
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = 8
else:
    raise FileNotFoundError(f"Không tìm thấy file BM25 Index tại {BM25_INDEX_PATH}. Hãy chạy ingest_data.py trước!")

# Khởi tạo bộ tìm kiếm lai bằng class đã tách ra file riêng
hybrid_retriever = CustomHybridRetriever(
    vector_retriever=vector_retriever,
    bm25_retriever=bm25_retriever
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

def format_docs(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get('source', 'Không rõ nguồn')
        dieu = doc.metadata.get('dieu', 'Không rõ điều')
        formatted.append(f"[{source} - {dieu}]\n{doc.page_content}")
    return "\n\n====================\n\n".join(formatted)

# Tạo RAG Chain kết hợp Prompt từ module riêng
rag_chain = (
    {"context": hybrid_retriever | format_docs, "question": RunnablePassthrough()}
    | LEGAL_PROMPT
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    print("\n [Hệ thống Trợ lý Luật Giao thông - MODULE DESIGN] đã sẵn sàng!")
    print(" Gõ 'quit' hoặc 'exit' và nhấn Enter để thoát ứng dụng.")
    print("-" * 60)
    
    while True:
        query = input("\n Nhập câu hỏi của bạn: ")
        
        if query.strip().lower() in ['quit', 'exit']:
            print("Tạm biệt bạn! Hẹn gặp lại.")
            break
            
        if not query.strip():
            continue
            
        print("Đang truy xuất dữ liệu lai và suy luận với Gemini...")
        
        try:
            docs = hybrid_retriever.invoke(query)
            print(f"\n[DEBUG] Số tài liệu tìm thấy: {len(docs)}")
            
            response = rag_chain.invoke(query)
            print("\n TRẢ LỜI TỪ GEMINI:")
            print(response)
            print("-" * 60)
            
        except Exception as e:
            print(f"❌ Có lỗi xảy ra: {e}")