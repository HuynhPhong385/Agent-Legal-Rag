# --- FILE app_rag.py ---
import os
from config import embeddings, PERSIST_DIRECTORY
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Kết nối vào Vector DB local
print("📚 Đang kết nối vào kho dữ liệu Legal Vector DB...")
vector_db = Chroma(
    persist_directory=PERSIST_DIRECTORY, 
    embedding_function=embeddings
)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 2. Khởi tạo LLM Gemini 2.5 Flash từ Google AI Studio
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

# 3. Thiết kế Prompt cho Chuyên gia Pháp lý
template = """Bạn là một trợ lý luật sư ảo chuyên nghiệp tại Việt Nam. 
Hãy sử dụng các điều luật được cung cấp dưới đây để trả lời câu hỏi của người dùng một cách chính xác, trích dẫn rõ ràng thuộc Luật hoặc Nghị định nào, Điều bao nhiêu.
Nếu thông tin dưới đây không đề cập đến câu trả lời, hãy thành thật trả lời là bạn không rõ dựa trên nguồn tài liệu hiện có, tuyệt đối không tự bịa đặt ra luật hay mức phạt.

Các điều luật liên quan trích xuất từ bộ dữ liệu:
{context}

Câu hỏi của người dùng: {question}

Câu trả lời của bạn:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get('source', 'Không rõ nguồn')
        dieu = doc.metadata.get('dieu', 'Không rõ điều')
        formatted.append(f"[{source} - {dieu}]\n{doc.page_content}")
    return "\n\n====================\n\n".join(formatted)

# 4. Tạo RAG Chain bằng LCEL
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. VÒNG LẶP CHẠY ỨNG DỤNG LIÊN TỤC
if __name__ == "__main__":
    print("\n🤖 [Hệ thống Trợ lý Luật Giao thông - CORE RAG] đã sẵn sàng!")
    print("👉 Gõ 'quit' hoặc 'exit' và nhấn Enter để thoát ứng dụng.")
    print("-" * 60)
    
    while True:
        query = input("\n❓ Nhập câu hỏi của bạn: ")
        
        # Kiểm tra điều kiện thoát
        if query.strip().lower() in ['quit', 'exit']:
            print("👋 Tạm biệt bạn! Hẹn gặp lại trong các phiên làm việc tới.")
            break
            
        # Bỏ qua nếu người dùng nhấn Enter mà không gõ gì
        if not query.strip():
            continue
            
        print("🧠 Đang truy xuất dữ liệu BKAI và suy luận với Gemini 2.5 Flash...")
        
        print("🧠 Đang truy xuất dữ liệu từ BKAI Embedding...")
        try:
            # 1. Lấy trực tiếp các tài liệu thô từ Vector DB
            docs = retriever.invoke(query)
            
            print("\n================ [DEBUG TRUY XUẤT] ================")
            print(f"Tổng số tài liệu tìm thấy: {len(docs)}")
            
            for idx, doc in enumerate(docs):
                print(f"\n📍 KẾT QUẢ THỨ {idx+1}:")
                print(f"   - File nguồn: {doc.metadata.get('source')}")
                print(f"   - Vị trí: {doc.metadata.get('chuong')} -> {doc.metadata.get('dieu')}")
                print(f"   - Nội dung thô trong DB (500 ký tự đầu):")
                print(f"     \"\"\"{doc.page_content[:500]}...\"\"\"")
                print("-" * 50)
            print("==================================================\n")
            
            # 2. Sau đó mới cho Gemini chạy
            print("🧠 Đang gửi sang Gemini 2.5 Flash suy luận...")
            response = rag_chain.invoke(query)
            print("\n✨ TRẢ LỜI TỪ GEMINI:")
            print(response)
            print("-" * 60)
            
        except Exception as e:
            print(f"❌ Có lỗi xảy ra: {e}")