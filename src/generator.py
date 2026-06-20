import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from src.retriever import HybridVectorGraphRetriever

load_dotenv()

class LegalRAGAgent:
    def __init__(self):
        # 1. Khởi tạo bộ truy vấn lai Vector + Graph
        self.retriever = HybridVectorGraphRetriever()
        
        # 2. Khởi tạo "bộ não" Gemini Flash
        # Dùng model "gemini-2.5-flash" hoặc tên phiên bản bạn đang dùng trong AI Studio
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.2, # Để thấp để ép bot trả lời chính xác theo luật, không bịa chế
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # 3. Thiết kế Prompt mẫu chuẩn cho hệ thống chuyên gia pháp lý
        self.prompt_template = ChatPromptTemplate.from_messages([
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
            ("human", "{question}")
        ])

    def answer(self, question: str) -> str:
        # Bước 1: Gọi Retriever tìm kiếm ngữ cảnh đầy đủ từ Chroma + Neo4j
        retrieved_docs = self.retriever.retrieve(question, top_k=2)
        
        # Bước 2: Định dạng cấu trúc ngữ cảnh để nhét vào Prompt
        formatted_context = ""
        for idx, doc in enumerate(retrieved_docs):
            formatted_context += f"\n--- Nguồn: {doc['source']} | Vị trí: {doc['chuong']} - {doc['dieu']} ---\n"
            formatted_context += f"{doc['content']}\n"
            
        # Bước 3: Trộn Prompt và gửi lên cho Gemini
        chain = self.prompt_template | self.llm
        response = chain.invoke({
            "context": formatted_context,
            "question": question
        })
        
        return response.content