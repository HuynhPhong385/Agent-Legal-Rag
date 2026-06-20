import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from src.retriever import HybridVectorGraphRetriever

load_dotenv()

class LegalRAGAgent:
    def __init__(self):
        # 1. Khởi tạo bộ truy vấn lai Vector + Graph
        self.retriever = HybridVectorGraphRetriever()
        
        # 2. Bộ não Gemini Flash
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.2,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # 3. Khởi tạo danh sách bộ nhớ động (Chat History) dạng list nội bộ
        self.chat_history = []
        
        # 4. Prompt hệ thống mở rộng hỗ trợ MessagesPlaceholder để nhét lịch sử vào
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
            # MessagesPlaceholder sẽ tự động chèn toàn bộ lịch sử các cặp câu hỏi/trả lời trước đó vào đây
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])

    def answer(self, question: str) -> str:
        # Bước 1: Gọi Retriever tìm kiếm ngữ cảnh đầy đủ từ Chroma + Neo4j
        retrieved_docs = self.retriever.retrieve(question, top_k=4)
        
        # Bước 2: Định dạng cấu trúc ngữ cảnh để nhét vào Prompt
        formatted_context = ""
        for idx, doc in enumerate(retrieved_docs):
            formatted_context += f"\n--- Nguồn: {doc['source']} | Vị trí: {doc['chuong']} - {doc['dieu']} ---\n"
            formatted_context += f"{doc['content']}\n"
            
        # Bước 3: Trộn Prompt, Lịch sử và Câu hỏi mới gửi lên cho Gemini
        chain = self.prompt_template | self.llm
        response = chain.invoke({
            "context": formatted_context,
            "history": self.chat_history,
            "question": question
        })
        
        # Bước 4: Lưu cặp hội thoại mới vào Bộ nhớ (Giới hạn giữ lại 6 tin nhắn gần nhất để tránh tràn context)
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=response.content))
        
        if len(self.chat_history) > 6:
            self.chat_history = self.chat_history[-6:]
            
        return response.content

    def clear_memory(self):
        """Hàm dùng để xóa trí nhớ khi người dùng muốn đổi chủ đề hoàn toàn"""
        self.chat_history = []
        print("[Memory] Đã xóa sạch trí nhớ hội thoại cũ!")