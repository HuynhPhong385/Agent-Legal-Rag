import os
from src.generator import LegalRAGAgent

def main():
    print("🤖 Đang khởi động Trợ lý Luật Giao thông (Hybrid RAG + Gemini)...")
    agent = LegalRAGAgent()
    print("🎯 Trợ lý đã sẵn sàng! Gõ 'exit' hoặc 'quit' để thoát.")
    print("=" * 60)
    
    while True:
        try:
            question = input("\n👤 Bạn hỏi: ")
            if question.strip().lower() in ['exit', 'quit']:
                print("Tạm biệt bạn!")
                break
                
            if not question.strip():
                continue
                
            # Chạy RAG sinh câu trả lời
            print("🤖 Thần chú RAG đang tìm kiếm và suy luận...")
            answer = agent.answer(question)
            
            print("\n🤖 Trợ lý phản hồi:")
            print(answer)
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\nĐã thoát chương trình.")
            break
        except Exception as e:
            print(f"❌ Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    main()