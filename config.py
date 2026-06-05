# --- FILE config.py (CẬP NHẬT) ---
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load các biến môi trường từ file .env (Để dành GOOGLE_API_KEY cho LLM sau này)
load_dotenv()

print("🔄 Đang tải mô hình nhúng local từ HuggingFace (Miễn phí)...")

# 2. Khởi tạo mô hình Embedding chạy local 100%
# Mô hình này hỗ trợ tiếng Việt rất tốt, nhẹ và không cần API Key
embeddings = HuggingFaceEmbeddings(
        model_name="bkai-foundation-models/vietnamese-bi-encoder"
    )
# Đường dẫn lưu database local
PERSIST_DIRECTORY = "./legal_vector_db"
print("✅ Đã cấu hình xong Embedding Local.")