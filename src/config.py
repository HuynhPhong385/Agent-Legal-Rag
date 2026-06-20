# --- FILE config.py (CẬP NHẬT) ---
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv()

embeddings = HuggingFaceEmbeddings(
        model_name="bkai-foundation-models/vietnamese-bi-encoder"
    )
# Đường dẫn lưu database local
PERSIST_DIRECTORY = "./legal_vector_db"
# Thêm vào file config.py của bạn
BM25_INDEX_PATH = "legal_vector_db/bm25_index.pkl"
print(" Đã cấu hình xong Embedding Local.")