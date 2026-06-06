# --- FILE check_nghi_dinh.py ---
from src.config import embeddings, PERSIST_DIRECTORY
from langchain_chroma import Chroma

vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

# Lấy toàn bộ dữ liệu ra để check
collection_data = vector_db.get()

print("--- DANH SÁCH CÁC ĐIỀU LẠC VÀO NGHỊ ĐỊNH XỬ PHẠT ---")
count = 0
for i in range(len(collection_data['ids'])):
    meta = collection_data['metadatas'][i]
    if meta.get('source') == "Nghị định xử phạt":
        print(f"-> {meta.get('chuong')} | {meta.get('dieu')}")
        count += 1

print(f"\n Total: Tổng số điều thực tế của Nghị định xử phạt trong DB là: {count}")