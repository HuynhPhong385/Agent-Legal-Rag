# # --- FILE inspect_db.py ---
# from config import embeddings, PERSIST_DIRECTORY
# from langchain_chroma import Chroma

# vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

# # Lấy thông tin tổng quan
# collection_data = vector_db.get()
# print(f"📊 Tổng số lượng vector đang có trong DB: {len(collection_data['ids'])}")

# # In thử 3 phần tử đầu tiên để xem cấu trúc
# print("\n🔍 CHI TIẾT 3 ĐIỀU LUẬT ĐẦU TIÊN TRONG DB:")
# for i in range(min(3, len(collection_data['ids']))):
#     print(f"\n--- [ID: {collection_data['ids'][i]}] ---")
#     print(f"Nguồn (Source): {collection_data['metadatas'][i].get('source')}")
#     print(f"Vị trí: {collection_data['metadatas'][i].get('chuong')} -> {collection_data['metadatas'][i].get('dieu')}")
#     print(f"Nội dung (Text): {collection_data['documents'][i][:200]}...")