# --- FILE read_raw_dieu6.py ---
import re
from langchain_community.document_loaders import Docx2txtLoader

# Thay đường dẫn này bằng đường dẫn chính xác tới file Nghị định 168 của bạn
FILE_PATH = "docs\_Nghị định xử phạt.docx"
print("🔍 Đang đọc trực tiếp file Word...")
loader = Docx2txtLoader(FILE_PATH)
raw_docs = loader.load()
full_text = raw_docs[0].page_content

# Tìm kiếm vị trí xung quanh chữ Điều 6 hoặc các đoạn lân cận để xem định dạng
print("\n--- KIỂM TRA ĐOẠN ĐẦU CỦA ĐIỀU 6 VÀ CÁC ĐIỀU XUNG QUANH ---")

# Tìm tất cả các dòng có chứa chữ "Điều" từ 4 đến 8 để xem cấu trúc viết
for line in full_text.split('\n'):
    if re.match(r'^\s*(Điều|điều|ĐIỀU)\s+[45678]', line) or "Điều 6" in line:
        print(f"👉 Dòng thô Python đọc được: {repr(line)}")