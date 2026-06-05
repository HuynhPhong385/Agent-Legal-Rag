import os
import re
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document

def split_by_div_and_program(full_text, doc_name):
    """
    Hàm bóc tách văn bản nâng cấp:
    - Bắt được cả 'Điều', 'điều', 'ĐIỀU'.
    - Chấp nhận mọi khoảng trắng (space, tab) sau chữ Điều.
    - Loại bỏ chunk rác/chunk tổng quan ở đầu file.
    """
    # Pattern nâng cấp: Bắt 'Điều/điều/ĐIỀU', chấp nhận nhiều khoảng trắng \s+ 
    # và theo sau là số, kết thúc bằng dấu chấm, dấu cách hoặc dấu hai chấm
    dieu_pattern = r'(?=\n(?:Điều|điều|ĐIỀU)\s+\d+[\.\s:])'
    
    chunks = re.split(dieu_pattern, full_text)
    
    legal_documents = []
    current_chuong = "Chương mở đầu / Tổng quan"
    
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
            
        # Kiểm tra xem đoạn chunk này có thực sự bắt đầu bằng chữ "Điều" không
        # Nếu không bắt đầu bằng "Điều" (tức là phần căn cứ pháp lý ở đầu file), ta bỏ qua
        if not re.match(r'^(?:Điều|điều|ĐIỀU)\s+\d+', chunk):
            # Tuy nhiên, vẫn tranh thủ quét xem đoạn đầu này có chứa tên Chương nào không
            chuong_match = re.search(r'(?:Chương|CHƯƠNG)\s+[IVXLCDM\d]+.*', chunk)
            if chuong_match:
                current_chuong = chuong_match.group(0).split('\n')[0].strip()
            continue # Bỏ qua không thêm chunk rác này vào danh sách Điều
            
        # Cập nhật Chương hiện tại nếu trong Điều có chứa thông tin Chương mới
        chuong_match = re.search(r'(?:Chương|CHƯƠNG)\s+[IVXLCDM\d]+.*', chunk)
        if chuong_match:
            current_chuong = chuong_match.group(0).split('\n')[0].strip()
        
        # Trích xuất tên Điều
        dieu_title_match = re.match(r'(?:Điều|điều|ĐIỀU)\s+\d+[\.\s:][^\n]*', chunk)
        dieu_title = dieu_title_match.group(0).strip() if dieu_title_match else "Tổng quan mục"
        
        doc = Document(
            page_content=chunk,
            metadata={
                "source": doc_name,
                "chuong": current_chuong,
                "dieu": dieu_title,
            }
        )
        legal_documents.append(doc)
        
    return legal_documents

def load_all_legal_docs(files_config):
    """
    Hàm duyệt qua danh sách file, tự động gộp tên nguồn nếu file bị chia phần.
    """
    all_chunks = []
    
    for key_name, file_path in files_config.items():
        if not os.path.exists(file_path):
            print(f"❌ Cảnh báo: Không tìm thấy file tại: {file_path}")
            continue
            
        print(f"📖 Đang xử lý: {key_name}...")
        
        # Chuẩn hóa tên nguồn: Nếu tên có chứa "(Phần...)" thì cắt bỏ để đồng nhất metadata
        # Ví dụ: "Luật Trật tự và an toàn GTĐB (Phần 1)" -> "Luật Trật tự và an toàn GTĐB"
        doc_name = re.sub(r'\s*\(Phần\s+\d+\)', '', key_name).strip()
        
        # Load text từ file docx
        loader = Docx2txtLoader(file_path)
        raw_docs = loader.load()
        full_text = raw_docs[0].page_content
        
        # Làm sạch khoảng trắng thừa
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', full_text).strip()
        
        # Cắt nhỏ theo Điều và truyền tên doc_name đã chuẩn hóa vào metadata
        chunks_from_doc = split_by_div_and_program(cleaned_text, doc_name)
        print(f"   -> Trích xuất được {len(chunks_from_doc)} điều.")
        
        all_chunks.extend(chunks_from_doc)
        
    print(f"\n✅ Hoàn thành! Tổng số Điều trích xuất được từ tất cả các file: {len(all_chunks)}")
    return all_chunks


# --- CẤU HÌNH ĐƯỜNG DẪN 4 FILE CỦA BẠN ---
# Bạn hãy thay đổi đường dẫn (path) chuẩn xác tới các file trên máy của bạn
DANH_SACH_FILE_LUAT = {
    "Luật Đường bộ": "docs\luật đường bộ.docx",
    "Luật Giao thông đường bộ": "docs\Luật Giao thông đường bộ.docx",
    "Luật Trật tự và an toàn GTĐB (Phần 1)": "docs\Luật Trật tự, an toàn giao thông đường bộ_P1.docx",
    "Luật Trật tự và an toàn GTĐB (Phần 2)": "docs\Luật Trật tự, an toàn giao thông đường bộ_P2.docx",
    "Nghị định xử phạt": "docs\_Nghị định xử phạt.docx"
}

# --- CHẠY PIPELINE ---
kho_du_lieu_phap_ly = load_all_legal_docs(DANH_SACH_FILE_LUAT)

# Kiểm tra thử dữ liệu sau khi gộp
if kho_du_lieu_phap_ly:
    print("\n--- Thử in thông tin của một chunk bất kỳ trong kho ---")
    print(f"Nguồn: {kho_du_lieu_phap_ly[100].metadata['source']}")
    print(f"Vị trí: {kho_du_lieu_phap_ly[100].metadata['chuong']} -> {kho_du_lieu_phap_ly[100].metadata['dieu']}")
    print(f"Nội dung đoạn đầu:\n{kho_du_lieu_phap_ly[100].page_content[:200]}...")