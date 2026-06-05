import os
import re
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document

def split_legal_document(full_text, doc_name):
    legal_documents = []
    
    # 1. CHUẨN HÓA VĂN BẢN
    cleaned_text = re.sub(r'[ \t]+', ' ', full_text)
    
    # 2. TÁCH THEO ĐIỀU (Dựa vào chữ Điều ở đầu dòng)
    dieu_pattern = r'(?=\n\s*(?:Điều|điều|ĐIỀU)\s+\d+)'
    dieu_chunks = re.split(dieu_pattern, cleaned_text)
    
    current_chuong = "Tổng quan"
    
    for d_chunk in dieu_chunks:
        d_chunk = d_chunk.strip()
        if not d_chunk:
            continue
            
        # Cập nhật Chương nếu có
        chuong_match = re.search(r'\n?\s*(?:Chương|CHƯƠNG)\s+[IVXLCDM\d]+.*', d_chunk)
        if chuong_match:
            current_chuong = chuong_match.group(0).split('\n')[0].strip()
            
        if not re.match(r'^(?:Điều|điều|ĐIỀU)\s+\d+', d_chunk):
            continue
            
        # Lấy tiêu đề Điều
        dieu_title_match = re.match(r'^(?:Điều|điều|ĐIỀU)\s+\d+[^\n]*', d_chunk)
        dieu_title = dieu_title_match.group(0).strip() if dieu_title_match else "Không rõ tiêu đề"
        
        # Nếu là Nghị định và Điều này quá dài, ta chia nhỏ theo cụm ký tự (Kích thước khoảng 1500 ký tự)
        # PHÂN TÁCH SIÊU TỐI ƯU CHO NGHỊ ĐỊNH (Băm nhỏ + Gối đầu ngữ cảnh)
        if "Nghị định" in doc_name:
            lines = d_chunk.split('\n')
            current_chunk_text = dieu_title + "\n"
            part_count = 1
            
            for line in lines:
                if line.strip() == dieu_title:
                    continue
                current_chunk_text += line + "\n"
                
                # Băm nhỏ hẳn xuống 700 ký tự để cô đọng hành vi vi phạm
                if len(current_chunk_text) > 700:
                    doc = Document(
                        page_content=current_chunk_text.strip(),
                        metadata={
                            "source": doc_name,
                            "chuong": current_chuong,
                            "dieu": f"{dieu_title} (Phần {part_count})",
                        }
                    )
                    legal_documents.append(doc)
                    
                    # KỸ THUẬT OVERLAP: Giữ lại 2 dòng cuối làm ngữ cảnh gối đầu cho mảnh sau
                    last_lines = current_chunk_text.strip().split('\n')[-2:]
                    current_chunk_text = dieu_title + "\n" + "\n".join(last_lines) + "\n"
                    part_count += 1
            
            if len(current_chunk_text) > len(dieu_title) + 20:
                doc = Document(
                    page_content=current_chunk_text.strip(),
                    metadata={
                        "source": doc_name,
                        "chuong": current_chuong,
                        "dieu": f"{dieu_title} (Phần {part_count})",
                    }
                )
                legal_documents.append(doc)
        else:
            # Nếu đoạn ngắn hoặc là file Luật: Giữ nguyên nguyên Điều
            doc = Document(
                page_content=d_chunk,
                metadata={
                    "source": doc_name,
                    "chuong": current_chuong,
                    "dieu": dieu_title,
                }
            )
            legal_documents.append(doc)
            
    return legal_documents

def load_all_legal_docs(files_config):
    all_chunks = []
    for key_name, file_path in files_config.items():
        if not os.path.exists(file_path):
            print(f"❌ Cảnh báo: Không tìm thấy file tại: {file_path}")
            continue
            
        print(f"📖 Đang bóc tách và phân cấp cấu trúc: {key_name}...")
        doc_name = re.sub(r'\s*\(Phần\s+\d+\)', '', key_name).strip()
        
        loader = Docx2txtLoader(file_path)
        raw_docs = loader.load()
        full_text = raw_docs[0].page_content
        
        cleaned_text = "\n" + full_text
        chunks_from_doc = split_legal_document(cleaned_text, doc_name)
        print(f"   -> Thành công: Tạo ra {len(chunks_from_doc)} mảnh ngữ cảnh tối ưu.")
        all_chunks.extend(chunks_from_doc)
        
    return all_chunks