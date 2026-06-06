import os
import re
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document

def split_legal_document_semantic(full_text, doc_name, embedding_model):
    # 1. Chuẩn hóa văn bản trước
    cleaned_text = re.sub(r'[ \t]+', ' ', full_text)
    cleaned_text = cleaned_text.replace('\xa0', ' ')
    
    # Tách sơ bộ theo Điều để giữ metadata chính xác cho từng cụm lớn
    dieu_pattern = r'(?=\n\s*(?:Điều|điều|ĐIỀU)\s+\d+)'
    dieu_chunks = re.split(dieu_pattern, cleaned_text)
    
    semantic_splitter = SemanticChunker(
        embedding_model, 
        breakpoint_threshold_type="percentile", # Ngắt khi độ lệch ngữ nghĩa đạt ngưỡng %
        breakpoint_threshold_amount=85         # Thường từ 85-95% là đẹp với văn bản luật
    )
    
    legal_documents = []
    current_chuong = "Tổng quan"
    
    for d_chunk in dieu_chunks:
        d_chunk = d_chunk.strip()
        if not d_chunk:
            continue
            
        # Xác định Chương
        chuong_match = re.search(r'\n?\s*(?:Chương|CHƯƠNG)\s+[IVXLCDM\d]+.*', d_chunk)
        if chuong_match:
            current_chuong = chuong_match.group(0).split('\n')[0].strip()
            
        if not re.match(r'^(?:Điều|điều|ĐIỀU)\s+\d+', d_chunk):
            continue
            
        # Lấy tiêu đề Điều
        dieu_title_match = re.match(r'^(?:Điều|điều|ĐIỀU)\s+\d+[^\n]*', d_chunk)
        dieu_title = dieu_title_match.group(0).strip() if dieu_title_match else "Không rõ tiêu đề"
        
        # Tiến hành băm nhỏ bằng Semantic Chunker dựa trên ngữ nghĩa của câu
        sub_chunks = semantic_splitter.split_text(d_chunk)
        
        for idx, chunk_text in enumerate(sub_chunks):
            # Ép tiêu đề Điều vào đầu mảnh nếu mảnh đó bị mất tiêu đề
            if dieu_title not in chunk_text:
                full_content = f"{dieu_title}\n{chunk_text}"
            else:
                full_content = chunk_text
                
            doc = Document(
                page_content=full_content.strip(),
                metadata={
                    "source": doc_name,
                    "chuong": current_chuong,
                    "dieu": f"{dieu_title} (Mảnh số {idx+1})",
                }
            )
            legal_documents.append(doc)
            
    return legal_documents

def load_all_legal_docs(files_config, embedding_model):
    all_chunks = []
    for key_name, file_path in files_config.items():
        if not os.path.exists(file_path):
            print(f" Cảnh báo: Không tìm thấy file tại: {file_path}")
            continue
            
        print(f" Đang bóc tách và phân cấp cấu trúc: {key_name}...")
        doc_name = re.sub(r'\s*\(Phần\s+\d+\)', '', key_name).strip()
        
        loader = Docx2txtLoader(file_path)
        raw_docs = loader.load()
        full_text = raw_docs[0].page_content
        
        cleaned_text = "\n" + full_text
        chunks_from_doc = split_legal_document_semantic(cleaned_text, doc_name, embedding_model)
        print(f"   -> Thành công: Tạo ra {len(chunks_from_doc)} mảnh ngữ cảnh tối ưu.")
        all_chunks.extend(chunks_from_doc)
        
    return all_chunks