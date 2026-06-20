import os
import sys
import pickle
from src.config import embeddings, PERSIST_DIRECTORY
from src.data_utils import load_all_legal_docs
from langchain_chroma import Chroma
from langchain_neo4j import Neo4jGraph
from src.config import BM25_INDEX_PATH
from dotenv import load_dotenv
load_dotenv()
# Định nghĩa cấu hình đường dẫn 4 file luật (đã chia nhỏ phần 1 & phần 2)
DANH_SACH_FILE_LUAT = {
    "Luật Đường bộ": "docs\luật đường bộ.docx",
    "Luật Giao thông đường bộ": "docs\Luật Giao thông đường bộ.docx",
    "Luật Trật tự và an toàn GTĐB (Phần 1)": "docs\Luật Trật tự, an toàn giao thông đường bộ_P1.docx",
    "Luật Trật tự và an toàn GTĐB (Phần 2)": "docs\Luật Trật tự, an toàn giao thông đường bộ_P2.docx",
    "Nghị định xử phạt": "docs\_Nghị định xử phạt.docx"
}

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)
def main():
    embedding_model = embeddings
    # 1. Đọc và bóc tách các file điều luật
    all_chunks = load_all_legal_docs(DANH_SACH_FILE_LUAT, embedding_model)
    print(f"\n Tổng số điều luật thu được: {len(all_chunks)}")
    
    if len(all_chunks) == 0:
        print("Không có dữ liệu. Hãy kiểm tra lại đường dẫn file Word.")
        return

    for index, chunk in enumerate(all_chunks):
        # Trích xuất nội dung và metadata từ đối tượng Document của LangChain
        content_text = chunk.page_content
        metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
        
        # Lấy thông tin nguồn và chương dựa trên logic bạn đã xử lý trước đó
        doc_name = metadata.get("source", "Unknown_Doc")
        current_chuong = metadata.get("chuong", "Tổng quan")
        
        # Câu lệnh Cypher ánh xạ cấu trúc phân tầng
        cypher_query = """
        // a. Tạo hoặc nhận diện Node Văn bản gốc
        MERGE (doc:LegalDocument {name: $doc_name})
        
        // b. Tạo hoặc nhận diện Node Chương thuộc văn bản đó
        MERGE (ch:Chapter {name: $chapter_name, doc_source: $doc_name})
        MERGE (ch)-[:BELONGS_TO]->(doc)
        
        // c. Tạo Node cho từng mảnh ngữ nghĩa cụ thể (Chunk)
        CREATE (chunk:DocumentChunk {
            id: $chunk_id,
            content: $content,
            global_index: $global_index
        })
        
        // d. Thiết lập liên kết cứng giữa mảnh và Chương/Văn bản
        CREATE (chunk)-[:PART_OF_CHAPTER]->(ch)
        CREATE (chunk)-[:PART_OF_DOC]->(doc)
        """
        
        params = {
            "doc_name": doc_name,
            "chapter_name": current_chuong,
            "chunk_id": f"chunk_global_{index}",
            "content": content_text,
            "global_index": index
        }
        
        # Thực thi đẩy lên Cloud
        graph.query(cypher_query, params)
        
    print("\n✅ Hoàn thành! Toàn bộ cấu trúc Đồ thị pháp lý đã sẵn sàng trên Neo4j AuraDB.")

if __name__ == "__main__":
    main()