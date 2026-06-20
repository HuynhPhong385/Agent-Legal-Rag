import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_neo4j import Neo4jGraph
from src.config import embeddings

load_dotenv()

class HybridVectorGraphRetriever:
    def __init__(self):
        # 1. Kết nối tới Vector DB (Chroma)
        self.vector_db = Chroma(
            persist_directory="legal_vector_db",
            embedding_function=embeddings
        )
        
        # 2. Kết nối tới Graph DB (Neo4j Local)
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            refresh_schema=False
        )

    def retrieve(self, query: str, top_k: int = 2) -> list:
        """
        Hàm lai truy vấn: Tìm bằng Chroma trước, sau đó dùng Neo4j khôi phục mạch văn bản qua global_index.
        """
        print(f"🔍 [Vector Search] Đang quét các mảnh tương đồng cho câu hỏi: '{query}'...")
        vector_results = self.vector_db.similarity_search(query, k=top_k)
        
        expanded_contexts = []
        
        print(f"🧠 [Graph Search] Đang khôi phục mạch ngữ cảnh từ Neo4j cho {len(vector_results)} mảnh...")
        for doc in vector_results:
            # Lấy thông tin trích dẫn từ metadata của Chroma
            dieu_title = doc.metadata.get("dieu", "Không rõ Điều")
            source_doc = doc.metadata.get("source", "Không rõ nguồn")
            chuong_title = doc.metadata.get("chuong", "Không rõ chương")
            
            # Lấy vị trí index của mảnh hiện tại
            current_index = doc.metadata.get("global_index")
            full_context = doc.page_content # Ngữ cảnh mặc định từ Chroma
            
            # Nếu có global_index, ta dùng Neo4j lấy mảnh hiện tại, mảnh trước và mảnh sau để tạo mạch văn bản liên tục
            if current_index is not None:
                cypher_query = """
                MATCH (c:DocumentChunk)
                WHERE c.global_index >= $start_idx AND c.global_index <= $end_idx
                RETURN c.content AS content
                ORDER BY c.global_index ASC
                """
                try:
                    # Mở rộng ngữ cảnh: lấy từ (index - 1) đến (index + 1) để tránh mất đầu mất đuôi của Điều luật
                    start_idx = max(0, current_index - 1)
                    end_idx = current_index + 1
                    
                    graph_results = self.graph.query(cypher_query, {"start_idx": start_idx, "end_idx": end_idx})
                    if graph_results:
                        # Ghép các nội dung văn bản (`content`) thu được từ Neo4j
                        full_context = "\n\n".join([res["content"] for res in graph_results if res["content"]])
                except Exception as e:
                    print(f"⚠️ Lỗi truy vấn Neo4j: {e}")
            
            enriched_doc = {
                "source": source_doc,
                "chuong": chuong_title,
                "dieu": dieu_title,
                "content": full_context
            }
            expanded_contexts.append(enriched_doc)
            
        return expanded_contexts