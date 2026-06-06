from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

class CustomHybridRetriever(BaseRetriever):
    vector_retriever: BaseRetriever
    bm25_retriever: BaseRetriever
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        # 1. Gọi cả 2 bộ tìm kiếm để lấy kết quả
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        
        # 2. Gộp kết quả và lọc trùng lặp dựa trên nội dung/điều luật
        seen = set()
        combined_docs = []
        
        for doc in vector_docs + bm25_docs:
            doc_key = (doc.page_content, doc.metadata.get('dieu'))
            if doc_key not in seen:
                seen.add(doc_key)
                combined_docs.append(doc)
                
        return combined_docs[:8]  # Trả về top 8 mảnh chất lượng nhấtfrom typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

class CustomHybridRetriever(BaseRetriever):
    vector_retriever: BaseRetriever
    bm25_retriever: BaseRetriever
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        
        seen = set()
        combined_docs = []
        
        for doc in vector_docs + bm25_docs:
            doc_key = (doc.page_content, doc.metadata.get('dieu'))
            if doc_key not in seen:
                seen.add(doc_key)
                combined_docs.append(doc)
                
        return combined_docs[:8]  