import os
from config import embeddings, PERSIST_DIRECTORY
from data_utils import load_all_legal_docs
from langchain_chroma import Chroma

# Định nghĩa cấu hình đường dẫn 4 file luật (đã chia nhỏ phần 1 & phần 2)
DANH_SACH_FILE_LUAT = {
    "Luật Đường bộ": "docs\luật đường bộ.docx",
    "Luật Giao thông đường bộ": "docs\Luật Giao thông đường bộ.docx",
    "Luật Trật tự và an toàn GTĐB (Phần 1)": "docs\Luật Trật tự, an toàn giao thông đường bộ_P1.docx",
    "Luật Trật tự và an toàn GTĐB (Phần 2)": "docs\Luật Trật tự, an toàn giao thông đường bộ_P2.docx",
    "Nghị định xử phạt": "docs\_Nghị định xử phạt.docx"
}

def main():
    print("=== BẮT ĐẦU PIPELINE XỬ LÝ VÀ NẠP DỮ LIỆU ===")
    
    # 1. Đọc và bóc tách các file điều luật
    kho_du_lieu_phap_ly = load_all_legal_docs(DANH_SACH_FILE_LUAT)
    print(f"\n Tổng số điều luật thu được để nhúng: {len(kho_du_lieu_phap_ly)}")
    
    if len(kho_du_lieu_phap_ly) == 0:
        print("❌ Không có dữ liệu để nạp. Hãy kiểm tra lại đường dẫn file Word.")
        return

    # 2. Tạo Vector DB với Gemini Embeddings và lưu trữ cứng vào thư mục local
    print(f"\n🔄 Đang gửi dữ liệu sang Google AI Studio để nhúng vector...")
    print(f"📦 Đang lưu trữ dữ liệu vào Chroma DB tại: {PERSIST_DIRECTORY}...")
    
    vector_db = Chroma.from_documents(
        documents=kho_du_lieu_phap_ly,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    
    print("✅ TIẾN TRÌNH HOÀN THÀNH MỸ MÃN!")
    
    # # 3. Kiểm tra nhanh thử nghiệm tìm kiếm
    # print("\n🔍 Thử nghiệm tìm kiếm nhanh với cơ sở dữ liệu mới:")
    # test_query = "Xử phạt nồng độ cồn xe máy"
    # results = vector_db.similarity_search(test_query, k=1)
    # if results:
    #     print(f" -> Kết quả tìm thấy phù hợp nhất thuộc: {results[0].metadata['source']}")
    #     print(f" -> Chi tiết: {results[0].metadata['dieu']}")

if __name__ == "__main__":
    main()