# Trợ Lý Luật Sư - Giao Thông Việt Nam (Agentic Legal RAG)

Hệ thống Trợ lý thông minh hỗ trợ tra cứu và tư vấn pháp luật về Giao thông Đường bộ Việt Nam. Dự án áp dụng kiến trúc **Hybrid Search RAG (Dense + Sparse)** tiên tiến kết hợp với kỹ thuật Prompt phản biện chuyên sâu, giúp trích xuất chính xác các quy định từ Luật Đường bộ 2024, Luật Trật tự an toàn giao thông đường bộ 2024 và Nghị định 100/2019/NĐ-CP mà không bị hiện tượng ảo giác (Hallucination) của LLM.

## Tính Năng Nổi Bật

- **Lõi tìm kiếm lai (Hybrid Retrieval Core)**: Kết hợp hài hòa giữa tìm kiếm ngữ nghĩa theo Vector (`ChromaDB` + Embeddings) và tìm kiếm từ khóa chính xác (`BM25`). Cơ chế này đảm bảo hệ thống vừa hiểu được ý định người dùng, vừa bốc đúng từ ngữ chuyên ngành trong các văn bản luật.
- **Cấu trúc chuẩn Doanh nghiệp (Production-ready)**: Mã nguồn được bao gói module hóa rõ ràng, phân tách biệt lập giữa logic cốt lõi hệ thống (`src/`), dữ liệu lưu trữ (`data/`, `docs/`) và các kịch bản thực thi (`scripts/`).
- **Prompt kỹ thuật cao (Few-Shot & Chain-of-Thought)**: Hệ thống được cài đặt các ví dụ mẫu phản biện để hướng dẫn mô hình tư duy từng bước độc lập, khắc phục triệt để các lỗi nhận nhầm hoặc râu ông nọ cắm cằm bà kia (ví dụ: phân biệt rõ ràng quy định giữa "xe máy thông thường" và "xe máy chuyên dùng").
- **Giao diện Web Chatbot mượt mà**: Tích hợp giao diện UI trực quan bằng **Streamlit**, mang lại trải nghiệm hỏi đáp pháp lý thân thiện, có lưu giữ lịch sử cuộc trò chuyện (Session State).

---

## Kiến Trúc Thư Mục Dự Án

```text
AGENTIC LEGAL RAG/
├── src/                # Mã nguồn cốt lõi (Core AI Logic)
│   ├── __init__.py     # Đánh dấu Python package chính thống
│   ├── config.py       # Cấu hình đường dẫn, khai báo LLM & Embedding
│   ├── data_utils.py   # Tiền xử lý văn bản & Semantic Chunking
│   ├── prompts.py      # Bộ mẫu câu lệnh Few-Shot & Chain-of-Thought
│   └── retrievers.py   # Bộ truy xuất lai CustomHybridRetriever
├── scripts/            # Các script chạy dòng lệnh (Executable workflows)
│   ├── ingest_data.py  # Script làm sạch, băm nhỏ dữ liệu và nạp vào DB
│   └── tools/          # Thư mục chứa các file nháp, test, debug hệ thống
├── docs/               # Thư mục chứa các file văn bản luật gốc (.docx)
├── requirements.txt    # Danh sách thư viện và phiên bản cụ thể
├── setup.py            # File cấu hình đóng gói dự án ở chế độ Editable
├── app_rag.py          # Ứng dụng chạy chatbot trên Terminal
└── app_web.py          # Ứng dụng chạy giao diện Chatbot trên trình duyệt Web
```
## Hướng Dẫn Cài Đặt Và Sử Dụng
1. Cài đặt môi trường và thư viện
# Cài đặt toàn bộ các thư viện cần thiết (langchain, chroma, streamlit,...)
pip install -r requirements.txt

# Cài đặt dự án dưới dạng Local Package (Editable Mode) để nhận diện gói src/
pip install -e .
2. Cấu hình khóa bảo mật API
Tạo một file tên là .env nằm ngay tại thư mục gốc của dự án (file này đã được chặn trong .gitignore để đảm bảo không bị lộ lên GitHub). Điền mã khóa Gemini API của bạn vào:

Đoạn mã
GOOGLE_API_KEY=mã_api_key_gemini_của_bạn_ở_đây
3. Nạp dữ liệu văn bản pháp lý (Ingestion)
Đặt toàn bộ các file luật bằng Word (.docx) mà bạn có vào bên trong thư mục docs/. Nếu có dữ liệu cũ, hãy tiến hành xóa thư mục legal_vector_db trước, sau đó chạy lệnh nạp dữ liệu:

Bash
python scripts/ingest_data.py
Hệ thống sẽ tiến hành đọc dữ liệu, băm nhỏ theo ngữ nghĩa (Semantic Chunking), tạo cơ sở dữ liệu Vector trên ChromaDB và lưu trữ chỉ mục từ khóa BM25 dưới dạng file .pkl.

4. Khởi động ứng dụng Chatbot
# Lựa chọn 1: Chạy trên giao diện Terminal
python app_rag.py

# Lựa chọn 2: Chạy giao diện Web UI trực quan
python -m streamlit run app_web.py