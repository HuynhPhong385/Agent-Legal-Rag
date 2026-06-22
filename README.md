# GraphRAG Luật Giao Thông — Hệ Thống Tra Cứu Pháp Luật Thông Minh

Hệ thống Trợ lý thông minh hỗ trợ tra cứu và tư vấn pháp luật về Giao thông Đường bộ Việt Nam. Dự án áp dụng kiến trúc **Advanced Hybrid GraphRAG (Knowledge Graph + Vector Store)** tiên tiến, kết hợp chặt chẽ giữa đồ thị tri thức và tìm kiếm ngữ nghĩa nhằm bảo toàn trọn vẹn mạch ngữ cảnh phân cấp cấu trúc luật (Chương/Điều/Khoản), khắc phục triệt để hiện tượng ảo giác (Hallucination) của các mô hình ngôn ngữ lớn (LLM).

---

## Tại Sao Lại Là GraphRAG Đối Với Dữ Liệu Luật?

Văn bản pháp luật Việt Nam có cấu trúc phân cấp và mối liên đới cực kỳ chặt chẽ (Khoản phải đi liền với Điều, Điều phải nằm trong Chương và áp dụng cho từng đối tượng phương tiện cụ thể). Cách tiếp cận RAG truyền thống (chỉ dùng Vector Search) thường bẻ vụn văn bản thành các mảnh rời rạc, khiến LLM dễ trích xuất sai mức phạt hoặc áp dụng nhầm loại phương tiện (như nhầm giữa xe máy thông thường và xe máy chuyên dùng).

Hệ thống này giải quyết bài toán bằng cơ chế **Hybrid Retrieval Core**:
1. **Dense Retrieval (ChromaDB):** Định vị nhanh các vùng văn bản có phân phối ngữ nghĩa gần nhất với câu hỏi người dùng dựa trên Vector Embeddings.
2. **Graph Retrieval (Neo4j):** Từ các điểm neo tìm được, hệ thống truy quét đồ thị tri thức để khôi phục trọn vẹn mối quan hệ thực thể, bốc thêm tiêu đề của Điều, Chương và tích hợp ngữ cảnh chính xác nhất trước khi gửi vào LLM sinh câu trả lời.

---

## Tính Năng Nổi Bật

* **Kiến trúc Tìm kiếm Lai (Hybrid Graph-Vector Retrieval):** Sự kết hợp hài hòa giữa `ChromaDB` (Tìm kiếm ngữ nghĩa) và `Neo4j` (Khôi phục mạch phân cấp cấu trúc luật). Đảm bảo hệ thống vừa hiểu được ý định người dùng, vừa trích xuất thông tin có tính hệ thống cao.
* **Tối ưu hóa Truy vấn (Query Rewriting / Transformation):** Sử dụng LLM (Gemini 2.5 Flash) đóng vai trò lớp đệm, tự động biên dịch câu hỏi khẩu ngữ, dân dã của người dùng (Ví dụ: *"vượt đèn đỏ"*, *"uống rượu bia"*) thành các thuật ngữ pháp lý quy chuẩn (Ví dụ: *"không chấp hành hiệu lệnh của đèn tín hiệu giao thông"*) trước khi thực hiện tìm kiếm.
* **Cấu trúc Mô-đun chuẩn Doanh nghiệp (Production-ready):** Mã nguồn được tổ chức sạch sẽ, tường minh. Tách biệt hoàn toàn phần giao diện người dùng UI (`app_web.py`), luồng thực thi RAG cốt lõi (`app_rag.py`), và các mô-đun xử lý chuyên biệt (`src/`) như `generator.py`, `retriever.py`, `data_utils.py`.
* **Prompt Kỹ thuật cao (Context-Grounded & Chain-of-Thought):** Hệ thống áp dụng kỹ thuật Prompt phản biện kết hợp kiểm soát dữ liệu đầu vào nghiêm ngặt, ép mô hình chỉ tư vấn dựa trên ngữ cảnh thực tế bốc từ đồ thị lên, tránh việc mô hình tự bịa đặt mức phạt hoặc trích dẫn sai điều khoản.
* **Giao diện Trực quan & Bộ lọc Động (Dynamic Filtering):** Tích hợp giao diện Streamlit phẳng hiện đại với Sidebar cho phép người dùng tùy chỉnh thu hẹp hoặc mở rộng phạm vi tra cứu theo từng văn bản cụ thể, lọc nâng cao theo loại phương tiện hoặc hành vi vi phạm thời gian thực.

---
![Giao diện ứng dụng]
<img width="1920" height="954" alt="Annotation 2026-06-07 124149" src="https://github.com/user-attachments/assets/625145cb-6877-4dd4-87ac-4cf49f834634" />

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
# Hướng Dẫn Cài Đặt Và Sử Dụng
```
1. Cài đặt môi trường và thư viện
# Cài đặt toàn bộ các thư viện cần thiết (langchain, chroma, streamlit,...)
pip install -r requirements.txt
## Cài đặt dự án dưới dạng Local Package (Editable Mode) để nhận diện gói src/
pip install -e .

2. Cấu hình khóa bảo mật API
Tạo một file tên là .env nằm ngay tại thư mục gốc của dự án (file này đã được chặn trong .gitignore để đảm bảo không bị lộ lên GitHub). Điền mã khóa Gemini API của bạn vào:
GOOGLE_API_KEY=mã_api_key_gemini_của_bạn_ở_đây

3. Nạp dữ liệu văn bản pháp lý (Ingestion)
Đặt toàn bộ các file luật bằng Word (.docx) mà bạn có vào bên trong thư mục docs/. Nếu có dữ liệu cũ, hãy tiến hành xóa thư mục legal_vector_db trước, sau đó chạy lệnh nạp dữ liệu:
python scripts/ingest_data.py
Hệ thống sẽ tiến hành đọc dữ liệu, băm nhỏ theo ngữ nghĩa (Semantic Chunking), tạo cơ sở dữ liệu Vector trên ChromaDB và lưu trữ chỉ mục từ khóa BM25 dưới dạng file .pkl.

4. Khởi động ứng dụng Chatbot
## Lựa chọn 1: Chạy trên giao diện Terminal
python app_rag.py
## Lựa chọn 2: Chạy giao diện Web UI trực quan
python -m streamlit run app_web.py
