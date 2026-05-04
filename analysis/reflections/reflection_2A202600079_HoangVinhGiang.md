# Individual Reflection — Lab 18

**Tên:** Hoàng Vĩnh Giang 
**Module phụ trách:** M3 — Reranking

---

## 1. Đóng góp kỹ thuật

- Module đã implement: `src/m3_rerank.py`
- Các hàm/class chính đã viết:
  - `CrossEncoderReranker` — class rerank top-20 → top-k bằng cross-encoder, hỗ trợ cả FlagEmbedding và sentence-transformers
  - `RerankResult` — dataclass lưu kết quả rerank
  - `benchmark_reranker` — đo latency rerank
- Số tests pass: 5/5 (theo `tests/test_m3.py`)

---

## 2. Kiến thức học được

- Khái niệm mới nhất: Cross-encoder reranking — dùng mô hình so sánh trực tiếp cặp (query, doc) để rerank, cho kết quả tốt hơn dense search nhưng chậm hơn nhiều.
- Điều bất ngờ nhất: FlagEmbedding cho tốc độ nhanh hơn sentence-transformers, nhưng cần xử lý fallback khi không cài đúng package.
- Kết nối với bài giảng (slide nào): Slide 10 ("Reranking: Cross-encoder vs Bi-encoder") — lý thuyết và trade-off latency/accuracy.

---

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: Xử lý fallback giữa hai thư viện FlagEmbedding và sentence-transformers, do môi trường cài đặt khác nhau.
- Cách giải quyết: Dùng try/except để import, test kỹ cả hai trường hợp.
- Thời gian debug: ~20 phút.

---

## 4. Nếu làm lại

- Sẽ làm khác điều gì: Viết test cho cả trường hợp FlagEmbedding không có sẵn, để CI/CD không bị fail do thiếu package.
- Module nào muốn thử tiếp: M5 (Enrichment) — vì muốn thử nghiệm các kỹ thuật làm giàu chunk trước khi embedding.

---

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 3 |
| Problem solving | 4 |
