# Individual Reflection — Lab 18

**Tên:** Nhữ Gia Bách  
**Module phụ trách:** M2

## 1. Đóng góp kỹ thuật

- Module đã implement: `src/m2_search.py` - Hybrid Search cho pipeline RAG.
- Các hàm/class chính đã viết:
  - `segment_vietnamese()` dùng `underthesea.word_tokenize()` và có fallback an toàn.
  - `BM25Search.index()` và `BM25Search.search()` để build và truy xuất sparse retrieval.
  - `DenseSearch.index()` và `DenseSearch.search()` để index/query trên Qdrant, kèm fallback local khi môi trường không có Qdrant hoặc model cache.
  - `reciprocal_rank_fusion()` để hợp nhất kết quả BM25 và dense retrieval.
- Số tests pass: `5/5`

## 2. Kiến thức học được

- Khái niệm mới nhất: hybrid retrieval, BM25 tokenization cho tiếng Việt, và Reciprocal Rank Fusion để trộn thứ hạng từ nhiều retriever.
- Điều bất ngờ nhất: dense search không tự động “tốt hơn” BM25 trong mọi trường hợp; với dữ liệu ngắn và query rõ ràng, BM25 thường rất mạnh và ổn định.
- Kết nối với bài giảng (slide nào): phần Retrieval trong bài về Production RAG, đặc biệt là đoạn so sánh sparse retrieval, dense retrieval và cách kết hợp bằng hybrid search.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: làm cho module chạy được cả trong môi trường đầy đủ phụ thuộc lẫn môi trường thiếu `underthesea`, `rank_bm25`, `sentence-transformers`, hoặc Qdrant.
- Cách giải quyết:
  - Viết fallback BM25 nội bộ nếu không có `rank_bm25`.
  - Viết fallback encoder deterministic cho dense search nếu không tải được model.
  - Giữ local mirror cho collection để search không bị chết khi Qdrant không sẵn.
  - Test tay từng bước `index -> search -> top-k` trước khi ghép vào pipeline.
- Thời gian debug: khoảng 1-2 giờ, chủ yếu ở phần ổn định môi trường và xử lý fallback.

## 4. Nếu làm lại

- Sẽ làm khác điều gì: tách riêng lớp adapter cho Qdrant và lớp in-memory fallback sớm hơn để code rõ hơn, bớt logic điều kiện trong cùng một file.
- Module nào muốn thử tiếp: M3 Reranking, vì reranker có thể cải thiện mạnh chất lượng context sau hybrid search.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 5 |
| Teamwork | 5 |
| Problem solving | 5 |
