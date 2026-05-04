# Individual Reflection — Lab 18

**Tên:** Trần Quang Quí  
**Module phụ trách:** M1 — Advanced Chunking Strategies

---

## 1. Đóng góp kỹ thuật

- Module đã implement: `src/m1_chunking.py`
- Các hàm chính đã viết:
  - `chunk_semantic()` — encode câu bằng `all-MiniLM-L6-v2`, tách chunk khi cosine similarity < threshold
  - `chunk_hierarchical()` — tạo parent chunks (2048 chars) → split thành child chunks (256 chars), mỗi child có `parent_id`
  - `chunk_structure_aware()` — regex split theo markdown headers (`#`, `##`, `###`), mỗi section là 1 chunk với `section` trong metadata
  - `compare_strategies()` — chạy cả 4 strategies, in bảng so sánh num_chunks / avg_length / min / max
- Số tests pass: 13/13

---

## 2. Kiến thức học được

- **Khái niệm mới nhất:** Hierarchical (parent-child) chunking — index children nhỏ vào vector DB để embedding chính xác, nhưng trả về parent cho LLM để có đủ context. Slide 8 giải thích rõ tại sao đây là default cho production.
- **Điều bất ngờ nhất:** Basic chunking (split theo `\n\n`) tưởng đơn giản nhưng lại cắt giữa ý, làm embedding mất ngữ nghĩa. Semantic chunking giải quyết đúng bài toán này bằng cách so sánh cosine similarity giữa các câu liên tiếp.
- **Kết nối với bài giảng:** Slide 8 ("3 Chunking Strategies — So sánh") trực tiếp map vào 3 hàm cần implement. Slide 7 ("Ingestion Pipeline") giải thích tại sao chunking là bước fix "Chunking Mismatch" trong OFFLINE failures — garbage in, garbage out.

---

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:** `test_hierarchical_valid_parent_ids` — test kiểm tra `parent_id` của child phải nằm trong tập `parent_id` của parents. Ban đầu tôi lưu `parent_id` vào `metadata` của parent thay vì dùng field `parent_id` của dataclass Chunk.
- **Cách giải quyết:** Đọc lại test kỹ hơn — test dùng `p.metadata.get("parent_id")` để lấy id từ parents, và `c.parent_id` từ children. Cần lưu cùng giá trị `pid` ở cả hai nơi.
- **Thời gian debug:** ~10 phút để đọc test và điều chỉnh logic.

---

## 4. Nếu làm lại

- **Sẽ làm khác:** Đọc test file trước khi implement thay vì đọc sau — test định nghĩa chính xác contract của từng hàm, tiết kiệm thời gian debug.
- **Module muốn thử tiếp:** M2 (Hybrid Search) — vì RRF fusion giữa BM25 và Dense search là phần thú vị nhất của production pipeline, và nó ảnh hưởng trực tiếp đến context_recall theo slide 6.

---

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 3 |
| Problem solving | 4 |
