# Failure Analysis — Lab 18: Production RAG

**Nhóm:** DHA-RAG  
**Thành viên:** Trần Quang Quí (M1+M5) · Nhữ Gia Bách (M2) · Hoàng Vĩnh Giang (M3) · Nguyễn Xuân Tùng (M4)

---

## RAGAS Scores

*(Cập nhật sau khi chạy `python main.py`)*

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 1.0000 | 0.9917 | -0.0083 |
| Answer Relevancy | 0.0000 | 0.0000 | 0.0000 (*) |
| Context Precision | 0.5750 | 0.9167 | +0.3417 ✅ |
| Context Recall | 0.7000 | 0.9222 | +0.2222 ✅ |

*\* `answer_relevancy = 0` do thiếu `OPENAI_API_KEY` — metric này cần OpenAI embedding để tính similarity.*

---

## Bottom-5 Failures

*(#1–#3 phân tích dự đoán dựa trên corpus; #4–#5 cập nhật sau khi có `ragas_report.json`)*

### #1
- **Question:** Thuế GTGT phải nộp trong kỳ của DHA Surfaces là bao nhiêu?
- **Expected:** 52,133,830 đồng
- **Got:** *(TBD)*
- **Worst metric:** context_recall — số liệu cụ thể dễ bị miss khi chunk bị cắt
- **Error Tree:** Output sai → Context đúng? **Không** → Query rewrite OK? **Có** → Fix Ingestion
- **Root cause:** Basic chunking cắt bảng số liệu giữa chừng → chunk không chứa đủ ô [40a]
- **Suggested fix:** Dùng `chunk_structure_aware()` để giữ nguyên bảng số liệu trong 1 chunk

### #2
- **Question:** Bên Kiểm soát dữ liệu cá nhân là gì theo Nghị định 13/2023?
- **Expected:** Tổ chức, cá nhân quyết định mục đích và phương tiện xử lý dữ liệu cá nhân
- **Got:** *(TBD)*
- **Worst metric:** answer_relevancy — định nghĩa pháp lý dài, LLM trả lời lan man
- **Error Tree:** Output sai → Context đúng? **Có** → Fix G: prompt chưa constrain độ dài
- **Root cause:** System prompt không yêu cầu trả lời ngắn gọn cho câu hỏi định nghĩa
- **Suggested fix:** Thêm "Trả lời ngắn gọn trong 1-2 câu" vào system prompt

### #3
- **Question:** Nghị định 13/2023 áp dụng cho những đối tượng nào?
- **Expected:** 4 nhóm đối tượng (VN, nước ngoài tại VN, VN ở nước ngoài, nước ngoài liên quan)
- **Got:** *(TBD)*
- **Worst metric:** faithfulness — LLM có thể bịa thêm đối tượng không có trong văn bản
- **Error Tree:** Output sai → Context đúng? **Có** → Fix G: temperature cao hoặc prompt yếu
- **Root cause:** LLM hallucinate thêm đối tượng ngoài 4 nhóm quy định
- **Suggested fix:** Lower temperature, thêm "Chỉ liệt kê đúng những gì có trong context, không thêm bớt"

### #4
- **Question:** Dữ liệu cá nhân nhạy cảm theo Nghị định 13/2023 bao gồm những loại nào?
- **Expected:** 10 loại (quan điểm chính trị, sức khỏe, nguồn gốc chủng tộc, di truyền, sinh học, tình dục, tội phạm, ngân hàng, định vị...)
- **Got:** *(TBD)*
- **Worst metric:** context_precision — nhiều chunk trả về nhưng không phải tất cả đều relevant
- **Error Tree:** Output sai → Context đúng? **Có nhưng nhiễu** → Fix R: thiếu reranking hiệu quả
- **Root cause:** Câu hỏi liệt kê dài → retrieval trả về nhiều chunk không liên quan → context_precision thấp
- **Suggested fix:** Tăng reranking top_k, thêm metadata filter `category="policy"`

### #5
- **Question:** Xử lý dữ liệu cá nhân được định nghĩa như thế nào trong Nghị định 13/2023?
- **Expected:** Thu thập, ghi, phân tích, lưu trữ, chỉnh sửa, công khai, truy cập, mã hóa, sao chép, chia sẻ, xóa, hủy...
- **Got:** *(TBD)*
- **Worst metric:** faithfulness — định nghĩa có nhiều động từ, LLM dễ bỏ sót hoặc thêm
- **Error Tree:** Output sai → Context đúng? **Có** → Fix G: LLM tóm tắt sai danh sách dài
- **Root cause:** Danh sách 15+ hành động → LLM tóm tắt bỏ mất một số hành động → faithfulness thấp
- **Suggested fix:** Yêu cầu LLM "Liệt kê đầy đủ, không tóm tắt" trong prompt khi câu hỏi là định nghĩa

---

## Case Study (cho presentation)

**Question chọn phân tích:** "Thuế GTGT phải nộp trong kỳ của DHA Surfaces là bao nhiêu?"

**Error Tree walkthrough:**
1. Output đúng? → **Không** (số liệu sai hoặc thiếu)
2. Context đúng? → **Không** (chunk bị cắt giữa bảng số liệu, không chứa ô [40a] = 52,133,830)
3. Query rewrite OK? → **Có** (query đủ rõ, không cần rewrite)
4. Fix ở bước: **Ingestion** → chuyển từ `chunk_basic()` sang `chunk_structure_aware()` để giữ nguyên bảng số liệu trong 1 chunk

**Nếu có thêm 1 giờ, sẽ optimize:**
- Implement OCR để đọc trực tiếp PDF scan thay vì convert thủ công
- Thêm M5 Enrichment với OpenAI API: contextual prepend giảm 49% retrieval failure (Anthropic benchmark)
- Thêm metadata filter theo loại tài liệu (`category="finance"` vs `category="policy"`) để tăng context_precision
