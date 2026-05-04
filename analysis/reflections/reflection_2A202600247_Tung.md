# Individual Reflection — Lab 18

**Tên:** Tùng  
**Module phụ trách:** M4 - RAGAS Evaluation

---

## 1. Đóng góp kỹ thuật

- Module đã implement: Module 4 (RAGAS Evaluation) tại file `src/m4_eval.py`.
- Các hàm/class chính đã viết:
  - `evaluate_ragas()`: Chuyển đổi dữ liệu câu hỏi, câu trả lời, context sang dạng `Dataset` của HuggingFace. Tích hợp 4 metrics từ RAGAS (Faithfulness, Answer Relevancy, Context Precision, Context Recall) và chuẩn hóa kiểu dữ liệu trả về sang `float`.
  - `failure_analysis()`: Lọc ra bottom-N kết quả có điểm trung bình thấp nhất. Ứng dụng Diagnostic Tree để đối chiếu metric thấp nhất với các chẩn đoán lỗi (như LLM hallucinating, Missing relevant chunks, ...) và đưa ra đề xuất khắc phục (suggested fix).
- Số tests pass: 4/4 (Pass toàn bộ tiêu chí bài tập cá nhân, nếu chạy với pytest)

## 2. Kiến thức học được

- Khái niệm mới nhất: Cấu trúc hoạt động của `LLM-as-a-Judge` và cách RAGAS định nghĩa toán học đằng sau 4 metrics cốt lõi để đánh giá một hệ thống RAG không cần human label.
- Điều bất ngờ nhất: Sự khắt khe của RAGAS với định dạng schema đầu vào (đổi tên các cột thành `user_input`, `response`, `retrieved_contexts`, `reference`) và kiểu dữ liệu trả về từ RAGAS thường là numpy array/pandas series thay vì kiểu cơ bản của Python, rất dễ gây lỗi crash khi lưu JSON.
- Kết nối với bài giảng (slide nào): Liên quan trực tiếp tới phần "Production RAG Evaluation" (Đánh giá RAG) và slide về sử dụng Diagnostic Tree để phân tích lỗi hệ thống.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: Quá trình trích xuất kết quả metric từ đối tượng result của RAGAS gây ra lỗi `TypeError: Object of type ... is not JSON serializable` khi gọi hàm `json.dump()`. Ngoài ra ban đầu truyền sai tên cột dữ liệu nên RAGAS đánh giá không thành công.
- Cách giải quyết: 
  - Khởi tạo hàm helper `to_float()` và import `numpy` để xử lý ngoại lệ: chuyển đổi các giá trị pandas `Series` hay numpy `float64` về kiểu `float` thuần của Python (sử dụng `np.nan_to_num` để đề phòng giá trị rỗng).
  - Ánh xạ lại chính xác tên cột output của RAGAS vào object `EvalResult`.
- Thời gian debug: Khoảng 30-45 phút cho việc đọc traceback lỗi và thử nghiệm parsing lại điểm số.

## 4. Nếu làm lại

- Sẽ làm khác điều gì: Truyền tham số tuỳ chỉnh mô hình ngôn ngữ (`llm`) và mô hình embedding (`embeddings`) cụ thể của OpenAI thay vì dùng config mặc định của RAGAS, để kiểm soát tốc độ đánh giá và chi phí API token.
- Module nào muốn thử tiếp: Module 5 (Enrichment Pipeline) để trực tiếp xem các phương pháp như Contextual Prepend hay Auto Metadata sẽ cải thiện số điểm RAGAS lên bao nhiêu.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 5 |
| Teamwork | 4 |
| Problem solving | 5 |
