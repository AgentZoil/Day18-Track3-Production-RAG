# Group Report — Lab 18: Production RAG

**Nhóm:** DHA-RAG  
**Ngày:** 2026-05-04

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Trần Quang Quí | M1: Advanced Chunking + M5: Enrichment | ✅ | 13/13 + 10/10 |
| Nhữ Gia Bách | M2: Hybrid Search | ✅ | 5/5 |
| Hoàng Vĩnh Giang | M3: Reranking | ✅ | 5/5 |
| Nguyễn Xuân Tùng | M4: RAGAS Evaluation | ✅ | 4/4 |

## Corpus & Test Set

- **Tài liệu 1:** Tờ khai thuế GTGT Q4/2024 — Công ty CP DHA Surfaces (`BCTC.md`)
- **Tài liệu 2:** Nghị định 13/2023/NĐ-CP về Bảo vệ dữ liệu cá nhân (`Nghi_dinh_13_2023_BVDLCN.md`)
- **Test set:** 10 câu hỏi (5 về BCTC, 5 về Nghị định 13)

## Kết quả RAGAS

*(Cập nhật sau khi chạy `python main.py`)*

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 1.0000 | 0.9917 | -0.0083 |
| Answer Relevancy | 0.0000 | 0.0000 | 0.0000 (*) |
| Context Precision | 0.5750 | 0.9167 | **+0.3417** ✅ |
| Context Recall | 0.7000 | 0.9222 | **+0.2222** ✅ |

*\* `answer_relevancy = 0` do thiếu `OPENAI_API_KEY`.*

## Key Findings

1. **Biggest improvement:** Context Precision tăng +34% (0.575 → 0.917) nhờ Hybrid Search (M2) + Reranking (M3) loại bỏ chunk không relevant
2. **Biggest challenge:** 2 file PDF corpus là ảnh scan — không extract text được trực tiếp, phải convert sang Markdown thủ công
3. **Surprise finding:** Hierarchical chunking (parent 2048 / child 256) phù hợp hơn basic chunking cho văn bản pháp lý vì giữ nguyên context điều khoản

## Presentation Notes (5 phút)

1. **RAGAS scores** (naive vs production): *(cập nhật sau)*
2. **Biggest win** — M2 Hybrid Search + M3 Reranking: Context Precision tăng từ 0.575 lên 0.917 (+34%), Context Recall tăng từ 0.70 lên 0.92 (+22%)
3. **Case study** — `answer_relevancy = 0`: RAGAS dùng OpenAI embedding để tính similarity giữa answer và question → thiếu API key → metric trả về 0. Fix: thêm `OPENAI_API_KEY` vào `.env`
4. **Next optimization nếu có thêm 1 giờ:** Thêm OCR để đọc trực tiếp PDF scan, triển khai M5 Enrichment với OpenAI API để contextual prepend cải thiện retrieval 49%
