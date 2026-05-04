# Task Assignment — Lab 18: Production RAG

**Nhóm:** 4 người · **Tổng điểm:** 100 (60 cá nhân + 40 nhóm)

---

## Phần A — Cá nhân (60 điểm)

| Thành viên | Module | File | Trạng thái |
|------------|--------|------|------------|
| Trần Quang Quí | **M1 — Advanced Chunking** | `src/m1_chunking.py` | 🔲 Todo |
| Nhữ Gia Bách | **M2 — Hybrid Search** | `src/m2_search.py` | 🔲 Todo |
| Hoàng Vĩnh Giang | **M3 — Reranking** | `src/m3_rerank.py` | 🔲 Todo |
| Nguyễn Xuân Tùng | **M4 — RAGAS Evaluation** | `src/m4_eval.py` | 🔲 Todo |

### Chi tiết từng module

**M1 — Quí**
- `chunk_semantic()`, `chunk_hierarchical()`, `chunk_structure_aware()`, `compare_strategies()`
- Test: `pytest tests/test_m1.py`

**M2 — Bách**
- `segment_vietnamese()` — underthesea word_tokenize
- `BM25Search.index()` + `BM25Search.search()` — rank_bm25
- `DenseSearch.index()` + `DenseSearch.search()` — bge-m3 + Qdrant
- `reciprocal_rank_fusion()` — merge BM25 + Dense với RRF
- Test: `pytest tests/test_m2.py`
- **Yêu cầu:** Docker chạy Qdrant — `docker compose up -d`

**M3 — Giang**
- `CrossEncoderReranker._load_model()` — load BAAI/bge-reranker-v2-m3
- `CrossEncoderReranker.rerank()` — score pairs (query, doc) → sort → top-k
- `benchmark_reranker()` — đo avg/min/max latency
- Test: `pytest tests/test_m3.py`

**M4 — Tùng**
- `evaluate_ragas()` — 4 metrics: faithfulness, answer_relevancy, context_precision, context_recall
- `failure_analysis()` — bottom-10, map vào Diagnostic Tree
- Test: `pytest tests/test_m4.py`

---

## Phần B — Nhóm (40 điểm)

**Deadline tập hợp module:** Sau khi mỗi người xong Phần A

### Bước 1 — Integrate pipeline (Quí)
Mở `src/pipeline.py` — pipeline đã viết sẵn, chỉ cần đảm bảo các modules import được:
```
M1 Chunking → M2 Hybrid Search → M3 Reranking → LLM Generate → M4 Evaluation
```
```bash
python src/pipeline.py
```

### Bước 2 — Run RAGAS + tạo report
```bash
python main.py
# Tạo reports/ragas_report.json + naive_baseline_report.json
```

### Bước 3 — Failure Analysis (cả nhóm)
- Đọc `reports/ragas_report.json` → tìm bottom-5 questions
- Đi qua Error Tree (slide 5 — Ngày 18): Output sai → Context sai? → Query rewrite?
- Điền `analysis/failure_analysis.md`

### Bước 4 — Group Report + Presentation
Điền `analysis/group_report.md`, chuẩn bị 4 điểm trình bày (5 phút):
1. RAGAS scores: naive vs production
2. Biggest win: module nào cải thiện nhiều nhất?
3. Case study: 1 question cụ thể + root cause
4. Next step: nếu thêm 1 giờ, optimize gì?

### Bước 5 — Kiểm tra trước nộp
```bash
python check_lab.py
```

---

## Reflection cá nhân (mỗi người tự làm)

Tạo file `analysis/reflections/reflection_[Tên].md` theo template `reflection_TEMPLATE.md`

| Thành viên | File reflection |
|------------|----------------|
| Trần Quang Quí | `reflection_QuangQui.md` 🔲 |
| Nhữ Gia Bách | `reflection_GiaBach.md` 🔲 |
| Hoàng Vĩnh Giang | `reflection_VinhGiang.md` 🔲 |
| Nguyễn Xuân Tùng | `reflection_XuanTung.md` 🔲 |

---

## Checklist nộp bài

- [ ] `src/m1_chunking.py` — tests pass (Quí)
- [ ] `src/m2_search.py` — tests pass (Bách)
- [ ] `src/m3_rerank.py` — tests pass (Giang)
- [ ] `src/m4_eval.py` — tests pass (Xuân Tùng)
- [ ] `reports/ragas_report.json` — có sau khi chạy `python main.py`
- [ ] `analysis/failure_analysis.md` — điền đầy đủ
- [ ] `analysis/group_report.md` — điền đầy đủ
- [ ] `analysis/reflections/` — đủ 4 file reflection
