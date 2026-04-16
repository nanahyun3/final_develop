# 평가 코드(거절 데이터) 디렉토리

실제 거절된 디자인 데이터로 벡터 DB 검색 성능을 평가하는 파이프라인.  
**흐름: 평가 DB 구성 → 평가 실행**

---

## 파일 설명

### `build_eval_db.py`
기존 벡터 DB를 복사하고, `_similar` 정답 이미지를 추가해 평가 전용 DB 구성.  
최초 1회만 실행.

###  `evaluate.ipynb`
쿼리(출원 디자인)를 평가 DB에 검색해서 정답(`_similar`)이 top-K 안에 있는지 확인.  
결과는 `eval_results/`에 CSV 3종으로 저장.

### `rejected_designs(통합본)/`
테스트 데이터. 파일명 규칙:
- 쿼리: `{출원번호}_img1.jpg`
- 정답: `{출원번호}_img1_similar.jpg`, `{출원번호}_img1_similar2.jpg`

### `eval_results/`
평가 결과 CSV 저장 위치.
- `detail.csv` — 이미지별 hit 여부 및 rank
- `per_app.csv` — 출원번호별 요약
- `summary.csv` — 전체 HitRate@K

---

## 실행 순서

```bash
python build_eval_db.py   # 최초 1회
python evaluate.py
```
