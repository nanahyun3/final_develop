# test_logic 디렉토리

벡터 DB 검색 성능을 테스트하는 파이프라인.  
**흐름: 데이터 생성 → 수동 필터링 → 평가**

---

## 파일 설명

### `make_ground_truth.py`
테스트 데이터셋(쿼리 & 정답 쌍) 자동 생성.  
생성된 쌍을 CSV로 저장.

### `live_or_drop.ipynb`
자동 생성된 쌍을 눈으로 확인하며 비유사한 쌍을 수동으로 드랍.  
최종 결과는 `ground_truth_task2_filtered.csv`에 저장.

### `ground_truth_task2_filtered.csv`
수동 검수를 마친 최종 테스트 데이터셋.

### `evaluate task2.ipynb`
`ground_truth_task2_filtered.csv`의 쿼리를 벡터 DB에 검색해서 정답이 나오는지 확인.
