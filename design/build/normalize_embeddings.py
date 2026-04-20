"""
임베딩 정규화 일괄 업데이트 스크립트

[트러블슈팅]
- 문제: CLIP / DINOv2 / SigLIP 임베딩 저장 시 L2 정규화 누락
- 발견: embeddings.py, embeddings_dinov2.py, embeddings_siglip.py 코드 리뷰 중 확인
- 원인: encode_image 함수에서 emb / emb.norm(...) 정규화 단계 미적용
- 영향: 벡터 크기(norm) 차이로 코사인 유사도 계산 왜곡
- 해결: 저장된 JSON 파일의 embedding 벡터를 L2 정규화하여 덮어쓰기
  - 코사인 유사도 = (A·B) / (|A||B|) 이며, 정규화 시 |A|=|B|=1이 되어
    내적만으로 정확한 의미적 유사도 비교 가능
"""

import json
import os
import numpy as np
from pathlib import Path

BASE_DIR = "/Users/nanahyun/Documents/GitHub/final_develop/design/data"

FOLDERS = [
    f"{BASE_DIR}/embeddings_original/통합",
    f"{BASE_DIR}/embeddings_sketch/통합",
    f"{BASE_DIR}/embeddings_original_dinov2/통합",
    f"{BASE_DIR}/embeddings_sketch_dinov2/통합",
    f"{BASE_DIR}/embeddings_original_siglip/통합",
    f"{BASE_DIR}/embeddings_sketch_siglip/통합",
]


def normalize(vec: list) -> list:
    arr = np.array(vec, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return vec
    return (arr / norm).tolist()


def process_folder(folder_path: str):
    files = list(Path(folder_path).glob("*.json"))
    total = len(files)
    print(f"\n[{folder_path}]")
    print(f"  총 {total}개 파일 처리 시작...")

    skipped = 0
    updated = 0
    errors = 0

    for i, file_path in enumerate(files, 1):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            emb = data.get("embedding")
            if emb is None:
                print(f"  [SKIP] embedding 키 없음: {file_path.name}")
                skipped += 1
                continue

            norm_val = np.linalg.norm(np.array(emb, dtype=np.float64))
            if abs(norm_val - 1.0) < 1e-6:
                skipped += 1
                continue

            data["embedding"] = normalize(emb)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)

            updated += 1

            if i % 1000 == 0:
                print(f"  진행: {i}/{total} ({i/total*100:.1f}%)")

        except Exception as e:
            print(f"  [ERROR] {file_path.name}: {e}")
            errors += 1

    print(f"  완료 → 업데이트: {updated}, 이미 정규화됨: {skipped}, 오류: {errors}")
    return updated, skipped, errors


if __name__ == "__main__":
    print("=" * 60)
    print("임베딩 L2 정규화 일괄 업데이트")
    print("=" * 60)

    total_updated = 0
    total_skipped = 0
    total_errors = 0

    for folder in FOLDERS:
        if not os.path.exists(folder):
            print(f"\n[SKIP] 폴더 없음: {folder}")
            continue
        u, s, e = process_folder(folder)
        total_updated += u
        total_skipped += s
        total_errors += e

    print("\n" + "=" * 60)
    print("전체 완료")
    print(f"  업데이트: {total_updated}")
    print(f"  이미 정규화됨(스킵): {total_skipped}")
    print(f"  오류: {total_errors}")
    print("=" * 60)
