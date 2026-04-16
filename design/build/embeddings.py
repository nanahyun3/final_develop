import json
import requests
import os
from pathlib import Path
from PIL import Image
from io import BytesIO
import torch
import numpy as np
import sys
import cv2
import clip
import time

'''
각 도면 json파일에서 이미지를 다운받는다. (-> "data/images" 폴더에 저장된다.)

다운받은 이미지를 불러와 두 가지 방식으로 CLIP(ViT-B/32) 임베딩 벡터(512차원)를 생성한다.

1. 원본 임베딩
   - 원본 이미지 그대로 임베딩
   - "data/embeddings_original" 폴더에 저장

2. 스케치 임베딩
   - Canny Edge Detection으로 전처리 (GaussianBlur → Gradient → NMS → Double Threshold → Edge Tracking)
   - 스케치 이미지 → "data/images_sketch" 폴더에 저장
   - 스케치 임베딩 → "data/embeddings_sketch" 폴더에 저장
'''
'''
# CLIP 동적 로드 - 최초 1회만
try:
    import clip
except ImportError:
    print("CLIP 패키지를 설치 중...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git"])
    import clip
'''


# CLIP 모델 로드 (ViT-B/32)
print("CLIP 모델 로딩 중...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"모델 로드 완료 (Device: {device})")

BASE_DIR = "/Users/nanahyun/Documents/GitHub/final_develop/design/data"

# json 파일이 있는 폴더 
JSON_FOLDER = f"{BASE_DIR}/json/2024-2026"
# 이미지 저장할 폴더
DOWNLOAD_DIR = f"{BASE_DIR}/images/2024-2026"
# 스케치 임베딩 저장할 폴더
EMBEDDING_OUTPUT = f"{BASE_DIR}/embeddings_sketch/2024-2026"
# 원본 임베딩 저장할 폴더
EMBEDDING_OUTPUT_ORIGINAL = f"{BASE_DIR}/embeddings_original/2024-2026"
# 스케치 전처리 이미지 저장할 폴더
SKETCH_DIR = f"{BASE_DIR}/images_sketch/2024-2026"
# 에러 기록 파일
ERROR_LOG = f"{BASE_DIR}/error_log.txt"

# 디렉토리 생성
Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(EMBEDDING_OUTPUT).mkdir(parents=True, exist_ok=True)
Path(EMBEDDING_OUTPUT_ORIGINAL).mkdir(parents=True, exist_ok=True)
Path(SKETCH_DIR).mkdir(parents=True, exist_ok=True)

#폴더 내 모든 파일 리스트 가져오기
list = os.listdir(JSON_FOLDER) 

#for 문으로 폴더 내 모든 json 파일 처리
for idx, filename in enumerate(list,1):
    if not filename.endswith(".json"):
        continue

    time.sleep(3.0)  # 서버 과부하 방지 위해 3초 대기
    
    JSON_FILE = os.path.join(JSON_FOLDER, filename)
    print(f"\n{'='*50}")
    print(f"{idx}/{len(list)} 처리 중: {filename}")
    print(f"{'='*50}")


    # JSON 파일 읽기
    print(f"\nJSON 파일 읽기: {JSON_FILE}")
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 이미지 경로 추출
    image_path = data.get('image', {}).get('imagePath')
    image_name = data.get('image', {}).get('imageName', 'image.jpg')
    design_id = data.get('design_id', 'unknown')

    # 이미 처리된 파일이면 skip (이어받기)
    image_number_pre = data.get('image', {}).get('number', '1')
    output_file_pre = os.path.join(EMBEDDING_OUTPUT, f"{design_id}-{image_number_pre}_embedding.json")
    if os.path.exists(output_file_pre):
        print(f"SKIP: 이미 처리됨 → {output_file_pre}")
        continue

    if not image_path:
        msg = f"[{filename}] imagePath를 찾을 수 없습니다."
        print(f"ERROR: {msg}")
        with open(ERROR_LOG, 'a', encoding='utf-8') as ef:
            ef.write(msg + "\n")
        continue

    print(f"이미지 경로: {image_path}")
    print(f"이미지 이름: {image_name}")
    print(f"디자인 ID: {design_id}")

    # 이미지 다운로드
    print("\n이미지 다운로드 중...")
    try:
        # get 요청으로 이미지 다운로드
        response = requests.get(image_path, timeout=60) 
        response.raise_for_status()
        
        # 이미지 저장
        image_file_path = os.path.join(DOWNLOAD_DIR, f"{design_id}_{image_name}")
        with open(image_file_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ 이미지 저장 완료: {image_file_path}")
        
        # 이미지 로드
        image = Image.open(image_file_path).convert('RGB')
        print(f"✓ 이미지 크기: {image.size}")

        # 원본 이미지 CLIP 임베딩
        original_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            original_embedding = model.encode_image(original_tensor).cpu().numpy()

        # 스케치 전처리 (Canny Edge Detection)
        img_array = np.array(image.convert('L'))  # 그레이스케일 변환
        blurred = cv2.GaussianBlur(img_array, (5, 5), 0)  # 1. 노이즈 제거
        edges = cv2.Canny(blurred, 30, 100)                # 2~5. Gradient → NMS → Double Threshold → Edge Tracking
        edges = cv2.bitwise_not(edges)                     # 반전: 흰 바탕 + 검정 선
        image = Image.fromarray(edges).convert('RGB')      # CLIP 입력 형식으로 복원
        sketch_file_path = os.path.join(SKETCH_DIR, f"{design_id}_{image_name}")
        image.save(sketch_file_path)
        print(f"✓ 스케치 이미지 저장 완료: {sketch_file_path}")

        # CLIP 전처리 및 임베딩
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_embedding = model.encode_image(image_tensor)
        
        # 임베딩을 CPU로 이동 후 numpy로 변환
        embedding_array = image_embedding.cpu().numpy()
        
        print(f"✓ 임베딩 완료")
        print(f"  - 임베딩 크기: {embedding_array.shape}")
        print(f"  - 임베딩 차원: {embedding_array.shape[1]}")
        print(f"  - 첫 10개 값: {embedding_array[0, :10]}")
        
        # 메타데이터 추출
        application_number = data.get('applicationNumber', '')
        registration_number = data.get('registrationNumber', '')
        status = data.get('status', {})
        meta = data.get('meta', {})
        image = data.get('image', {})
        creative = data.get('creative', {})
        
        # 결과 저장 (JSON 형식)
        image_number = image.get('number', '1')
        id_field = f"{design_id}-IMG-{image_number}"
        
        output_file = os.path.join(EMBEDDING_OUTPUT, f"{design_id}-{image_number}_embedding.json")
        result = {
            "id": id_field,
            "embedding": embedding_array.tolist()[0],  # 첫 번째 배치 항목
            "metadata": {
                "design_id": design_id, #디자인id
                "applicationNumber": application_number, #출원번호
                "registrationNumber": registration_number, #등록번호
                "regFg": status.get('regFg', ''), #등록여부
                "admstStat": status.get('admstStat', ''), #행정상태
                "lastDispositionDate": status.get('lastDispositionDate', ''), #최종처분일
                "articleName": meta.get('articleName', ''), #상품명
                "LCCode": meta.get('LCCode', ''), #LCCode
                "image_id": image.get('image_id', ''), #이미지id
                "imagePath": image_path, #이미지경로
                "imageNumber": image_number, #도면번호
                "designSummary": creative.get('designSummary', ''), #디자인 요약
                "applicantName": meta.get('applicantName', '') #출원인명
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 스케치 임베딩 저장 완료: {output_file}")

        # 원본 임베딩 저장
        output_file_original = os.path.join(EMBEDDING_OUTPUT_ORIGINAL, f"{design_id}-{image_number}_embedding.json")
        result_original = {
            "id": id_field,
            "embedding": original_embedding.tolist()[0],
            "metadata": result["metadata"]
        }
        with open(output_file_original, 'w', encoding='utf-8') as f:
            json.dump(result_original, f, indent=2, ensure_ascii=False)

        print(f"✓ 원본 임베딩 저장 완료: {output_file_original}")
        
        
    except requests.exceptions.RequestException as e:
        msg = f"[{filename}] 이미지 다운로드 실패 - {e}"
        print(f"ERROR: {msg}")
        with open(ERROR_LOG, 'a', encoding='utf-8') as ef:
            ef.write(msg + "\n")
        continue
    except Exception as e:
        msg = f"[{filename}] {type(e).__name__}: {e}"
        print(f"ERROR: {msg}")
        with open(ERROR_LOG, 'a', encoding='utf-8') as ef:
            ef.write(msg + "\n")
        continue

    print("작업 완료!")
    print(f"다운로드된 이미지: {DOWNLOAD_DIR}")
    print(f"임베딩 저장 위치: {EMBEDDING_OUTPUT}")