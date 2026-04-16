import json
import os
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel

'''
embeddings.py에서 이미 저장된 이미지를 불러와 SigLIP-SO400M으로 임베딩한다.

- 원본 이미지  → "data/images"         → "data/embeddings_original_siglip"
- 스케치 이미지 → "data/images_sketch"  → "data/embeddings_sketch_siglip"
'''

# SigLIP 모델 로드
print("SigLIP 모델 로딩 중...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(device)
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
model.eval()
print(f"모델 로드 완료 (Device: {device})")

BASE_DIR = "/Users/nanahyun/Documents/GitHub/final_develop/design/data"

# json 파일이 있는 폴더
# data/json/1981-2011 폴더로만 우선 처리
JSON_FOLDER = f"{BASE_DIR}/json/2024-2026"
# 원본 이미지 폴더 (embeddings.py에서 저장)
DOWNLOAD_DIR = f"{BASE_DIR}/images/2024-2026"
# 스케치 이미지 폴더 (embeddings.py에서 저장)
SKETCH_DIR = f"{BASE_DIR}/images_sketch/2024-2026"
# 스케치 임베딩 저장할 폴더
EMBEDDING_OUTPUT = f"{BASE_DIR}/embeddings_sketch_siglip/2024-2026"
# 원본 임베딩 저장할 폴더
EMBEDDING_OUTPUT_ORIGINAL = f"{BASE_DIR}/embeddings_original_siglip/2024-2026"
# 에러 기록 파일
ERROR_LOG = f"{BASE_DIR}/error_log.txt"

# 디렉토리 생성
Path(EMBEDDING_OUTPUT).mkdir(parents=True, exist_ok=True)
Path(EMBEDDING_OUTPUT_ORIGINAL).mkdir(parents=True, exist_ok=True)


def encode_image(pil_image):
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.get_image_features(**inputs)
    # get_image_features may return a tensor or a ModelOutput object
    if hasattr(output, 'pooler_output') and output.pooler_output is not None:
        embedding = output.pooler_output
    elif hasattr(output, 'last_hidden_state'):
        embedding = output.last_hidden_state[:, 0, :]
    else:
        embedding = output
    return embedding.cpu().numpy()


# 폴더 내 모든 파일 리스트 가져오기
list = os.listdir(JSON_FOLDER)

# for 문으로 폴더 내 모든 json 파일 처리
for idx, filename in enumerate(list, 1):
    if not filename.endswith(".json"):
        continue

    JSON_FILE = os.path.join(JSON_FOLDER, filename)
    print(f"\n{'='*50}")
    print(f"{idx}/{len(list)} 처리 중: {filename}")
    print(f"{'='*50}")

    # JSON 파일 읽기
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    image_name = data.get('image', {}).get('imageName', 'image.jpg')
    image_path = data.get('image', {}).get('imagePath')
    design_id = data.get('design_id', 'unknown')

    # 이미 처리된 파일이면 skip (이어받기)
    image_number_pre = data.get('image', {}).get('number', '1')
    output_file_pre = os.path.join(EMBEDDING_OUTPUT, f"{design_id}-{image_number_pre}_embedding.json")
    output_file_orig_pre = os.path.join(EMBEDDING_OUTPUT_ORIGINAL, f"{design_id}-{image_number_pre}_embedding.json")
    if os.path.exists(output_file_pre) and os.path.exists(output_file_orig_pre):
        print(f"SKIP: 이미 처리됨 → {output_file_pre}")
        continue

    print(f"디자인 ID: {design_id}")

    try:
        # 원본 이미지 로드
        image_file_path = os.path.join(DOWNLOAD_DIR, f"{design_id}_{image_name}")
        if not os.path.exists(image_file_path):
            msg = f"[{filename}] 원본 이미지 없음 → {image_file_path}"
            print(f"ERROR: {msg}")
            with open(ERROR_LOG, 'a', encoding='utf-8') as ef:
                ef.write(msg + "\n")
            continue

        image = Image.open(image_file_path).convert('RGB')
        print(f"✓ 원본 이미지 로드: {image_file_path}")

        # 스케치 이미지 로드
        sketch_file_path = os.path.join(SKETCH_DIR, f"{design_id}_{image_name}")
        if not os.path.exists(sketch_file_path):
            msg = f"[{filename}] 스케치 이미지 없음 → {sketch_file_path}"
            print(f"ERROR: {msg}")
            with open(ERROR_LOG, 'a', encoding='utf-8') as ef:
                ef.write(msg + "\n")
            continue

        sketch_image = Image.open(sketch_file_path).convert('RGB')
        print(f"✓ 스케치 이미지 로드: {sketch_file_path}")

        # SigLIP 임베딩
        original_embedding = encode_image(image)
        sketch_embedding = encode_image(sketch_image)
        print(f"✓ 임베딩 완료 (차원: {sketch_embedding.shape[1]})")

        # 메타데이터 추출
        application_number = data.get('applicationNumber', '')
        registration_number = data.get('registrationNumber', '')
        status = data.get('status', {})
        meta = data.get('meta', {})
        image_meta = data.get('image', {})
        creative = data.get('creative', {})

        image_number = image_meta.get('number', '1')
        id_field = f"{design_id}-IMG-{image_number}"

        metadata = {
            "design_id": design_id,                                       #디자인id
            "applicationNumber": application_number,                      #출원번호
            "registrationNumber": registration_number,                    #등록번호
            "regFg": status.get('regFg', ''),                            #등록여부
            "admstStat": status.get('admstStat', ''),                    #행정상태
            "lastDispositionDate": status.get('lastDispositionDate', ''), #최종처분일
            "articleName": meta.get('articleName', ''),                  #상품명
            "LCCode": meta.get('LCCode', ''),                            #LCCode
            "image_id": image_meta.get('image_id', ''),                  #이미지id
            "imagePath": image_path,                                      #이미지경로
            "imageNumber": image_number,                                  #도면번호
            "designSummary": creative.get('designSummary', ''),          #디자인 요약
            "applicantName": meta.get('applicantName', '')               #출원인명
        }

        # 스케치 임베딩 저장
        output_file = os.path.join(EMBEDDING_OUTPUT, f"{design_id}-{image_number}_embedding.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"id": id_field, "embedding": sketch_embedding.tolist()[0], "metadata": metadata},
                      f, indent=2, ensure_ascii=False)
        print(f"✓ 스케치 임베딩 저장 완료: {output_file}")

        # 원본 임베딩 저장
        output_file_original = os.path.join(EMBEDDING_OUTPUT_ORIGINAL, f"{design_id}-{image_number}_embedding.json")
        with open(output_file_original, 'w', encoding='utf-8') as f:
            json.dump({"id": id_field, "embedding": original_embedding.tolist()[0], "metadata": metadata},
                      f, indent=2, ensure_ascii=False)
        print(f"✓ 원본 임베딩 저장 완료: {output_file_original}")

    except Exception as e:
        msg = f"[{filename}] {type(e).__name__}: {e}"
        print(f"ERROR: {msg}")
        with open(ERROR_LOG, 'a', encoding='utf-8') as ef:
            ef.write(msg + "\n")
        continue

    print("작업 완료!")
    print(f"임베딩 저장 위치: {EMBEDDING_OUTPUT}")
