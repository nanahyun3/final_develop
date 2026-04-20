"""
평가 전용 sketch ChromaDB 구성
- 기존 DB 복사(sketch 컬렉션) 
- "원본" answer 이미지들을 추가한 평가DB 구축
"""


import cv2
import clip
import torch
import numpy as np
import chromadb
from PIL import Image
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel, AutoProcessor


# ── 설정 ──────────────────────────────────────────
# 기존 DB 경로
DB_CLIP   = "/Users/nanahyun/Documents/GitHub/final_develop/design/chroma_db"
DB_DINOV2 = "/Users/nanahyun/Documents/GitHub/final_develop/design/chroma_db_dinov2"
DB_SIGLIP = "/Users/nanahyun/Documents/GitHub/final_develop/design/chroma_db_siglip"

# 기존 DB 컬렉션명
EXISTING_COLLECTION_SKETCH = "design_sketch"

# 평가용DB 경로
EVALDB_PATH = "/Users/nanahyun/Documents/GitHub/final_develop/design/성능 테스트/dataset/chroma_eval_db"

# 평가용DB 컬렉션명 (모델별 별도 컬렉션)
EVAL_COLLECTION_MAP = {
    DB_CLIP:   "design_sketch_eval_clip",
    DB_DINOV2: "design_sketch_eval_dinov2",
    DB_SIGLIP: "design_sketch_eval_siglip",
}

# 정답 이미지 폴더
IMAGE_DIR = "/Users/nanahyun/Documents/GitHub/final_develop/design/성능 테스트/dataset/answer"
# ─────────────────────────────────────────────────


device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
print(f"CLIP 로드 완료 (Device: {device})")

# DINOv2-Large
dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
dinov2_model     = AutoModel.from_pretrained("facebook/dinov2-large").to(device)
dinov2_model.eval()
print("DINOv2-Large 로드 완료")

# SigLIP-SO400M
siglip_processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
siglip_model     = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(device)
siglip_model.eval()
print("SigLIP-SO400M 로드 완료")


def to_sketch(pil_image):
    img_array = np.array(pil_image.convert('L'))
    blurred   = cv2.GaussianBlur(img_array, (5, 5), 0)
    edges     = cv2.Canny(blurred, 30, 100)
    edges     = cv2.bitwise_not(edges)
    return Image.fromarray(edges).convert('RGB')

def _normalize(vec):
    '''L2 정규화 함수'''
    arr  = np.array(vec, dtype=np.float64)
    norm = np.linalg.norm(arr)
    return (arr / norm).tolist() if norm > 0 else vec

def embed_clip(pil_image):
    '''CLIP 임베딩 생성'''
    tensor = clip_preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = clip_model.encode_image(tensor).cpu().numpy()
    return _normalize(vec[0].tolist())

def embed_dinov2(pil_image):
    '''DINOv2 임베딩 생성'''
    inputs = dinov2_processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dinov2_model(**inputs)
    return _normalize(outputs.last_hidden_state[:, 0, :].cpu().numpy()[0].tolist())

def embed_siglip(pil_image):
    '''SigLIP 임베딩 생성'''
    inputs = siglip_processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = siglip_model.get_image_features(**inputs)
    if hasattr(output, 'pooler_output') and output.pooler_output is not None:
        vec = output.pooler_output
    elif hasattr(output, 'last_hidden_state'):
        vec = output.last_hidden_state[:, 0, :]
    else:
        vec = output
    return _normalize(vec.cpu().numpy()[0].tolist())


# 벡터 DB 3개를 순회하면서 하나씩 복사.
for existing_db_path, existing_collection_name in [
    (DB_CLIP,   EXISTING_COLLECTION_SKETCH),
    (DB_DINOV2, EXISTING_COLLECTION_SKETCH),
    (DB_SIGLIP, EXISTING_COLLECTION_SKETCH),
]:
    existing_client     = chromadb.PersistentClient(path=existing_db_path) # 기존 DB 클라이언트 연결
    existing_collection = existing_client.get_collection(existing_collection_name)  # 기존 컬렉션
    total               = existing_collection.count()                                # 기존 컬렉션 문서 수

    print(f"\n[복사 시작] {existing_db_path} - {existing_collection_name} ({total}건)")

    eval_client     = chromadb.PersistentClient(path=EVALDB_PATH) # 평가 DB 클라이언트 연결
    eval_collection = eval_client.get_or_create_collection(
        name=EVAL_COLLECTION_MAP[existing_db_path],
        metadata={"hnsw:space": "cosine"}
    ) # 평가 DB 컬렉션 생성/연결


    # 배치 단위로 기존DB에서 문서를 읽어와 평가DB에 추가
    batch_size = 500
    offset     = 0 # offset : 복사 진행 상황 추적

    while offset < total:

        chunk = existing_collection.get(
            include=["embeddings", "metadatas"],
            limit=batch_size,
            offset=offset
        ) # 기존DB에서 배치 단위로 문서 조회

        if not chunk["ids"]:
            break # 더 이상 문서가 없으면 종료

        # 평가DB에 문서 추가
        eval_collection.add(
            ids=chunk["ids"],
            embeddings=chunk["embeddings"],
            metadatas=chunk["metadatas"],
        )

        offset += batch_size # 오프셋 업데이트

        print(f"  복사 진행: {min(offset, total)}/{total}건")


    # 복사 완료된 DB에 정답 이미지 추가
    
    image_paths = [
        p for p in Path(IMAGE_DIR).iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    if not image_paths:
        print(f"  [경고] 정답 이미지 없음: {IMAGE_DIR} — 스킵")
        continue

    # 정답 이미지마다 모델별 임베딩 생성 후 평가DB에 추가
    for img_path in image_paths:
        pil_image = to_sketch(Image.open(img_path).convert("RGB"))

        if existing_db_path == DB_CLIP:
            # CLIP으로 정답 이미지 임베딩 생성 및 저장
            clip_vec = embed_clip(pil_image)
            eval_collection.add(
                ids=[f"answer_clip_{img_path.stem}"],
                embeddings=[clip_vec],
                metadatas=[{"source": "answer_image", "filename": img_path.name}]
            )

        elif existing_db_path == DB_DINOV2:
            # DINOv2로 정답 이미지 임베딩 생성 및 저장
            dinov2_vec = embed_dinov2(pil_image)
            eval_collection.add(
                ids=[f"answer_dinov2_{img_path.stem}"],
                embeddings=[dinov2_vec],
                metadatas=[{"source": "answer_image", "filename": img_path.name}]
            )

        elif existing_db_path == DB_SIGLIP:
            # SigLIP로 정답 이미지 임베딩 생성 및 저장
            siglip_vec = embed_siglip(pil_image)
            eval_collection.add(
                ids=[f"answer_siglip_{img_path.stem}"],
                embeddings=[siglip_vec],
                metadatas=[{"source": "answer_image", "filename": img_path.name}]
            )
        print(f"    추가 완료: {img_path.name}")
        
    print(f"  복사 완료: {eval_collection.name} ({eval_collection.count()}건)")
    

