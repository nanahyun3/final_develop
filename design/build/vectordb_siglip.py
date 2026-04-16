import os
import json
import chromadb

BASE_DIR = "/Users/nanahyun/Documents/GitHub/final_develop/design/data"

# 원본 임베딩 폴더
EMBEDDING_DIR_ORIGINAL = f"{BASE_DIR}/embeddings_original_siglip/통합"
# 스케치 임베딩 폴더
EMBEDDING_DIR_SKETCH = f"{BASE_DIR}/embeddings_sketch_siglip/통합"

# ChromaDB 클라이언트 초기화
chroma_client = chromadb.PersistentClient(path=f"{BASE_DIR}/../chroma_db_siglip")  # 새로운 DB 경로 설정

# 컬렉션 2개 생성 (이미 존재하면 불러옴)
original_collection = chroma_client.get_or_create_collection(
    name="design_original",
    metadata={"hnsw:space": "cosine"}
)
sketch_collection = chroma_client.get_or_create_collection(
    name="design_sketch",
    metadata={"hnsw:space": "cosine"}
)
print("컬렉션 생성 완료: design_original, design_sketch")


def build_metadata(data):
    m = data.get("metadata", {})
    return {
        "design_id": m.get("design_id", ""),              #디자인id
        "applicationNumber": m.get("applicationNumber", ""), #출원번호
        "registrationNumber": m.get("registrationNumber", ""), #등록번호
        "regFg": m.get("regFg", ""),                      #등록여부
        "admstStat": m.get("admstStat", ""),              #행정상태
        "lastDispositionDate": m.get("lastDispositionDate", ""), #최종처분일
        "articleName": m.get("articleName", ""),          #상품명
        "LCCode": m.get("LCCode", ""),                    #LCCode
        "image_id": m.get("image_id", ""),                #이미지id
        "imagePath": m.get("imagePath", ""),              #이미지경로
        "imageNumber": m.get("imageNumber", ""),          #도면번호
        "designSummary": m.get("designSummary", ""),      #디자인 요약
        "applicantName": m.get("applicantName", "")       #출원인명
    }


def load_to_collection(collection, embedding_dir):
    count = 0
    for filename in os.listdir(embedding_dir):
        if not filename.endswith("_embedding.json"):
            continue

        filepath = os.path.join(embedding_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        collection.add(
            ids=[data.get("id")],
            embeddings=[data.get("embedding")],
            metadatas=[build_metadata(data)]
        )
        count += 1

    print(f"✅ {collection.name}: {count}개 저장 완료")


print("\n원본 임베딩 저장 중...")
load_to_collection(original_collection, EMBEDDING_DIR_ORIGINAL)

print("\n스케치 임베딩 저장 중...")
load_to_collection(sketch_collection, EMBEDDING_DIR_SKETCH)

print("\n✅ 모든 embedding json을 벡터 DB에 저장 완료")
