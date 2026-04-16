"""
Ground Truth CSV 생성 스크립트
- 같은 출원인 & 같은 등록일자, 다른 출원번호 쌍 추출
- 그룹(출원인+날짜) 내 특허 3개 이상 → query 1개 + relevant 2개
"""
import json
import os
import csv
import random
from collections import defaultdict

# 경로 설정
BASE_DIR = r"/Users/nanahyun/Documents/GitHub/final_develop/design/data"
JSON_DIR = f"{BASE_DIR}/json/통합"
IMAGE_DIR = f"{BASE_DIR}/images/통합"
OUTPUT_DIR = f"{BASE_DIR}/test_data/huristic"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "huristic_testdata.csv")

# ── 쌍 개수 설정 (0 = 전체) ──
MAX_QUERIES = 50

# 이미지 파일 목록을 set으로 로드 (빠른 존재 확인)
image_files = set(os.listdir(IMAGE_DIR))

# JSON 파일 파싱
app_drawings = defaultdict(list)  # appNum -> [{drawingNum, imageFile, exists}]
app_meta = {}  # appNum -> {applicantName, LCCode}

for fname in sorted(os.listdir(JSON_DIR)):
    if not fname.endswith(".json"):
        continue
    fpath = os.path.join(JSON_DIR, fname)
    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)

    app_num = data["applicationNumber"]
    drawing_num = data["image"]["number"]
    lc_code = data["meta"]["LCCode"]
    applicant = data["meta"]["applicantName"].strip()
    last_date = data["status"]["lastDispositionDate"]
    image_name = data["image"]["imageName"]

    idx = os.path.splitext(image_name)[0]
    image_file = f"{app_num}-{lc_code}-{drawing_num}_{idx}.jpg"

    app_drawings[app_num].append({
        "drawingNum": drawing_num,
        "imageFile": image_file,
        "exists": image_file in image_files,
    })

    if app_num not in app_meta:
        app_meta[app_num] = {
            "applicantName": applicant,
            "lastDispositionDate": last_date,
            "LCCode": lc_code,
        }

# ── 같은 출원인 & 같은 등록일자, 다른 출원번호 쌍 생성 ──
# 출원번호별 도면0 이미지 수집
app_rep = {}  # appNum -> imageFile (도면0)
for app_num, drawings in app_drawings.items():
    by_num = {d["drawingNum"]: d for d in drawings}
    rep = by_num.get("0")
    if rep and rep["exists"]:
        app_rep[app_num] = rep["imageFile"]

# (출원인 + 등록일자)별 출원번호 그룹핑
group_apps = defaultdict(list)
for app_num, meta in app_meta.items():
    if app_num in app_rep:
        key = (meta["applicantName"], meta["lastDispositionDate"])
        group_apps[key].append({
            "appNum": app_num,
            "imageFile": app_rep[app_num],
        })

# 그룹 내 특허 3개 이상 → query 1개 + relevant 2개
candidates = []
for (applicant, date), apps in group_apps.items():
    if len(apps) < 3:
        continue
    for i, query_app in enumerate(apps):
        others = apps[:i] + apps[i+1:]
        rel_pair = others[:2]
        candidates.append({
            "query_image":     query_app["imageFile"],
            "relevant_images": [r["imageFile"] for r in rel_pair],
            "applicant":       applicant,
            "date":            date,
        })

random.seed(18)
if MAX_QUERIES > 0 and len(candidates) > MAX_QUERIES:
    candidates = random.sample(candidates, MAX_QUERIES)

rows = []
for item in candidates:
    for rel_img in item["relevant_images"]:
        rows.append({
            "query_image":     item["query_image"],
            "relevant_image":  rel_img,
            "similarity_type": "hard_positive",
            "note": f"같은 출원인({item['applicant']}) 같은 등록일({item['date']}) 다른 출원번호",
        })

# ── CSV 저장 ──
fieldnames = ["query_image", "relevant_image", "similarity_type", "note"]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"같은 출원인 & 같은 등록일 다른 출원번호: {len(candidates)}개 query, 총 {len(rows)}행 → {OUTPUT_CSV}")
