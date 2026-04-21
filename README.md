# 디자인 특허 유사도 분석 시스템

디자인 특허 이미지를 입력하면 유사 디자인을 검색하고 FTO(Freedom to Operate) 리포트를 생성하는 챗봇 시스템

## 주요 기능

- **유사 디자인 검색** — CLIP / DINOv2 / SigLIP 임베딩 + ChromaDB 벡터 검색
- **Hybrid Retrieval** — Dense(이미지) + BM25(텍스트) 결합 재랭킹
- **FTO 리포트 생성** — GPT-4o VLM 기반 유사점·차이점 분석
- **챗봇 인터페이스** — LangGraph 기반 멀티턴 대화

## 구조

```
design/
├── build/       # 임베딩 생성 및 벡터 DB 구축
├── src/         # 챗봇 서버 및 검색 로직
└── 성능 테스트/ # 모델 성능 평가 (Recall@K, MRR)
```

## 실행

```bash
pip install -r requirements.txt
uvicorn design/src/api:app --reload
```

## 기술 스택

| 역할 | 사용 기술 |
|---|---|
| 임베딩 | CLIP ViT-B/32 · DINOv2-L · SigLIP-SO400M |
| 벡터 DB | ChromaDB |
| LLM / VLM | GPT-4o |
| 챗봇 프레임워크 | LangGraph |
| 서버 | FastAPI |
