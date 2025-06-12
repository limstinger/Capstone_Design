import os
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    pipeline
)

# ---------- 설정 ----------
device   = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1) 파인튜닝된 금융 NER 모델 디렉토리
NER_OUT_DIR = os.path.join(BASE_DIR, "outputs", "kf-deberta-fin-ner")

# 2) 파인튜닝된 FinBERT 감성 모델 디렉토리
SENTI_OUT_DIR = os.path.join(BASE_DIR, "finance_project", "finetuned")

# 3) 뉴스 데이터 CSV 경로
CSV_PATH = os.path.join(BASE_DIR, "models", "data", "articles_export.csv")

# ---------- 1) NER 파이프라인 로드 ----------
tokenizer_ner = AutoTokenizer.from_pretrained(NER_OUT_DIR, trust_remote_code=True)
model_ner     = AutoModelForTokenClassification.from_pretrained(
    NER_OUT_DIR,
    trust_remote_code=True
).to(device)

ner_pipeline = pipeline(
    "ner",
    model=model_ner,
    tokenizer=tokenizer_ner,
    aggregation_strategy="simple",  # 토큰 단위 → 개체 단위 병합
    device=0 if device=="cuda" else -1
)

# ---------- 2) 감성 분석 파이프라인 로드 ----------
tokenizer_senti = AutoTokenizer.from_pretrained(SENTI_OUT_DIR)
model_senti     = AutoModelForSequenceClassification.from_pretrained(
    SENTI_OUT_DIR
).to(device)

senti_pipeline = pipeline(
    "sentiment-analysis",
    model=model_senti,
    tokenizer=tokenizer_senti,
    device=0 if device=="cuda" else -1
)

# ---------- 3) 뉴스 데이터 로드 및 날짜 보정 ----------
df = pd.read_csv(CSV_PATH)
df['datetime'] = pd.to_datetime(
    df['pubDate'],
    format="%a, %d %b %Y %H:%M:%S %z",
    errors="coerce"
)
# 기준일(자정)
df['date'] = df['datetime'].dt.normalize()
# 15:30 이후 발표된 기사는 다음날로 할당
cutoff = pd.to_datetime("15:30").time()
mask   = df['datetime'].dt.time >= cutoff
df.loc[mask, 'date'] += pd.Timedelta(days=1)
df.drop(columns=['datetime'], inplace=True)

# ---------- 4) 엔티티 추출 + 감성 점수 매기기 ----------
records = []
for _, row in df.iterrows():
    headline = row['headline']
    ents = ner_pipeline(headline)
    # ents 예시: [{"word":"삼성전자","entity_group":"ORG", ...}, ...]
    for ent in ents:
        grp = ent['entity_group']
        if grp in ("ORG","PER","LOC","MISC"):
            # headline 전체 문장 감성
            out = senti_pipeline(headline)[0]
            # POSITIVE → +score, NEGATIVE → –score, 중립은 score label에 따라 다르게 처리
            score = out['score'] if out['label']=="POSITIVE" else -out['score']
            records.append({
                "date": row['date'],
                "entity": ent['word'],
                "sentiment_score": score
            })

# ---------- 5) 일별·엔티티별 집계 피처 생성 ----------
feat_entity = (
    pd.DataFrame(records)
      .groupby(["date","entity"])
      .agg(
         mean_sent=("sentiment_score","mean"),
         std_sent =("sentiment_score","std"),
         count    =("sentiment_score","count")
      )
      .reset_index()
)

# 결과 출력(또는 CSV로 저장 등 후처리)
print(feat_entity.head())
feat_entity.to_csv("entity_sentiment_features.csv", index=False)