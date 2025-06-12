import pandas as pd
import torch
import scipy.special
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # scripts가 data/processed 아래라면 부모 한 단계
RAW_DIR       = PROJECT_ROOT / "raw"

RAW_FILE      = RAW_DIR / "articles_export.csv"
OUT_FILE      = RAW_DIR / "daily_kfdeberta_sentiment.csv"

# 2. 원본 뉴스 파일 로드
df = pd.read_csv(RAW_FILE, parse_dates=['adjustedDate'])
df = df.dropna(subset=['headline'])
df.rename(columns={'adjustedDate':'date'}, inplace=True)

# 2. KF-DeBERTa 모델 로드
MODEL = "kakaobank/kf-deberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# 3. 감성 점수 추출 함수
def get_sentiment_kf(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = scipy.special.softmax(logits.cpu().numpy().squeeze())
    return probs  # [neg, neu, pos]

# 4. 감성 계산 및 데이터 합치기
probs = df['headline'].apply(get_sentiment_kf)
df_probs = pd.DataFrame(probs.tolist(), columns=['neg', 'neu', 'pos'])
df = pd.concat([df.reset_index(drop=True), df_probs], axis=1)

# 5. 날짜별 감성 집계
daily = df.groupby(df['date'].dt.date)[['pos','neu','neg']].mean().reset_index()
daily.columns = ['date', 'sent_pos', 'sent_neu', 'sent_neg']

# 6. 결과 저장
daily.to_csv("daily_kfdeberta_sentiment.csv", index=False)
print("✅ 일별 KF‑DeBERTa 감성 벡터가 'daily_kfdeberta_sentiment.csv'에 저장되었습니다.")