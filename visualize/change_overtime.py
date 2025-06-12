from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 경로 설정
root = Path(__file__).resolve().parent.parent
sent_path = root / "data" / "raw" / "daily_kfdeberta_sentiment.csv"
kospi_path = root / "data" / "raw" / "KOSPI Historical Data.csv"

# 2. 데이터 불러오기
df_sent = pd.read_csv(sent_path)
df_kospi = pd.read_csv(kospi_path)

# 3. 컬럼 소문자화 및 날짜 파싱
df_sent.columns = df_sent.columns.str.lower()
df_kospi.columns = df_kospi.columns.str.lower()
df_sent['date'] = pd.to_datetime(df_sent['date'])
df_kospi['date'] = pd.to_datetime(df_kospi['date'])

# 4. 'change %' → float 변환 (%와 , 제거)
df_kospi['change'] = df_kospi['change %'].astype(str).str.replace('%', '').str.replace(',', '').astype(float)

# 5. 병합
df_merge = pd.merge(
    df_kospi[['date', 'change']],
    df_sent[['date', 'sent_pos', 'sent_neu', 'sent_neg']],
    on='date',
    how='inner'
)
df_merge.sort_values('date', inplace=True)

# 6. 상승일 여부 및 감성 일치 여부 계산
df_merge['up'] = df_merge['change'] > 0.0
df_merge['dominant_sent'] = df_merge[['sent_pos', 'sent_neu', 'sent_neg']].idxmax(axis=1)
df_merge['match'] = (
    ((df_merge['up']) & (df_merge['dominant_sent'] == 'sent_pos')) |
    ((~df_merge['up']) & (df_merge['dominant_sent'] == 'sent_neg'))
)

# 7. 일치율 계산 (NaN 방지)
up_days = df_merge[df_merge['up']]
down_days = df_merge[~df_merge['up']]
up_match = up_days['match'].mean() if len(up_days) > 0 else 0
down_match = down_days['match'].mean() if len(down_days) > 0 else 0
total_match = df_merge['match'].mean()

# 8. 정규화
scaler = MinMaxScaler()
df_scaled = df_merge.copy()
df_scaled[['change', 'sent_pos', 'sent_neu', 'sent_neg']] = scaler.fit_transform(
    df_scaled[['change', 'sent_pos', 'sent_neu', 'sent_neg']]
)

# 9. 시각화
plt.figure(figsize=(15, 6))

# 상승/하락일 배경 색상 강조
for i in range(len(df_scaled) - 1):
    color = 'green' if df_merge.iloc[i]['up'] else 'red'
    plt.axvspan(df_scaled['date'].iloc[i], df_scaled['date'].iloc[i+1], color=color, alpha=0.03)

# 선 그래프
plt.plot(df_scaled['date'], df_scaled['change'], label='KOSPI Change', color='black', linewidth=1.5)
plt.plot(df_scaled['date'], df_scaled['sent_pos'], label='Positive Sentiment', color='green', alpha=0.6)
plt.plot(df_scaled['date'], df_scaled['sent_neu'], label='Neutral Sentiment', color='blue', alpha=0.6)
plt.plot(df_scaled['date'], df_scaled['sent_neg'], label='Negative Sentiment', color='red', alpha=0.6)

# 감성 일치한 날 표시
matched = df_scaled[df_merge['match']]
plt.scatter(matched['date'], matched['change'], color='gold', label='Matched Days', s=30, marker='o', alpha=0.8)

# 주석 텍스트로 일치율 표시
plt.text(df_scaled['date'].iloc[10], 1.05,
         f"📈 상승일 감성 일치율: {up_match * 100:.2f}%\n"
         f"📉 하락일 감성 일치율: {down_match * 100:.2f}%\n"
         f"📊 전체 일치율: {total_match * 100:.2f}%",
         fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 스타일 설정
plt.title("KOSPI Change vs News Sentiment Over Time (Normalized)", fontsize=15)
plt.xlabel("Date")
plt.ylabel("Normalized Value (0~1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()