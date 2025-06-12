import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 1. 경로 설정
root = Path(__file__).resolve().parent.parent
sent_path = root / "data" / "raw" / "daily_kfdeberta_sentiment.csv"
kospi_path = root / "data" / "raw" / "KOSPI Historical Data.csv"

# 2. 데이터 불러오기
df_sent = pd.read_csv(sent_path, parse_dates=["date"])
df_kospi = pd.read_csv(kospi_path)
df_kospi["date"] = pd.to_datetime(df_kospi["Date"])
df_kospi["kospi_change"] = df_kospi["Change %"].str.replace("%", "").astype(float) / 100
df_kospi = df_kospi[["date", "kospi_change"]]

# 3. 날짜 기준 병합
df = pd.merge(df_kospi, df_sent, on="date", how="inner").sort_values("date")

# 4. 상관관계 분석
corr = df[["kospi_change", "sent_pos", "sent_neu", "sent_neg"]].corr()

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("KOSPI Return vs News Sentiment (Correlation)")
plt.tight_layout()
plt.show()

# 5. 산점도 분석
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(["sent_pos", "sent_neu", "sent_neg"]):
    sns.scatterplot(data=df, x=col, y="kospi_change", ax=axes[i])
    axes[i].set_title(f"KOSPI Return vs {col}")
plt.tight_layout()
plt.show()