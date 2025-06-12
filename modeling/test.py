import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
import optuna

root = Path(__file__).resolve().parents[2]

df = pd.read_csv(root/'data/raw/merged_macro_dataset.csv', parse_dates=['date'])
df_sent = pd.read_csv(root/'data/raw/daily_kfdeberta_sentiment.csv', parse_dates=['date'])

    # 종속 데이터 유지하며 left merge
df = df.merge(df_sent, on='date', how='left')

for col in ['kospi', 'usd_krw', 'eur_krw', 'cny_krw', 'jpy_krw',
                'usd_jpy', 'gold_krw', 'wti_krw', 'kosdaq']:
        if col in df.columns:
            # 따옴표 제거, 쉼표 제거, 숫자 변환
            df[col] = (df[col]
                       .astype(str)
                       .str.replace('"', '', regex=False)
                       .str.replace(',', '', regex=False)
                       .astype(float))
        else:
            raise KeyError(f"Column '{col}' not found in CSV")

df[['sent_pos','sent_neu','sent_neg']] = df[['sent_pos','sent_neu','sent_neg']].fillna(0)

df['ret'] = df['kospi'].pct_change()
df['ret_lag1'] = df['ret'].shift(1)

print(df.dtypes)

print(df.head(20))


