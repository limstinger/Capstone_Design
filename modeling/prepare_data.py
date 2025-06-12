import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import ta

def label_binary(returns):
    return (returns > 0).astype(int)

root = Path(__file__).resolve().parents[2]

def prepare_data(seq_len, feature_cols):
    df = pd.read_csv(root/'data/raw/merged_macro_dataset.csv', parse_dates=['date'])
    df_sent = pd.read_csv(root/'data/raw/daily_kfdeberta_sentiment.csv', parse_dates=['date'])

    df = df.merge(df_sent, on='date', how='left')
    for col in ['kospi', 'usd_krw', 'eur_krw', 'cny_krw', 'jpy_krw',
                'usd_jpy', 'gold_krw', 'wti_krw', 'kosdaq']:
        if col in df.columns:
            df[col] = (df[col]
                       .astype(str)
                       .str.replace('"','', regex=False)
                       .str.replace(',','', regex=False)
                       .astype(float))
        else:
            raise KeyError(f"Column '{col}' not found")

    df[['sent_pos','sent_neu','sent_neg']] = df[['sent_pos','sent_neu','sent_neg']].fillna(0)
    df['ret'] = df['kospi'].pct_change()
    df['ret_lag1'] = df['ret'].shift(1)
    df['target'] = label_binary(df['ret'])

    df['rsi_14'] = ta.momentum.RSIIndicator(close=df['kospi'], window=14).rsi()
    df['macd'] = ta.trend.MACD(close=df['kospi'], window_slow=26, window_fast=12).macd()
    df['boll_upper'] = ta.volatility.BollingerBands(close=df['kospi']).bollinger_hband()
    df['boll_lower'] = ta.volatility.BollingerBands(close=df['kospi']).bollinger_lband()
    df['volatility_std10'] = df['kospi'].rolling(10).std()

    df.dropna(inplace=True)

    if len(df) < seq_len + 1:
        raise ValueError("데이터가 너무 작아서 시퀀스를 구성할 수 없습니다.")

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    X, y = [], []
    vals = df[feature_cols].values
    labs = df['target'].values
    for i in range(len(df) - seq_len):
        X.append(vals[i:i+seq_len])
        y.append(labs[i+seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)