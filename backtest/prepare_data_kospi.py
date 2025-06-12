import numpy as np
import pandas as pd
from pathlib import Path

def label_binary(returns):
    """
    수익률 returns(배열) > 0 → 1 (상승), <= 0 → 0 (하락/횡보)
    """
    return (returns > 0).astype(int)

def prepare_data_kospi(seq_len, feature_cols, kospi_csv="data/raw/kospi_index_daily.csv"):
    """
    :param seq_len:       과거 몇 개의 일봉을 한 샘플로 묶을지 지정
    :param feature_cols:  ['open','high','low','close','change_pct','volume'] 등
    :param kospi_csv:     코스피 지수 일봉 CSV 경로
    :return: X (np.ndarray, shape=(n_samples, seq_len, n_features)), 
              y (np.ndarray, shape=(n_samples,), binary label)
    """
    df = pd.read_csv(kospi_csv, parse_dates=['date'])
    df.sort_values("date", inplace=True)

    # (A) 전일 대비 수익률 계산 → 이진 레이블
    df["ret"] = df["close"].pct_change()
    df["target"] = label_binary(df["ret"])
    df.dropna(inplace=True)  # 첫 행은 ret 계산 불가 → 제거

    # (B) Sliding window 시퀀스 X, 레이블 y
    X, y = [], []
    vals = df[feature_cols].values  # shape = (n_rows, n_features)
    labs = df["target"].values      # shape = (n_rows,)

    for i in range(len(df) - seq_len):
        X.append(vals[i:i + seq_len])
        y.append(labs[i + seq_len])  # seq_len 이후의 레이블

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y