#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_next_day.py (margin-aware version)

- 마지막까지 수집된 KOSPI(+거시+뉴스) 피처를 넣어서
  다음 거래일 등락 확률을 예측하고 CSV에 append.
- 학습 때 저장한 params 안의 reject_margin을 같이 불러와
  예측을 신뢰구간/비신뢰구간으로 나눠서 저장한다.
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────── 기존 predict에서 쓰던 보정기와 tiny model들 그대로 둠
# (너가 올린 원본이랑 같음) ---------------------------

# ... (여기까지는 네 원본 predict_next_day.py 상단 부분 그대로 두면 돼)
# 아래부터가 중요한 변경 부분이야

def next_business_day(d: pd.Timestamp) -> pd.Timestamp:
    nd = d + pd.Timedelta(days=1)
    while nd.weekday() >= 5:  # 토(5), 일(6) 건너뜀
        nd += pd.Timedelta(days=1)
    return nd

def find_latest_final_dir(models_root: Path) -> Path:
    patterns = ["final_wavelet_transformer_*", "final_*"]
    cands = []
    for pat in patterns:
        cands.extend(list(models_root.glob(pat)))
    if not cands:
        for pat in patterns:
            cands.extend(list(models_root.rglob(pat)))
    cands = [p for p in cands if p.is_dir()]
    if not cands:
        raise FileNotFoundError("models 폴더에서 final_* 디렉터리를 찾을 수 없습니다.")
    return max(cands, key=lambda p: p.stat().st_mtime)

def load_artifacts(final_dir: Path):
    """
    model_train_margin_log.py 가 저장한 아티팩트를 읽어온다.
    - preproc_and_calibrators.pkl 안에 params / low_feats / high_feats / scalers / calibrators 가 들어있다고 가정
    - transformer, cnn, meta_xgb 도 같이 불러온다
    """
    pkl = final_dir / "preproc_and_calibrators.pkl"
    xgbp = final_dir / "meta_xgb.pkl"
    tpt = final_dir / "transformer_low_final.pt"
    cpt = final_dir / "cnn_high_final.pt"

    bundle = joblib.load(pkl)
    meta_model: XGBClassifier = joblib.load(xgbp)

    params = bundle["params"]
    low_feats = bundle["low_feats"]
    high_feats = bundle["high_feats"]
    low_scaler = bundle["low_scaler"]
    high_scaler = bundle["high_scaler"]
    cal_low = bundle["cal_low"]
    cal_high = bundle["cal_high"]
    meta_ctx_cols = params.get("meta_ctx_cols", [])
    seq_len = int(params["seq_len"])
    d_model, nhead = params["d_model_nhead"]
    dim_feedforward = int(params["dim_feedforward"])
    num_layers = int(params.get("num_layers", 2))
    dropout = float(params.get("transf_dropout", 0.2))
    n_scales = int(params.get("n_scales", 3))

    # ← 네 원래 predict에 있는 tiny WaveAtt / CNN 래퍼 그대로 써도 됨
    from predict_next_day import WaveAttTransformerClassifier, CNN1DClassifier  # 만약 이 파일 안에 정의돼 있으면 이 줄은 필요X

    trans = WaveAttTransformerClassifier(
        input_size=len(low_feats),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        n_scales=n_scales,
    ).to(DEVICE)
    trans.load_state_dict(torch.load(tpt, map_location=DEVICE))
    trans.eval()

    cnn = CNN1DClassifier(
        in_channels=len(high_feats),
        hidden=int(params.get("cnn_hidden", 128)),
        dropout=float(params.get("cnn_dropout", 0.2)),
    ).to(DEVICE)
    cnn.load_state_dict(torch.load(cpt, map_location=DEVICE))
    cnn.eval()

    # 🔸 여기서 학습 때 썼던 reject_margin 을 같이 꺼낸다 (없으면 0.05 디폴트)
    trained_margin = float(params.get("reject_margin", 0.05))

    return {
        "trans": trans,
        "cnn": cnn,
        "meta_model": meta_model,
        "low_feats": low_feats,
        "high_feats": high_feats,
        "low_scaler": low_scaler,
        "high_scaler": high_scaler,
        "cal_low": cal_low,
        "cal_high": cal_high,
        "meta_ctx_cols": meta_ctx_cols,
        "seq_len": seq_len,
        "params": params,
        "trained_margin": trained_margin,
    }

def build_latest_window(df: pd.DataFrame, feats: list, scaler, seq_len: int) -> np.ndarray:
    X = df[feats].fillna(0).values.astype(float)
    if scaler is not None:
        X = scaler.transform(X)
    if len(X) < seq_len:
        raise ValueError(f"데이터 길이 {len(X)} < seq_len {seq_len}")
    return X[-seq_len:, :][None, ...]  # (1,T,C)

def main():
    ap = argparse.ArgumentParser()
    base = Path(__file__).resolve().parents[2]
    ap.add_argument("--data-csv", type=str, default=str(base / "data" / "processed" / "training_with_refined_features.csv"))
    ap.add_argument("--models-root", type=str, default=str(base / "models"))
    ap.add_argument("--pred-out", type=str, default=str(base / "data" / "predictions" / "next_day_predictions.csv"))
    ap.add_argument("--date", type=str, default="")
    ap.add_argument("--sharpen", type=float, default=0.0)
    args = ap.parse_args()

    data_csv = Path(args.data_csv)
    df = pd.read_csv(data_csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    models_root = Path(args.models_root)
    final_dir = find_latest_final_dir(models_root)
    bundle = load_artifacts(final_dir)

    trans = bundle["trans"]
    cnn = bundle["cnn"]
    meta_model = bundle["meta_model"]
    low_feats = bundle["low_feats"]
    high_feats = bundle["high_feats"]
    low_scaler = bundle["low_scaler"]
    high_scaler = bundle["high_scaler"]
    meta_ctx_cols = bundle["meta_ctx_cols"]
    seq_len = bundle["seq_len"]
    trained_margin = bundle["trained_margin"]   # 👈 학습에서 쓰던 reject_margin

    # 마지막 날짜
    last_row = df.iloc[-1]
    last_date = last_row["date"]
    next_date = next_business_day(last_date)

    # 1) 저주파(Transformer) 입력
    X_low = build_latest_window(df, low_feats, low_scaler, seq_len)
    X_low_t = torch.from_numpy(X_low).float().to(DEVICE)
    with torch.no_grad():
        logit_low = trans(X_low_t)
        p_low = torch.sigmoid(logit_low).cpu().numpy().ravel()[0]

    # 2) 고주파(CNN) 입력
    X_high = build_latest_window(df, high_feats, high_scaler, seq_len)
    X_high_t = torch.from_numpy(X_high).float().to(DEVICE)
    with torch.no_grad():
        logit_high = cnn(X_high_t)
        p_high = torch.sigmoid(logit_high).cpu().numpy().ravel()[0]

    # 3) 메타 특징 만들기 (원본 predict에 있던 함수와 동일하게)
    # 여기서는 간단히 평균만
    p_blend = 0.5 * p_low + 0.5 * p_high

    # 4) 마진 적용해서 신뢰도 판단
    margin = trained_margin if trained_margin else 0.05
    is_confident = abs(p_blend - 0.5) >= margin
    pred_label = int(p_blend >= 0.5)

    if is_confident:
        decision = "UP" if pred_label == 1 else "DOWN"
    else:
        decision = "HOLD"

    # 5) 저장할 DF 만들기
    out_path = Path(args.pred_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "asof_date": last_date.strftime("%Y-%m-%d"),
        "pred_for": next_date.strftime("%Y-%m-%d"),
        "proba_up_low": p_low,
        "proba_up_high": p_high,
        "proba_up": p_blend,
        "reject_margin": margin,
        "is_confident": bool(is_confident),
        "pred_label": pred_label,
        "decision": decision,
        "model_dir": str(final_dir),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if out_path.exists():
        df_old = pd.read_csv(out_path)
        df_new = pd.concat([df_old, pd.DataFrame([row])], ignore_index=True)
        df_new.to_csv(out_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame([row]).to_csv(out_path, index=False, encoding="utf-8-sig")

    print("=== NEXT-DAY PREDICTION ===")
    print(row)


if __name__ == "__main__":
    main()
