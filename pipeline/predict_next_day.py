#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_next_day.py
- safe load + missing feature handling
- driver feature by gradient (default) or perturbation
"""

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === 1) pickle 호환용 dummy ===
class ProbabilityCalibrator:
    def __init__(self, method="identity", **kwargs):
        self.method = method
        self.kwargs = kwargs

    def __call__(self, proba):
        return self.predict_proba(proba)

    def predict_proba(self, proba):
        return np.asarray(proba)


# === 2) 기본 모델 ===
class WaveAttTransformerClassifier(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        n_scales: int = 3,
    ):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_size, d_model)
        self.enc = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.head = torch.nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.enc(h)
        out = self.head(h[:, -1, :])
        return out


class CNN1DClassifier(torch.nn.Module):
    def __init__(self, in_channels: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2)
        self.bn1 = torch.nn.BatchNorm1d(hidden)
        self.conv2 = torch.nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm1d(hidden)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B,T,C) -> (B,C,T)
        x = x.transpose(1, 2)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x.mean(dim=-1)
        x = self.dropout(x)
        return self.head(x)


# === 3) torch safe load ===
def _load_torch_model_safely(model: torch.nn.Module, ckpt_path: Path) -> torch.nn.Module:
    state = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    new_state = {}
    for k, v in state.items():
        if k == "n_averaged":
            continue
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print("[WARN] missing keys in state_dict:", missing)
    if unexpected:
        print("[WARN] unexpected keys in state_dict:", unexpected)

    model.eval()
    return model


# === 4) 유틸 ===
def next_business_day(d: pd.Timestamp) -> pd.Timestamp:
    nd = d + pd.Timedelta(days=1)
    while nd.weekday() >= 5:
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


def build_latest_window(df: pd.DataFrame, feats: list, scaler, seq_len: int) -> np.ndarray:
    """
    - 모델이 기대하는 feat가 df에 없으면 0칼럼으로 만들어서라도 맞춰준다.
    - 순서는 feats 순서를 그대로 유지한다.
    - 반환: (1, T, F) 스케일 적용 후 numpy
    """
    df_work = df.copy()
    missing = [f for f in feats if f not in df_work.columns]
    if missing:
        print("[INFO] missing feature(s) in input CSV, filling with 0:", missing)
        for col in missing:
            df_work[col] = 0.0

    X = df_work[feats].fillna(0).values.astype(float)
    if scaler is not None:
        X = scaler.transform(X)

    if len(X) < seq_len:
        raise ValueError(f"데이터 길이 {len(X)} < seq_len {seq_len}")
    return X[-seq_len:, :][None, ...]


# === 4.5) 예측/영향도 유틸 ===
def _predict_pair_probs_raw(trans, cnn, X_low_t, X_high_t):
    """캘리브레이터 미적용, 시그모이드 확률(미분가능) 반환"""
    with torch.no_grad():
        p_low = torch.sigmoid(trans(X_low_t)).cpu().numpy().ravel()[0]
        p_high = torch.sigmoid(cnn(X_high_t)).cpu().numpy().ravel()[0]
    return float(p_low), float(p_high)


def _predict_pair_probs_cal(trans, cnn, X_low_t, X_high_t, cal_low, cal_high):
    """캘리브레이터 적용(미분 X)"""
    with torch.no_grad():
        p_low = torch.sigmoid(trans(X_low_t)).cpu().numpy().ravel()[0]
        p_high = torch.sigmoid(cnn(X_high_t)).cpu().numpy().ravel()[0]
    if cal_low is not None and hasattr(cal_low, "predict_proba"):
        p_low = float(cal_low.predict_proba([[p_low]])[0][0])
    if cal_high is not None and hasattr(cal_high, "predict_proba"):
        p_high = float(cal_high.predict_proba([[p_high]])[0][0])
    return float(p_low), float(p_high)


def compute_local_impacts_perturb(
    trans, cnn,
    X_low, X_high,
    low_feats, high_feats,
    blend_w_low=0.5
):
    """기존: 마지막 시점 피처를 baseline(0)으로 대체해 |Δp| 측정"""
    X_low_t = torch.from_numpy(X_low).float().to(DEVICE)
    X_high_t = torch.from_numpy(X_high).float().to(DEVICE)
    p_low0, p_high0 = _predict_pair_probs_raw(trans, cnn, X_low_t, X_high_t)
    p0 = blend_w_low * p_low0 + (1.0 - blend_w_low) * p_high0

    impacts = []
    for i, feat in enumerate(low_feats):
        Xl = X_low.copy()
        Xl[0, -1, i] = 0.0
        Xl_t = torch.from_numpy(Xl).float().to(DEVICE)
        p_low1, p_high1 = _predict_pair_probs_raw(trans, cnn, Xl_t, X_high_t)
        p1 = blend_w_low * p_low1 + (1.0 - blend_w_low) * p_high1
        impacts.append({"feat": feat, "side": "low", "delta": p1 - p0, "abs_delta": abs(p1 - p0)})

    for i, feat in enumerate(high_feats):
        Xh = X_high.copy()
        Xh[0, -1, i] = 0.0
        Xh_t = torch.from_numpy(Xh).float().to(DEVICE)
        p_low1, p_high1 = _predict_pair_probs_raw(trans, cnn, X_low_t, Xh_t)
        p1 = blend_w_low * p_low1 + (1.0 - blend_w_low) * p_high1
        impacts.append({"feat": feat, "side": "high", "delta": p1 - p0, "abs_delta": abs(p1 - p0)})

    impacts.sort(key=lambda d: d["abs_delta"], reverse=True)
    return impacts


def compute_local_importance_gradient(
    trans, cnn,
    X_low, X_high,
    low_feats, high_feats,
    blend_w_low=0.5
):
    """
    마지막 타임스텝(T-1)의 각 피처에 대한 |∂p/∂x| 계산 (p = blend(sigmoid(logit_low), sigmoid(logit_high)))
    캘리브레이터 미적용. 반환: abs_grad 큰 순으로 정렬된 리스트
    """
    trans.eval(); cnn.eval()

    Xl = torch.from_numpy(X_low).float().to(DEVICE)
    Xh = torch.from_numpy(X_high).float().to(DEVICE)
    Xl.requires_grad_(True)
    Xh.requires_grad_(True)

    # forward (미분 가능)
    logit_low = trans(Xl)
    logit_high = cnn(Xh)
    p_low = torch.sigmoid(logit_low).squeeze(-1)   # (B=1,)
    p_high = torch.sigmoid(logit_high).squeeze(-1) # (B=1,)
    p_blend = blend_w_low * p_low + (1.0 - blend_w_low) * p_high

    # backward: dp/dx
    trans.zero_grad(set_to_none=True)
    cnn.zero_grad(set_to_none=True)
    if Xl.grad is not None: Xl.grad.zero_()
    if Xh.grad is not None: Xh.grad.zero_()

    p_blend.backward(torch.ones_like(p_blend))  # d( sum p )/dx

    # 마지막 타임스텝의 그라디언트
    grad_low_last = Xl.grad[0, -1, :].detach().cpu().numpy()   # (F_low,)
    grad_high_last = Xh.grad[0, -1, :].detach().cpu().numpy()  # (F_high,)

    # 절대값 민감도
    impacts = []
    for i, feat in enumerate(low_feats):
        impacts.append({"feat": feat, "side": "low", "abs_grad": float(abs(grad_low_last[i]))})
    for i, feat in enumerate(high_feats):
        impacts.append({"feat": feat, "side": "high", "abs_grad": float(abs(grad_high_last[i]))})

    impacts.sort(key=lambda d: d["abs_grad"], reverse=True)
    return impacts


def apply_domain_cap(candidates, domain_map, caps):
    """
    candidates: 중요도 순 정렬된 리스트(dict들)
    domain_map: feat -> domain 문자열 매핑 함수
    caps: {"fx": 1, "commodity": 1, ...} 형태. 각 도메인에서 최대 몇 개까지 허용할지.
    반환: 상한 적용된 새 리스트(순서 유지)
    """
    used = {k: 0 for k in caps.keys()}
    out = []
    for c in candidates:
        dom = domain_map(c["feat"])
        if dom in caps:
            if used[dom] >= caps[dom]:
                continue
            used[dom] += 1
        out.append(c)
    return out


# === 5) artifacts ===
def load_artifacts(final_dir: Path):
    pkl = final_dir / "preproc_and_calibrators.pkl"
    xgbp = final_dir / "meta_xgb.pkl"
    tpt = final_dir / "transformer_low_final.pt"
    cpt = final_dir / "cnn_high_final.pt"

    bundle = joblib.load(pkl)
    meta_model = joblib.load(xgbp) if xgbp.exists() else None

    params = bundle["params"]
    low_feats = bundle["low_feats"]
    high_feats = bundle["high_feats"]
    low_scaler = bundle["low_scaler"]
    high_scaler = bundle["high_scaler"]
    cal_low = bundle.get("cal_low", None)
    cal_high = bundle.get("cal_high", None)
    seq_len = int(params["seq_len"])
    d_model, nhead = params["d_model_nhead"]
    dim_feedforward = int(params["dim_feedforward"])
    num_layers = int(params.get("num_layers", 2))
    dropout = float(params.get("transf_dropout", 0.2))
    n_scales = int(params.get("n_scales", 3))

    trans = WaveAttTransformerClassifier(
        input_size=len(low_feats),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        n_scales=n_scales,
    ).to(DEVICE)
    if tpt.exists():
        trans = _load_torch_model_safely(trans, tpt)
    else:
        print("[WARN] transformer checkpoint not found:", tpt)

    cnn = CNN1DClassifier(
        in_channels=len(high_feats),
        hidden=int(params.get("cnn_hidden", 128)),
        dropout=float(params.get("cnn_dropout", 0.2)),
    ).to(DEVICE)
    if cpt.exists():
        cnn = _load_torch_model_safely(cnn, cpt)
    else:
        print("[WARN] cnn checkpoint not found:", cpt)

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
        "seq_len": seq_len,
        "params": params,
        "trained_margin": trained_margin,
    }


# === 6) main ===
def main():
    ap = argparse.ArgumentParser()
    base = Path(__file__).resolve().parents[2]
    ap.add_argument("--data-csv", type=str, default=str(base / "data" / "processed" / "training_with_refined_features.csv"))
    ap.add_argument("--models-root", type=str, default=str(base / "models"))
    ap.add_argument("--pred-out", type=str, default=str(base / "data" / "predictions" / "next_day_predictions.csv"))
    ap.add_argument("--blend-w-low", type=float, default=0.5, help="low/high 블렌딩 가중치")
    ap.add_argument("--driver-method", type=str, default="gradient", choices=["gradient", "perturbation"],
                    help="driver 산출 방법 (gradient=기본, perturbation=baseline 대체)")
    ap.add_argument("--domain-cap", type=str, default="", help="예: fx:1,commodity:1 로 도메인 상한")
    args = ap.parse_args()

    # --- 데이터 로드
    data_csv = Path(args.data_csv)
    df = pd.read_csv(data_csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    # --- 아티팩트
    models_root = Path(args.models_root)
    final_dir = find_latest_final_dir(models_root)
    bundle = load_artifacts(final_dir)

    trans = bundle["trans"]
    cnn = bundle["cnn"]
    low_feats = bundle["low_feats"]
    high_feats = bundle["high_feats"]
    low_scaler = bundle["low_scaler"]
    high_scaler = bundle["high_scaler"]
    seq_len = bundle["seq_len"]
    trained_margin = bundle["trained_margin"]
    cal_low = bundle["cal_low"]
    cal_high = bundle["cal_high"]

    # --- 날짜
    last_row = df.iloc[-1]
    last_date = last_row["date"]
    next_date = next_business_day(last_date)

    # --- 윈도우 (스케일 적용 후)
    X_low = build_latest_window(df, low_feats, low_scaler, seq_len)
    X_high = build_latest_window(df, high_feats, high_scaler, seq_len)

    # --- 예측(캘리브레이터 적용: 출력용)
    p_low_out, p_high_out = _predict_pair_probs_cal(
        trans, cnn,
        torch.from_numpy(X_low).float().to(DEVICE),
        torch.from_numpy(X_high).float().to(DEVICE),
        cal_low, cal_high
    )
    p_blend = args.blend_w_low * p_low_out + (1.0 - args.blend_w_low) * p_high_out

    margin = trained_margin if trained_margin else 0.05
    is_confident = abs(p_blend - 0.5) >= margin
    pred_label = int(p_blend >= 0.5)
    decision = "HOLD" if not is_confident else ("UP" if pred_label == 1 else "DOWN")

    # --- driver 후보 계산
    if args.driver_method == "gradient":
        candidates = compute_local_importance_gradient(
            trans, cnn, X_low, X_high, low_feats, high_feats, blend_w_low=args.blend_w_low
        )
        # 키 이름 통일
        for c in candidates:
            c["score"] = c.pop("abs_grad")
    else:
        candidates = compute_local_impacts_perturb(
            trans, cnn, X_low, X_high, low_feats, high_feats, blend_w_low=args.blend_w_low
        )
        for c in candidates:
            c["score"] = c.pop("abs_delta")

    # --- (선택) 도메인 상한 적용
    def feat_domain(name: str) -> str:
        n = name.lower()
        if any(k in n for k in ["eur", "usd", "krw", "jpy", "cny", "aud", "gbp"]):
            return "fx"
        if any(k in n for k in ["wti", "brent", "oil", "xau", "gold", "silver", "copper"]):
            return "commodity"
        if "vkospi" in n or "vix" in n:
            return "vol"
        if any(k in n for k in ["cpi", "ppi", "gdp", "unemp", "m2"]):
            return "macro"
        return "other"

    caps = {}
    if args.domain_cap:
        # 예: "fx:1,commodity:1"
        for token in args.domain_cap.split(","):
            token = token.strip()
            if not token:
                continue
            k, v = token.split(":")
            caps[k.strip()] = int(v)
        candidates = apply_domain_cap(candidates, feat_domain, caps)

    # --- top1 선택
    top1 = candidates[0]
    driver_feat = top1["feat"]
    driver_side = top1["side"]
    driver_score = float(top1["score"])

    # 원시/스케일 값
    driver_raw = float(last_row.get(driver_feat, np.nan))
    if driver_side == "low":
        driver_scaled = float(X_low[0, -1, low_feats.index(driver_feat)])
    else:
        driver_scaled = float(X_high[0, -1, high_feats.index(driver_feat)])

    # --- 저장
    out_path = Path(args.pred_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "asof_date": last_date.strftime("%Y-%m-%d"),
        "pred_for": next_date.strftime("%Y-%m-%d"),
        "proba_up_low": float(p_low_out),
        "proba_up_high": float(p_high_out),
        "proba_up": float(p_blend),
        "reject_margin": float(margin),
        "is_confident": bool(is_confident),
        "pred_label": int(pred_label),
        "decision": decision,
        "model_dir": str(final_dir),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "driver_side": driver_side,
        "driver_feature": driver_feat,
        "driver_feature_raw": driver_raw,
        "driver_feature_scaled": driver_scaled,
        "driver_score": driver_score,
        "driver_method": args.driver_method,
        "domain_cap": args.domain_cap,
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
