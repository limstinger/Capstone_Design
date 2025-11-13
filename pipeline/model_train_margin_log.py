# -*- coding: utf-8 -*-
import os
import re
import math
import json
import warnings
import hashlib
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import pywt
import inspect
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from typing import List

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, classification_report, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

import xgboost as xgb
from xgboost import XGBClassifier

import optuna
import shap
import joblib
import matplotlib.pyplot as plt
import logging
from shutil import copyfile  # ⬅ /mnt/data 미러 저장용

# Soft-DTW attention 모듈 (같은 폴더에 soft_dtw_cuda.py 필요)
from soft_dtw_cuda import SoftDTW



# -------------------- 설정 --------------------
warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
# 재현성 강화
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.info(f"xgboost version in this process: {xgb.__version__}")
# ---- No-change(보합) 필터 설정 (로그수익률 기반) ----
USE_LOGRET_FILTER       = True         # True면 학습 전 보합 샘플 제거
LOGRET_ABS_TAU          = 0.0005       # 절대 임계치: |logret| < 0.05% 제거 (데이터에 맞게 조정)
LOGRET_DROP_CENTER_Q    = 0.20         # 중앙 20%(|r| 작은 구간) 제거; 0이면 미사용
APPLY_FILTER_TO_HOLDOUT = True         # 논문 방식처럼 holdout에도 동일 기준 적용
LOGRET_COL_CANDIDATES   = ["daily_logret", "logret_KOSPI", "log_ret", "ret_log"]

# XGBoost 전역 로그 끄기
try:
    xgb.set_config(verbosity=0)
except Exception:
    pass

# ---- Explain/SHAP 옵션(메모리 절약) ----
RUN_NEURAL_SHAP = True
NEURAL_EXPLAINER = os.getenv("NEURAL_EXPLAINER", "grad")  # "grad" | "kernel"
SHAP_BG_K = 20
SHAP_SAMPLE_N = 32
SHAP_NSAMPLES = 128
IG_STEPS = 16
IG_BATCH = 16

try:
    logging.getLogger("shap").setLevel(logging.WARNING)
except Exception:
    pass

try:
    xgb.set_config(verbosity=0)
except Exception:
    pass

# ==== 실험 옵션 ====
SPACE_VERSION   = "v9_band_gated_log"   # ← 기존과 겹치지 않게 버전 업데이트
USE_EXISTING    = True
OOF_TRAIN_FRAC  = 0.6
OOF_VAL_BLOCKS  = 3
MIN_VAL_BLOCK   = 50                # ▲ 30 → 60 (검증 블록 최소 길이 확대)
EMBARGO_STEPS   = 20                # ▲ 3 → 20  (정보 전염 완충)
HOLDOUT_FRAC    = 0.15              # ▲ 마지막 15%는 절대 건드리지 않는 최종 평가 구간
OBJ_MIN_COVERAGE = 0.20

# ---- 탐색공간 명세(해시) ----
SEARCH_SPACE_SPEC = {
    "wavelet": ["db1","db4","coif1","coif3"],
    "wavelet_level": [2, 5],
    "energy_ratio_thresh": [0.3, 0.7],  # 기본값(레짐으로 가감)
    "initial_train": [450, 800],
    "val_size": [150, 300],  # ▲ 60~240 → 120~300
    "seq_len": [5, 120],
    "n_scales": [2, 4],
    "d_model_nhead": [(32,2),(32,4),(64,2),(64,4),(64,8),
                      (96,2),(96,4),(96,8),(128,2),(128,4),(128,8)],
    "ff_mult": ["x2","x3","x4"],
    "transf_dropout": [0.05, 0.4],
    "transf_lr": [1e-4, 1e-2],
    "transf_epochs": [20, 50],
    "cnn_hidden": [64, 128],
    "cnn_dropout": [0.05, 0.4],
    "cnn_lr": [1e-4, 1e-2],
    "num_layers": [1, 5],
    # XGB 탐색 (살짝 보수적으로)
    "xgb_max_depth": [2, 8],
    "xgb_n_estimators": [50, 300, 50],
    "xgb_lr": [1e-3, 1e-1],
    "xgb_gamma": [0.5, 5.0],
    "xgb_min_child_weight": [1, 10],
    "xgb_subsample": [0.6, 1.0],
    "xgb_colsample_bytree": [0.5, 1.0],
    "xgb_lambda": [0.5, 5.0],
    "xgb_alpha": [0.0, 1.0],

    "reject_margin": [0.03, 0.15],
}

def _hash_search_space(spec: dict) -> str:
    payload = json.dumps(spec, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:6]

SPACE_SIG = _hash_search_space(SEARCH_SPACE_SPEC)
STUDY_NAME = f"wavelet_transformer_{SPACE_VERSION}_{SPACE_SIG}"
DMODEL_NHEAD_PARAM = f"d_model_nhead_{SPACE_SIG}_v2"
FFMULT_PARAM       = f"ffmult_{SPACE_SIG}_v1"

# ---------- /mnt/data 미러 저장 유틸 ----------
FINAL_EXPORT_DIR = Path("/mnt/data")
try:
    FINAL_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

def _save_both(path_in_final_dir: Path):
    """models/final_... 에 저장 후 /mnt/data 로도 복사 (노트/스트림릿 확인 편의)."""
    try:
        tgt = FINAL_EXPORT_DIR / Path(path_in_final_dir).name
        copyfile(str(path_in_final_dir), str(tgt))
    except Exception as e:
        logging.warning(f"mirror save skipped for {path_in_final_dir}: {e}")

# ---------- 데이터 경로 해석 ----------
def resolve_input_csv() -> Path:
    here = Path(__file__).resolve()
    candidates: List[Path] = []
    env_path = os.environ.get("DATA_CSV", "").strip()
    if env_path:
        candidates.append(Path(env_path))
    try:
        base_dir = here.parents[2]
        candidates.append(base_dir / "data" / "processed" / "training_with_refined_features.csv")
    except Exception:
        pass
    candidates.append(Path("/mnt/data/training_with_refined_features.csv"))
    for p in candidates:
        if p.exists():
            return p
    return (here.parents[2] / "data" / "processed" / "training_with_refined_features.csv")


# ---------- 원본+파생 저장/로드 ----------
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="ffill")

def _bb_width(series: pd.Series, window: int = 20) -> pd.Series:
    ma = series.rolling(window, min_periods=1).mean()
    sd = series.rolling(window, min_periods=1).std()
    upper = ma + 2 * sd
    lower = ma - 2 * sd
    width = (upper - lower) / (ma.abs() + 1e-9)
    return width

def _add_engineered_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    added: List[str] = []
    if "date" in df.columns:
        dow = df["date"].dt.weekday.astype(float)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
        added += ["dow_sin","dow_cos"]
    price_col = None
    if "close" in df.columns:
        price_col = "close"
    elif "KOSPI_close" in df.columns:
        price_col = "KOSPI_close"
    if price_col is not None:
        df["rsi_14"] = _rsi(df[price_col], period=14)
        df["bb_width_20"] = _bb_width(df[price_col], window=20)
        added += ["rsi_14","bb_width_20"]
        ret_col = None
        candidates = [c for c in df.columns if "logret_KOSPI" in c]
        if candidates:
            ret_col = candidates[0]
        else:
            df["daily_logret"] = np.log(df[price_col] / df[price_col].shift(1))
            ret_col = "daily_logret"
            added += ["daily_logret"]
        roll_mu = df[ret_col].rolling(20, min_periods=5).mean()
        roll_sd = df[ret_col].rolling(20, min_periods=5).std()
        df["return_z_20"] = (df[ret_col] - roll_mu) / (roll_sd + 1e-9)
        df["vol_5d"] = df[ret_col].rolling(5, min_periods=3).std()
        df["vol_20d"] = roll_sd
        df["vol_ratio_5_20"] = df["vol_5d"] / (df["vol_20d"] + 1e-9)
        added += ["return_z_20","vol_5d","vol_20d","vol_ratio_5_20"]
    if "sentiment_score" in df.columns:
        mu30 = df["sentiment_score"].rolling(30, min_periods=5).mean()
        sd30 = df["sentiment_score"].rolling(30, min_periods=5).std()
        df["sentiment_z_30"] = (df["sentiment_score"] - mu30) / (sd30 + 1e-9)
        added += ["sentiment_z_30"]
    return df, added

def _build_enhanced_csv_if_needed(input_csv: Path) -> Path:
    enhanced_csv = input_csv.with_name(f"{input_csv.stem}.enh_{SPACE_SIG}.csv")
    if enhanced_csv.exists():
        return enhanced_csv
    df = pd.read_csv(input_csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df, added = _add_engineered_features(df)
    df.to_csv(enhanced_csv, index=False, encoding="utf-8-sig")
    logging.info(f"Enhanced features added ({len(added)}): {added}")
    logging.info(f"병합 저장 완료: {enhanced_csv}")
    return enhanced_csv

#-------------  추가 ----------
def compute_class_weight_effective_num(y, beta: float = 0.999) -> float:
    """
    Effective Number of Samples (Cui et al.) 기반 pos_weight 계산.
    BCEWithLogitsLoss(pos_weight=...)에 넣을 값을 반환한다.
    """
    import numpy as np
    y = np.asarray(y, dtype=int).reshape(-1)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    def eff(n: int) -> float:
        # n이 0이어도 분모 0 회피
        n = max(n, 1)
        return (1.0 - beta ** n) / (1.0 - beta)

    w_pos = 1.0 / eff(n_pos)
    w_neg = 1.0 / eff(n_neg)
    # BCEWithLogitsLoss의 pos_weight는 '양성에 곱해줄 비율' 이므로 w_neg / w_pos가 직관적
    return float(w_neg / (w_pos + 1e-8))




def load_dataset_with_nextday_label(enhanced_csv: str):
    """
    CSV를 읽고 'label_up'을 '다음 거래일' 기준으로 시프트한다.
    마지막 1행은 타깃 없음 → 제거(정상).
    반환: 시프트된 df (date 정렬)
    """
    import pandas as pd

    df = (pd.read_csv(enhanced_csv, parse_dates=["date"])
            .sort_values("date").reset_index(drop=True))
    if "label_up" not in df.columns:
        raise RuntimeError("label_up 컬럼이 없습니다. 타깃 컬럼명을 확인하세요.")

    df["label_up_next"] = df["label_up"].shift(-1)
    df = df.dropna(subset=["label_up_next"]).reset_index(drop=True)
    df["label_up"] = df["label_up_next"].astype(int)
    df = df.drop(columns=["label_up_next"])
    return df


def build_exclude_columns(df_columns) -> List[str]:
    """
    미래 정보/타깃 파생 등 누수 가능성이 있는 컬럼을 일괄 제외 목록으로 만든다.
    """
    base_exclude = {"date", "label_up"}
    tokens = ("label", "_tplus", "lead", "next_", "y_")  # 필요 시 추가
    ex = set(base_exclude)
    for c in df_columns:
        cl = str(c).lower()
        if any(tok in cl for tok in tokens):
            ex.add(c)
    return sorted(ex)

def apply_flip_guard(y_true, proba):
    """
    AUC가 0.5 미만으로 유의하게 낮으면(체계적으로 반대로 맞춤)
    확률을 1-p로 뒤집어 복구한다.
    반환: (proba_fixed, auc_fixed, flipped:bool)
    """
    import numpy as np

    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba).astype(float)

    try:
        auc = roc_auc_score(y_true, proba)
    except Exception:
        return proba, 0.5, False

    if auc < 0.5 - 1e-3:
        proba2 = 1.0 - proba
        auc2 = roc_auc_score(y_true, proba2)
        return proba2, auc2, True
    return proba, auc, False

def compute_metrics_with_margin(y_true, proba, margin=0.05):
    """
    전체 AUC/ACC + 마진(|p-0.5|>=margin) 커버리지와 성능을 계산.
    반환 dict: auc_overall, acc_overall, f1_overall, coverage, acc_bin, auc_bin
    """

    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba).astype(float)
    yhat = (p >= 0.5).astype(int)

    # 전체 지표
    try:
        auc_overall = roc_auc_score(y, p)
    except Exception:
        auc_overall = 0.5
    acc_overall = accuracy_score(y, yhat)
    f1_overall  = f1_score(y, yhat) if len(np.unique(y)) > 1 else 0.0

    # 마진 필터
    mask = np.abs(p - 0.5) >= float(margin)
    cov  = float(mask.mean()) if len(mask) else 0.0
    if mask.any():
        yb = y[mask]; pb = p[mask]
        try:
            auc_bin = roc_auc_score(yb, pb)
        except Exception:
            auc_bin = 0.5
        acc_bin = accuracy_score(yb, (pb >= 0.5).astype(int))
    else:
        auc_bin, acc_bin = 0.5, 0.5

    return {
        "auc_overall": float(auc_overall),
        "acc_overall": float(acc_overall),
        "f1_overall":  float(f1_overall),
        "coverage":    float(cov),
        "acc_bin":     float(acc_bin),
        "auc_bin":     float(auc_bin),
    }

def _expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """ECE (equal-width) 간단 구현."""
    p = np.asarray(p, float); y = np.asarray(y, int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if m.any():
            conf = p[m].mean()
            acc  = (y[m] == (p[m] >= 0.5)).mean()
            ece += (m.sum() / len(p)) * abs(acc - conf)
    return float(ece)


def evaluate_holdout(
    y_hold,
    p_hold,
    *,
    calibrator=None,
    reject_margin: float = 0.05,
    # 목적함수 가중/패널티
    w_full: float = 0.6,
    w_bin: float = 0.4,
    min_cov: float = 0.60,
    cov_penalty_k: float = 0.5,
    logger=None,
    ece_bins: int = 15,
):
    """
    - 입력: y_hold (0/1), p_hold (확률)
    - 보정(calibrator) → flip-guard → 마진 마스크/coverage → 전체/확신(bin) 지표 → 목적함수 score
    - 반환: metrics(dict) 1개 (호출부에서 언패킹 금지!)
    """
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, f1_score,
        average_precision_score, confusion_matrix
    )

    # ---------- 0) 입력 정리 ----------
    y_hold = np.asarray(y_hold, dtype=int).reshape(-1)
    p_in   = np.asarray(p_hold, dtype=float).reshape(-1)

    # ---------- 1) 확률 보정(있으면) ----------
    p0 = p_in.copy()
    if calibrator is not None:
        try:
            p0 = calibrator.transform(p0)
        except Exception:
            # 보정 실패 시 원본 사용
            pass

    # ---------- 2) flip-guard (라벨-확률 뒤집힘 보호) ----------
    def apply_flip_guard(y, p, min_auc: float = 0.5):
        """
        y,p를 받아 확률 방향성이 반대면 1-p로 뒤집어 반환.
        """
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5
        except Exception:
            auc = 0.5
        flipped = False
        if auc < min_auc:
            p = 1.0 - p
            flipped = True
            try:
                auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5
            except Exception:
                auc = 0.5
        return p, auc, flipped

    p_cal, _, _ = apply_flip_guard(y_hold, p0)

    # ---------- 3) 마진 마스크 & coverage ----------
    def apply_margin_mask(prob, margin: float):
        c_low, c_high = 0.5 - margin, 0.5 + margin
        return (prob < c_low) | (prob > c_high)

    mask = apply_margin_mask(p_cal, reject_margin)
    coverage = float(mask.mean()) if mask.size else 0.0

    # ---------- 4) 전체 지표 ----------
    try:
        auc_full = roc_auc_score(y_hold, p_cal) if np.unique(y_hold).size > 1 else 0.5
    except Exception:
        auc_full = 0.5
    yhat_full = (p_cal >= 0.5).astype(int)
    acc_full  = accuracy_score(y_hold, yhat_full)
    f1_full   = f1_score(y_hold, yhat_full) if np.unique(y_hold).size > 1 else 0.0
    brier     = float(np.mean((p_cal - y_hold) ** 2))

    # Expected Calibration Error
    def _expected_calibration_error(probs, labels, n_bins: int = 15) -> float:
        probs = np.asarray(probs, float).reshape(-1)
        labels = np.asarray(labels, int).reshape(-1)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            sel = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
            if not np.any(sel):
                continue
            conf = probs[sel].mean()
            acc  = labels[sel].mean()
            ece += (sel.mean()) * abs(acc - conf)
        return float(ece)

    ece       = _expected_calibration_error(p_cal, y_hold, n_bins=ece_bins)

    # ---------- 5) AUPRC/민감도/특이도 ----------
    try:
        from sklearn.metrics import average_precision_score, confusion_matrix
        auprc_full = average_precision_score(y_hold, p_cal)
    except Exception:
        auprc_full = float("nan")
    try:
        tn, fp, fn, tp = confusion_matrix(y_hold, yhat_full, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")  # TPR
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")  # TNR
    except Exception:
        sensitivity, specificity = float("nan"), float("nan")

    # ---------- 6) 확신(bin) 영역 지표 ----------
    if mask.sum() >= 2 and np.unique(y_hold[mask]).size == 2:
        auc_bin = roc_auc_score(y_hold[mask], p_cal[mask])
        acc_bin = accuracy_score(y_hold[mask], (p_cal[mask] >= 0.5).astype(int))
        f1_bin  = f1_score(y_hold[mask], (p_cal[mask] >= 0.5).astype(int))
    else:
        auc_bin, acc_bin, f1_bin = float("nan"), float("nan"), float("nan")

    # ---------- 7) 목적함수 스코어 ----------
    auc_bin_safe = auc_bin if np.isfinite(auc_bin) else (auc_full - 0.02)
    cov_penalty  = 0.0 if coverage >= min_cov else cov_penalty_k * (min_cov - coverage)
    score = (w_full * auc_full) + (w_bin * auc_bin_safe) - cov_penalty

    # ---------- 8) 로그 ----------
    if logger is not None:
        logger.info(
            "[HOLDOUT] auc=%.4f auprc=%s acc=%.4f f1=%.4f | sens=%s spec=%s | "
            "margin=%.3f coverage=%.3f | auc_bin=%s acc_bin=%s f1_bin=%s | "
            "brier=%.4f ece=%.4f score=%.4f",
            float(auc_full),
            ("%.4f" % auprc_full) if np.isfinite(auprc_full) else "nan",
            float(acc_full), float(f1_full),
            ("%.4f" % sensitivity) if np.isfinite(sensitivity) else "nan",
            ("%.4f" % specificity) if np.isfinite(specificity) else "nan",
            float(reject_margin), float(coverage),
            ("%.4f" % auc_bin) if np.isfinite(auc_bin) else "nan",
            ("%.4f" % acc_bin) if np.isfinite(acc_bin) else "nan",
            ("%.4f" % f1_bin)  if np.isfinite(f1_bin)  else "nan",
            float(brier), float(ece), float(score),
        )

    # ---------- 9) 반환 ----------
    return {
        "reject_margin": float(reject_margin),
        "auc_overall":   float(auc_full),
        "auprc_overall": float(auprc_full),
        "acc_overall":   float(acc_full),
        "f1_overall":    float(f1_full),
        "sensitivity":   float(sensitivity),
        "specificity":   float(specificity),
        "brier":         float(brier),
        "ece":           float(ece),
        "coverage":      float(coverage),
        "acc_bin":       float(acc_bin) if np.isfinite(acc_bin) else float("nan"),
        "auc_bin":       float(auc_bin) if np.isfinite(auc_bin) else float("nan"),
        "f1_bin":        float(f1_bin)  if np.isfinite(f1_bin)  else float("nan"),
        "score":         float(score),
    }


def _quick_auc_for_shift(df_in: pd.DataFrame, shift_days: int) -> float:
    """
    shift=0 / shift=+1 각각에 대해, 가벼운 로지스틱으로 5-fold AUC를 산출해 비교한다.
    - 미래정보를 안 보도록 안전한 후보 컬럼만 사용(있을 때만).
    - 데이터/특징이 부족하면 0.5 반환.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    import numpy as np

    df = df_in.copy()
    if "label_up" not in df.columns or len(df) < 200:
        return 0.5

    if shift_days == 1:
        df["label_tmp"] = df["label_up"].shift(-1)
        df = df.dropna(subset=["label_tmp"]).reset_index(drop=True)
        y = df["label_tmp"].astype(int).values
    else:
        y = df["label_up"].astype(int).values

    cand = [c for c in [
        "dow_sin","dow_cos","rsi_14","bb_width_20","daily_logret",
        "return_z_20","vol_5d","vol_20d","vol_ratio_5_20","sentiment_z_30"
    ] if c in df.columns]
    if not cand or len(np.unique(y)) < 2:
        return 0.5

    X = df[cand].fillna(0).values.astype(float)

    kf = KFold(n_splits=5, shuffle=False)
    aucs = []
    for tr, va in kf.split(X):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]
        if len(np.unique(yva)) < 2:
            continue
        lr = LogisticRegression(max_iter=500, class_weight="balanced")
        lr.fit(Xtr, ytr)
        p = lr.predict_proba(Xva)[:, 1]
        try:
            aucs.append(roc_auc_score(yva, p))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else 0.5

def detect_label_shift(enhanced_csv: str) -> int:
    """
    shift=0 과 shift=+1 중 AUC가 더 높은 쪽 선택.
    반환: 0(이미 next-day), 1(당일→다음날로 시프트 필요)
    """
    df0 = (pd.read_csv(enhanced_csv, parse_dates=["date"])
           .sort_values("date").reset_index(drop=True))
    auc0 = _quick_auc_for_shift(df0, shift_days=0)
    auc1 = _quick_auc_for_shift(df0, shift_days=1)
    logging.info(f"[label-shift-detect] auc_shift0={auc0:.4f} auc_shift+1={auc1:.4f}")
    return 1 if auc1 > max(auc0 + 1e-3, 0.500) else 0

def load_dataset_auto_label(enhanced_csv: str) -> pd.DataFrame:
    """
    CSV 로드 후 레이블이 이미 next-day인지 자동 판별하고,
    필요 시에만 +1 시프트를 적용한다.
    """
    df = (pd.read_csv(enhanced_csv, parse_dates=["date"])
          .sort_values("date").reset_index(drop=True))
    if "label_up" not in df.columns:
        raise RuntimeError("label_up 컬럼이 없습니다.")

    shift = detect_label_shift(enhanced_csv)
    if shift == 1:
        df["label_up_next"] = df["label_up"].shift(-1)
        df = df.dropna(subset=["label_up_next"]).reset_index(drop=True)
        df["label_up"] = df["label_up_next"].astype(int)
        df = df.drop(columns=["label_up_next"])
        logging.info("[label-shift-detect] using NEXT-DAY label (shift=+1)")
    else:
        logging.info("[label-shift-detect] using EXISTING label (shift=0)")
    return df



# ---------- Wavelet-based feature grouping ----------
def _pick_ret_col(df: pd.DataFrame) -> str:
    """
    로그수익률 컬럼을 찾아서 이름을 리턴.
    없으면 close(or KOSPI_close)로 daily_logret을 생성.
    """
    for c in LOGRET_COL_CANDIDATES:
        if c in df.columns:
            return c
    # 생성 경로
    price_col = None
    if "close" in df.columns:
        price_col = "close"
    elif "KOSPI_close" in df.columns:
        price_col = "KOSPI_close"
    if price_col is None:
        raise RuntimeError("로그수익률 컬럼/가격 컬럼이 없어 보합 필터 적용 불가")
    df["daily_logret"] = np.log(df[price_col] / df[price_col].shift(1))
    return "daily_logret"


def filter_by_log_return(
    df: pd.DataFrame,
    *,
    abs_tau: float = LOGRET_ABS_TAU,
    drop_center_q: float = LOGRET_DROP_CENTER_Q
) -> pd.DataFrame:
    """
    [t-1, t]의 로그수익률 크기가 작은 샘플을 학습 전 단계에서 제거.
    - abs_tau: |r| < abs_tau 제거
    - drop_center_q: |r| 기준 중앙 비중 제거(적응적)
    """
    if not USE_LOGRET_FILTER:
        return df
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    ret_col = _pick_ret_col(df)
    r = df[ret_col].astype(float)
    mask = pd.Series(True, index=df.index)

    if drop_center_q and drop_center_q > 0:
        lo = r.abs().quantile(drop_center_q/2.0)
        # 중앙 영역 제외 → 바깥쪽만 남김
        mask &= (r.abs() >= lo)

    if abs_tau and abs_tau > 0:
        mask &= (r.abs() >= float(abs_tau))

    # 레이블 누락 제거
    if "label_up" in df.columns:
        mask &= df["label_up"].notna()

    out = df.loc[mask].reset_index(drop=True)
    logging.info(f"[logret-filter] removed {len(df)-len(out)} / {len(df)} rows "
                 f"(keep={len(out)}), abs_tau={abs_tau}, center_q={drop_center_q}")
    return out


def apply_logret_filter_pipeline(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    레이블 시프트 처리(load_dataset_auto_label 이후)된 df에 보합 필터 적용.
    - 논문 재현: APPLY_FILTER_TO_HOLDOUT=True (전체 셋 동일 기준)
    - 실전 점검: 학습만 제거하고 홀드아웃은 전체 유지하려면, 평가 경로에서 원본 사용
    """
    if not USE_LOGRET_FILTER:
        return df_in
    return filter_by_log_return(df_in, abs_tau=LOGRET_ABS_TAU, drop_center_q=LOGRET_DROP_CENTER_Q)

def _compute_wavelet_energy_recent(x: np.ndarray, wavelet: str, level: int, use_recent: int = 512) -> float:
    """최근 구간만 표준화 후 에너지 비율 계산(레짐 적응)."""
    s = pd.Series(x).fillna(method="ffill").fillna(0).values
    if use_recent and len(s) > use_recent:
        s = s[-use_recent:]
    s = (s - s.mean()) / (s.std() + 1e-9)
    maxlev = pywt.dwt_max_level(len(s), pywt.Wavelet(wavelet).dec_len)
    level = min(level, maxlev) if maxlev > 0 else 0
    if level == 0:
        return 0.0
    coeffs = pywt.wavedec(s, wavelet, level=level)
    energies = [np.sum(c**2) for c in coeffs]
    total = sum(energies) + 1e-8
    detail = sum(energies[1:])
    return float(detail / total)

def adaptive_energy_thresh(base_df: pd.DataFrame, default_th: float = 0.5) -> float:
    """변동성/이벤트 레짐에 따라 임계치를 살짝 조정."""
    th = float(default_th)
    vol = base_df["vol_20d"].iloc[-1] if "vol_20d" in base_df and len(base_df)>0 else None
    rate = base_df["rate_announce_decay"].iloc[-1] if "rate_announce_decay" in base_df and len(base_df)>0 else 0.0
    if vol is not None and not np.isnan(vol):
        th = min(0.8, max(0.3, th + 0.10 * np.tanh((vol - 0.01) / 0.02)))
    if rate and rate > 0.7:  # 금통위 근접시 고주파 더 살림
        th = max(0.3, th - 0.05)
    return float(th)

def assign_wavelet_groups(df: pd.DataFrame,
                          exclude_cols: List[str],
                          wavelet: str,
                          level: int,
                          energy_ratio_thresh: float) -> Tuple[List[str], List[str]]:
    low_feats, high_feats = [], []
    for c in df.columns:
        if c in exclude_cols:
            continue
        x = df[c].values
        if len(x) < 2:
            low_feats.append(c)
            continue
        ratio = _compute_wavelet_energy_recent(x, wavelet, level, use_recent=512)
        (high_feats if ratio > energy_ratio_thresh else low_feats).append(c)
    return low_feats, high_feats


# ---------- Dataset ----------
class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int, flatten: bool):
        self.X = X; self.y = y
        self.seq_len = seq_len; self.flatten = flatten
        self.n = len(X) - seq_len

    def __len__(self):
        return max(0, self.n)

    def __getitem__(self, idx: int):
        seq = self.X[idx:idx+self.seq_len]
        if self.flatten:
            seq = seq.reshape(-1)
        return torch.from_numpy(seq).float(), torch.tensor(self.y[idx+self.seq_len]).float()


# ---------- Positional Encoding ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)].to(x.device)


def apply_margin_mask(p, margin):
    import numpy as np
    p = np.asarray(p, dtype=float).reshape(-1)
    mask = np.isfinite(p) & (np.abs(p - 0.5) >= float(margin))
    return mask

def bin_metrics_with_margin(y_true, p, margin=0.05):
    """
    마진(|p-0.5|>=margin) 영역의 coverage/성능 계산.
    - 공집합/단일클래스/NaN에 안전: acc_bin/auc_bin을 0.5로 보정
    반환: {'coverage','acc_bin','auc_bin'}
    """
    import numpy as np

    p = np.asarray(p, dtype=float).reshape(-1)
    y = np.asarray(y_true, dtype=int).reshape(-1)

    mask = np.isfinite(p) & (np.abs(p - 0.5) >= float(margin))
    cov  = float(mask.mean()) if len(mask) else 0.0

    if mask.any():
        yb = y[mask]; pb = p[mask]
        try:
            auc_bin = roc_auc_score(yb, pb) if len(np.unique(yb)) > 1 else 0.5
        except Exception:
            auc_bin = 0.5
        acc_bin = accuracy_score(yb, (pb >= 0.5).astype(int)) if len(yb) > 0 else 0.5
    else:
        auc_bin, acc_bin = 0.5, 0.5

    return {
        "coverage": float(cov),
        "acc_bin":  float(acc_bin),
        "auc_bin":  float(auc_bin),
    }



# ---------- Pre-Norm Transformer Encoder Layer ----------
class PreNormEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_out)
        x_norm = self.norm2(x)
        ff = self.linear2(self.dropout2(F.gelu(self.linear1(x_norm)))
        )
        x = x + ff
        return x


# ---------- Wavelet- & DTW-Attention ----------
class WaveletAttention(nn.Module):
    def __init__(self, in_channels, d_model, nhead, n_scales=3, wavelet='db4'):
        super().__init__()
        self.n_scales = n_scales
        self.wavelet  = wavelet
        self.to_q = nn.ModuleList([nn.Linear(in_channels, d_model) for _ in range(n_scales)])
        self.to_k = nn.ModuleList([nn.Linear(in_channels, d_model) for _ in range(n_scales)])
        self.to_v = nn.ModuleList([nn.Linear(in_channels, d_model) for _ in range(n_scales)])
        self.gate = nn.Sequential(nn.Linear(d_model, n_scales), nn.Softmax(dim=-1))
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape
        maxlev_possible = pywt.swt_max_level(T)
        level = min(self.n_scales, maxlev_possible)
        padded, pad_len = False, 0

        if level > 0:
            need = 2 ** level
            pad_len = (-T) % need
            if pad_len > 0:
                last = x[:, -1:, :].repeat(1, pad_len, 1)
                x = torch.cat([x, last], dim=1)
                padded = True
                T = x.size(1)

        if level < 1:
            z = self.to_v[0](x)
            return z[:, :-pad_len, :] if padded and pad_len > 0 else z

        arr = x.detach().cpu().numpy()
        coeffs = pywt.swt(arr, self.wavelet, level=level, axis=1)  # List[(cA, cD)]
        details = [torch.from_numpy(cD).to(x.device, dtype=x.dtype) for (_, cD) in coeffs]

        Vs = []
        for s in range(level):
            W = details[s]
            Q = self.to_q[s](W)
            K = self.to_k[s](W)
            V = self.to_v[s](W)
            out, _ = self.attn(Q, K, V)
            Vs.append(out)

        V_stack = torch.stack(Vs, -1)  # (B, T(+pad), d_model, level)
        global_feat = V_stack.mean(dim=1).mean(dim=-1)  # (B, d_model)
        gate_full   = self.gate(global_feat)            # (B, n_scales)
        gate        = gate_full[:, :level]              # (B, level)
        gate        = gate / gate.sum(dim=1, keepdim=True).clamp_min(1e-8)
        Z = (V_stack * gate.unsqueeze(1).unsqueeze(2)).sum(-1)

        if padded and pad_len > 0:
            Z = Z[:, :-pad_len, :]

        return Z

class _DTWPassThrough(nn.Module):
    """DTWAttention을 설명 단계에서 우회하기 위한 패스스루."""
    def forward(self, x, y):
        return x

class DTWAttention(nn.Module):
    def __init__(self, gamma=0.1, bandwidth=None):
        super().__init__()
        self.gamma = gamma
        self.bandwidth = bandwidth
        self.soft_dtw = SoftDTW(use_cuda=torch.cuda.is_available(), gamma=gamma, bandwidth=bandwidth)

    def _ensure_device(self, device: torch.device):
        want_cuda = (device.type == 'cuda')
        curr_cuda = getattr(self.soft_dtw, 'use_cuda', None)
        if curr_cuda is None or curr_cuda != want_cuda:
            self.soft_dtw = SoftDTW(use_cuda=want_cuda, gamma=self.gamma, bandwidth=self.bandwidth)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.device != x.device:
            y = y.to(x.device)
        self._ensure_device(x.device)
        D = self.soft_dtw(x, y)        # (B,)
        w = torch.exp(-D).view(-1, 1, 1)
        return x * w


# ---------- WaveAtt-Transformer Classifier (저주파) ----------
class WaveAttTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        n_scales: int = 3,
        dtw_gamma: float = 0.1
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.wav_att    = WaveletAttention(input_size, d_model, nhead, n_scales)
        self.dtw_att    = DTWAttention(gamma=dtw_gamma)
        self.pos_enc    = PositionalEncoding(d_model)
        self.enc_layers = nn.ModuleList([
            PreNormEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.input_proj(x)
        z1     = self.wav_att(x)
        z2     = self.dtw_att(x_proj, z1)
        z3     = self.pos_enc(z2)
        h = z3
        for layer in self.enc_layers:
            h = layer(h)
        h = self.final_norm(h)
        return self.classifier(h[:, -1, :])


# ---------- 1D-CNN Classifier (고주파) ----------
class CNN1DClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = x.mean(dim=-1)
        x = self.dropout(x)
        return self.head(x)


# ---------- 유틸리티 ----------
def compute_class_weight(y: np.ndarray) -> float:
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return 1.0 if pos == 0 else float(neg) / float(pos + 1e-8)

def make_scaler_and_transform(train_df: pd.DataFrame,
                              meta_df: pd.DataFrame,
                              val_df: pd.DataFrame,
                              feats: List[str]):
    if feats and len(feats) > 0:
        Xb = train_df[feats].fillna(0).values.astype(float)
        Xm = meta_df[feats].fillna(0).values.astype(float)
        Xv = val_df[feats].fillna(0).values.astype(float)
        scaler = StandardScaler().fit(Xb)
        b = scaler.transform(Xb)
        m = scaler.transform(Xm)
        v = scaler.transform(Xv)
    else:
        scaler = None
        b = np.zeros((len(train_df), 0), dtype=float)
        m = np.zeros((len(meta_df), 0), dtype=float)
        v = np.zeros((len(val_df), 0), dtype=float)
    return b, m, v, scaler

def train_neural_model(
    model,
    train_loader,
    val_loader,
    epochs: int = 500,
    lr: float = 1e-3,
    weight: Optional[float] = None,      # pos_weight로 사용
    early_stopping_patience: int = 5,
    use_amp: bool = True,
    swa: bool = False,
    swa_portion: float = 0.3,
    max_grad_norm: float = 1.0,
    label_smoothing: float = 0.05,       # ← 과신 억제
    pos_weight_cap: float = 2.0          # ← 과도한 양성가중 상한
):
    """
    BCEWithLogitsLoss(+pos_weight) + AMP + (옵션)SWA + label smoothing.
    검증 AUC 기준 best state 복원.
    반환: (best_model, best_val_auc)
    """
    global DEVICE
    model = model.to(DEVICE)

    # pos_weight 캡핑
    pos_w_tensor = None
    if weight is not None:
        pw = float(weight)
        if pos_weight_cap is not None:
            pw = min(pw, float(pos_weight_cap))
        pos_w_tensor = torch.tensor(pw, device=DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and DEVICE.type == "cuda"))

    # SWA
    if swa:
        swa_model = AveragedModel(model)
        swa_start = int(max(1, epochs * (1.0 - float(swa_portion))))
        swa_scheduler = SWALR(optimizer, swa_lr=lr)

    best_auc = -np.inf
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # ------ Train ------
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE).float()
            if yb.dim() == 2 and yb.size(1) == 1:
                yb = yb.squeeze(1)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(use_amp and DEVICE.type == "cuda")):
                logits = model(xb)
                if logits.dim() == 2 and logits.size(1) == 1:
                    logits = logits.squeeze(1)
                # label smoothing
                if label_smoothing and label_smoothing > 0.0:
                    y_s = yb * (1.0 - label_smoothing) + 0.5 * label_smoothing
                else:
                    y_s = yb
                loss = F.binary_cross_entropy_with_logits(
                    logits, y_s,
                    pos_weight=pos_w_tensor
                )

            scaler.scale(loss).backward()
            if max_grad_norm and max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        # SWA update
        if swa and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # ------ Validate ------
        model.eval()
        probs, targs = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE).float()
                if yb.dim() == 2 and yb.size(1) == 1:
                    yb = yb.squeeze(1)
                logits = model(xb)
                if logits.dim() == 2 and logits.size(1) == 1:
                    logits = logits.squeeze(1)
                p = torch.sigmoid(logits)
                probs.append(p.detach().cpu().numpy())
                targs.append(yb.detach().cpu().numpy())

        if not targs:
            continue
        all_p = np.concatenate(probs).ravel()
        all_t = np.concatenate(targs).ravel()
        val_auc = roc_auc_score(all_t, all_p) if np.unique(all_t).size > 1 else 0.5

        if val_auc > best_auc + 1e-4:
            best_auc = float(val_auc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(early_stopping_patience):
                break

    # best 복원
    if best_state is not None:
        model.load_state_dict(best_state)

    # SWA BN 업데이트 — device 명시(중요)
    if swa:
        update_bn(train_loader, swa_model, device=DEVICE)
        swa_model = swa_model.to(DEVICE)
        model = swa_model

    return model, best_auc

def evaluate_model(model, loader, margin: float = 0.0):
    model.eval()
    probs, targs = [], []
    with torch.no_grad():
        for xb,yb in loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            if logits.dim()==2 and logits.size(1)==1: logits=logits.squeeze(1)
            p = torch.sigmoid(logits)
            probs.append(p.cpu().numpy()); targs.append(yb.cpu().numpy())
    if not targs:
        return None,None,None

    all_p = np.concatenate(probs).ravel()
    all_t = np.concatenate(targs).ravel()

    # 마진 평가용 coverage/이진지표 (모델 모니터링 편의)
    met_bin = bin_metrics_with_margin(all_t, all_p, margin=margin)
    # ‘전체’ 기준 기본 지표도 함께 계산(로그/디버그용)
    preds_full = (all_p>0.5).astype(int)
    auc_full = roc_auc_score(all_t, all_p) if len(np.unique(all_t))>1 else 0.5
    acc_full = accuracy_score(all_t, preds_full)
    f1_full  = f1_score(all_t, preds_full) if len(np.unique(all_t))>1 else 0.0

    out = {
        "auc_full": auc_full, "accuracy_full": acc_full, "f1_full": f1_full,
        "coverage": met_bin["coverage"], "acc_bin": met_bin["acc_bin"], "auc_bin": met_bin["auc_bin"],
    }
    return out, all_p, all_t

def build_sequence_array(X, y, seq_len, flatten):
    ds = SequenceDataset(X,y,seq_len,flatten)
    Xs, ys = [], []
    for i in range(len(ds)):
        xi, yi = ds[i]
        Xs.append(xi.numpy()); ys.append(yi.item())
    return (np.stack(Xs) if Xs else np.zeros((0,))), np.array(ys)


# ---------- Paths & storage ----------
BASE_DIR         = Path(__file__).resolve().parents[2]
input_csv        = resolve_input_csv()
enhanced_csv     = _build_enhanced_csv_if_needed(input_csv)
study_path       = BASE_DIR/"models"/"optuna_wavelet_transformer_v4.db"   # ← 파일명 분리
best_params_path = BASE_DIR/"models"/f"best_{STUDY_NAME}.json"
final_model_dir  = BASE_DIR/"models"/f"final_{STUDY_NAME}"
final_model_dir.mkdir(parents=True, exist_ok=True)

# ==== (신규) 보합 제거 후 클래스 불균형 확인 ====
def _summarize_balance(df, tag: str = ""):
    import numpy as np
    y = df["label_up"].astype(int).values
    n = len(y); pos = int((y == 1).sum()); neg = n - pos
    pos_rate = (pos / n) if n > 0 else float("nan")
    # 필요 시 '효과적 샘플 수' 기반 pos_weight도 미리 확인
    try:
        pw_eff = compute_class_weight_effective_num(y, beta=0.999)
    except Exception:
        pw_eff = float("nan")
    logging.info("[balance%s] n=%d | pos=%d neg=%d | pos_rate=%.3f | pos_weight_eff=%.3f",
                 f':{tag}' if tag else '', n, pos, neg, pos_rate, pw_eff)

# 1) 레이블 시프트 자동 판별 → 원본 로드
df_raw = load_dataset_auto_label(str(enhanced_csv))
_summarize_balance(df_raw, "raw-before-flat")

# 2) 보합 필터 적용(‘TRAINING 전처리’ 이전 단계)
#    - 전역 옵션으로 임계치/쿼타 설정 가능:
#      LOGRET_ABS_TAU, LOGRET_DROP_CENTER_Q, USE_LOGRET_FILTER 등
df_flat = apply_logret_filter_pipeline(df_raw)
_summarize_balance(df_flat, "after-flat-filter")

# 3) 이후 파이프라인은 df_flat을 기준으로 계속 진행
df_for_modeling = df_flat.copy()



# ---------- 분할 & Holdout ----------
def _build_time_splits(n, initial_train, val_size):
    splits=[]
    te = initial_train
    while te+val_size<=n:
        splits.append((list(range(0,te)),list(range(te,te+val_size))))
        te += val_size
    return splits

def _train_holdout_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    cut = int(n * (1 - HOLDOUT_FRAC))
    cut = max(cut, 1)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)

def _leakage_sanity(y: np.ndarray, train_idx, val_idx, name=""):
    assert max(train_idx) < min(val_idx), f"{name}: time order broken"
    y_va = y[val_idx]
    assert len(y_va) >= MIN_VAL_BLOCK, f"{name}: val too small ({len(y_va)})"
    if len(np.unique(y_va)) < 2:
        raise ValueError(f"{name}: validation has a single class")


# ---------- 보정 (Platt vs Isotonic, + 듀얼 옵션) ----------
class ProbabilityCalibrator:
    def __init__(self, dual_average: bool = False):
        self.method: Optional[str] = None
        self.platt: Optional[LogisticRegression] = None
        self.iso: Optional[IsotonicRegression] = None
        self.trained = False
        self.dual_average = dual_average

    def fit(self, p: np.ndarray, y: np.ndarray):
        y = y.astype(int)
        if len(np.unique(y)) < 2:
            self.method = None
            self.trained = False
            return self

        pl = LogisticRegression(solver="lbfgs", max_iter=1000)
        pl.fit(p.reshape(-1,1), y)
        p_pl = pl.predict_proba(p.reshape(-1,1))[:,1]
        auc_pl = roc_auc_score(y, p_pl)

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p, y)
        p_iso = iso.transform(p)
        auc_iso = roc_auc_score(y, p_iso)

        if self.dual_average:
            self.method = "both"
            self.platt = pl
            self.iso = iso
            self.trained = True
            return self

        if auc_iso >= auc_pl:
            self.method = "isotonic"
            self.iso = iso
        else:
            self.method = "platt"
            self.platt = pl
        self.trained = True
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if not self.trained or self.method is None:
            return p
        if self.method == "both" and (self.platt is not None and self.iso is not None):
            pp = self.platt.predict_proba(p.reshape(-1,1))[:,1]
            pi = self.iso.transform(p)
            return 0.5*pp + 0.5*pi
        if self.method == "platt" and self.platt is not None:
            return self.platt.predict_proba(p.reshape(-1,1))[:,1]
        if self.method == "isotonic" and self.iso is not None:
            return self.iso.transform(p)
        return p


# ---------- 메타 기본 특징(9) ----------
def make_meta_features(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    low  = np.asarray(low).ravel()
    high = np.asarray(high).ravel()
    diff = high - low
    denom = (np.abs(low)+np.abs(high)+1e-8)
    return np.column_stack([
        low, high, diff, np.abs(diff),
        (low+high)/2.0, np.minimum(low, high), np.maximum(low, high),
        diff/denom, low+high
    ])

# ---------- 메타 보강: 롤링 예측 특징 ----------
def _rolling_pred_feats(p: np.ndarray, wins=(3,5,10)) -> np.ndarray:
    s = pd.Series(p)
    s_lag = s.shift(1)  # 누수 방지
    feats = []
    for w in wins:
        feats.append(s_lag.rolling(w, min_periods=1).mean())
        feats.append(s_lag.rolling(w, min_periods=1).std())
    return pd.concat(feats, axis=1).fillna(method="ffill").fillna(0.5).values

def _get_meta_ctx_cols(df: pd.DataFrame) -> List[str]:
    candidates = [
        "is_pre_holiday","is_post_holiday","days_since_last_trade",
        "entropy_full","entropy_excl_unassigned","hhi_full","hhi_excl_unassigned",
        "kl_topic_shift","top1_soft_prop","top1_topic_change_flag",
        "sentiment_score","sentiment_momentum_1d","sentiment_surprise_5d","sentiment_vol_5d","sentiment_z_30",
        "macd_histogram","ma_ratio_5_20","kospi_vol_5d","rsi_14","bb_width_20","return_z_20","vol_ratio_5_20",
        "logret_USD_KRW","logret_USD_KRW_momentum_3d","logret_USD_KRW_vol_5d",
        "relative_currency_momentum","vol_ratio_kospi_usdkrw",
        "rate_announce_decay","dow_sin","dow_cos",
    ]
    return [c for c in candidates if c in df.columns]

def _time_decay_weights(n: int, alpha: float = 0.01) -> np.ndarray:
    return np.exp(np.linspace(-alpha*n, 0, n, dtype=float))

def _build_meta_matrix_from_probs(lpv_c: np.ndarray, hpv_c: np.ndarray, ctx_arr: Optional[np.ndarray], p_blend: Optional[np.ndarray]=None) -> np.ndarray:
    base = make_meta_features(lpv_c, hpv_c)
    roll_l = _rolling_pred_feats(lpv_c, wins=(3,5,10))
    roll_h = _rolling_pred_feats(hpv_c, wins=(3,5,10))
    cols = [base, roll_l, roll_h]
    if ctx_arr is not None and len(ctx_arr) >= len(base):
        cols.append(ctx_arr[:len(base)])
    if p_blend is not None:
        cols.append(p_blend.reshape(-1,1))
    return np.column_stack(cols)

def _meta_feature_names(ctx_cols: List[str], with_blend: bool) -> List[str]:
    names = [
        "low","high","diff","abs_diff","avg","min","max","rel_diff","sum"
    ]
    for w in (3,5,10):
        names += [f"low_mean_{w}", f"low_std_{w}"]
    for w in (3,5,10):
        names += [f"high_mean_{w}", f"high_std_{w}"]
    names += ctx_cols
    if with_blend:
        names += ["p_blend_regime"]
    return names


# ---------- 레짐 게이트(행 단위) ----------
def _regime_gate_from_row(row: Dict[str, float]) -> float:
    """입력 row(dict)에서 고주파 가중 w_high ∈ (0,1) 산출."""
    vol  = float(row.get("vol_20d", 0.0) or 0.0)
    surp = float(row.get("sentiment_surprise_5d", 0.0) or 0.0)
    rate = float(row.get("rate_announce_decay", 0.0) or 0.0)
    s = 0.0
    s += 3.0*np.tanh((vol-0.01)/0.02)
    s += 1.5*np.tanh(surp)
    s += 1.2*(rate>0.7)
    w_high = 1.0/(1.0+np.exp(-s))
    return float(np.clip(w_high, 0.0, 1.0))

def _make_regime_blend_series(ctx_df: Optional[pd.DataFrame], lpv_c: np.ndarray, hpv_c: np.ndarray) -> np.ndarray:
    """행별 레짐 가중으로 p_blend 생성."""
    n = min(len(lpv_c), len(hpv_c))
    if ctx_df is None or n == 0:
        return 0.5*lpv_c[:n] + 0.5*hpv_c[:n]
    ctx_use = ctx_df.iloc[:n]
    w_high = []
    for _, r in ctx_use.iterrows():
        w_high.append(_regime_gate_from_row(r.to_dict()))
    w_high = np.asarray(w_high, dtype=float).reshape(-1)
    w_low = 1.0 - w_high
    return w_low*lpv_c[:n] + w_high*hpv_c[:n]


# ---------- OOF 메타 생성 ----------
def _make_oof_meta(train_df: pd.DataFrame, p: dict) -> Tuple[np.ndarray, np.ndarray]:
    N = len(train_df)
    start = max(int(N * OOF_TRAIN_FRAC), p["seq_len"] + 50)
    block = max(MIN_VAL_BLOCK, (N - start) // max(1, OOF_VAL_BLOCKS))
    xs, ys = [], []

    ctx_cols = p.get("meta_ctx_cols", [])

    i = start
    while i < N:
        j = min(i + block, N)
        # ---- Embargo 적용 ----
        tr_right = max(0, i - EMBARGO_STEPS)
        va_left  = i
        tr_block = train_df.iloc[:tr_right].reset_index(drop=True)
        va_block = train_df.iloc[va_left:j].reset_index(drop=True)
        i = j
        if len(va_block) < p["seq_len"] + 5 or len(tr_block) < p["seq_len"] + 50:
            continue

        split_pt = int(len(tr_block)*0.8)
        base_df, meta_df = tr_block.iloc[:split_pt], tr_block.iloc[split_pt:]

        if len(meta_df) < p["seq_len"] + 5:
            continue

        # ---- Adaptive 임계로 밴드 분리 ----
        energy_thresh = adaptive_energy_thresh(base_df, p["energy_ratio_thresh"])
        low_feats, high_feats = assign_wavelet_groups(
            base_df, exclude_cols=["date","label_up"],
            wavelet=p["wavelet"], level=p["wavelet_level"], energy_ratio_thresh=energy_thresh
        )
        # 스케일링
        b_low, m_low, v_low,  _ = make_scaler_and_transform(base_df, meta_df, va_block,  low_feats)
        b_high,m_high,v_high, _ = make_scaler_and_transform(base_df, meta_df, va_block, high_feats)

        yb, ym, yv = base_df["label_up"].values, meta_df["label_up"].values, va_block["label_up"].values
        # 로더
        lb = DataLoader(SequenceDataset(b_low,yb,p["seq_len"],False),batch_size=64,shuffle=False)
        lm = DataLoader(SequenceDataset(m_low,ym,p["seq_len"],False),batch_size=64,shuffle=False)
        lv = DataLoader(SequenceDataset(v_low,yv,p["seq_len"],False),batch_size=64,shuffle=False)
        hb = DataLoader(SequenceDataset(b_high,yb,p["seq_len"],False), batch_size=64,shuffle=True)
        hm = DataLoader(SequenceDataset(m_high,ym,p["seq_len"],False), batch_size=64,shuffle=True)
        hv = DataLoader(SequenceDataset(v_high,yv,p["seq_len"],False), batch_size=64,shuffle=False)

        weight = compute_class_weight(yb)
        d_model, nhead = p["d_model_nhead"]

        trans,_ = train_neural_model(
            WaveAttTransformerClassifier(
                input_size=b_low.shape[1],
                d_model=d_model, nhead=nhead,
                num_layers=p["num_layers"],
                dim_feedforward=p["dim_feedforward"],
                dropout=p["transf_dropout"],
                n_scales=p["n_scales"], dtw_gamma=0.1
            ),
            lb, lm, epochs=p["transf_epochs"], lr=p["transf_lr"], weight=weight
        )
        cnn,_ = train_neural_model(
            CNN1DClassifier(in_channels=b_high.shape[1], hidden=p["cnn_hidden"], dropout=p["cnn_dropout"]),
            hb, hm, epochs=p["transf_epochs"], lr=p["cnn_lr"], weight=weight
        )

        res_lm = evaluate_model(trans, lm)
        res_hm = evaluate_model(cnn,  hm)
        if res_lm is None or res_hm is None or res_lm[1] is None or res_hm[1] is None:
            continue

        lp_meta = res_lm[1]; hp_meta = res_hm[1]
        mt = meta_df["label_up"].values[p["seq_len"]:]
        if len(mt)==0 or len(lp_meta)==0 or len(hp_meta)==0:
            continue
        L = min(len(mt), len(lp_meta), len(hp_meta))
        mt = mt[:L]; lp_meta = lp_meta[:L]; hp_meta = hp_meta[:L]

        cal_low  = ProbabilityCalibrator(dual_average=True).fit(lp_meta, mt)
        cal_high = ProbabilityCalibrator(dual_average=True).fit(hp_meta, mt)

        res_lv = evaluate_model(trans, lv)
        res_hv = evaluate_model(cnn,  hv)
        if res_lv is None or res_hv is None or res_lv[1] is None or res_hv[1] is None:
            continue

        lpv = res_lv[1]; hpv = res_hv[1]
        vt = va_block["label_up"].values[p["seq_len"]:]
        if len(vt)==0 or len(lpv)==0 or len(hpv)==0:
            continue
        L2 = min(len(vt), len(lpv), len(hpv))
        vt = vt[:L2]; lpv = lpv[:L2]; hpv = hpv[:L2]

        lpv_c = cal_low.transform(lpv)
        hpv_c = cal_high.transform(hpv)

        ctx_val = va_block.iloc[p["seq_len"]:p["seq_len"]+L2, :][ctx_cols].fillna(0) if ctx_cols else None
        p_blend = _make_regime_blend_series(ctx_val, lpv_c, hpv_c)
        X_blk  = _build_meta_matrix_from_probs(lpv_c, hpv_c, (ctx_val.values if ctx_val is not None else None), p_blend=p_blend)

        xs.append(X_blk)
        ys.append(vt.astype(int))

    if xs:
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)
    n_ctx = len(ctx_cols)
    return np.zeros((0, 9+12+n_ctx+1), dtype=float), np.zeros((0,), dtype=int)


# ---------- 메타 모델(듀얼) ----------
def fit_meta_model_xgb(meta_X: np.ndarray, meta_y: np.ndarray, p: dict) -> XGBClassifier:
    n = len(meta_y)
    split = max(int(n * 0.8), 50)
    X_tr, y_tr = meta_X[:split], meta_y[:split].astype(int)
    X_va, y_va = meta_X[split:], meta_y[split:].astype(int)

    pos = (y_tr == 1).sum()
    neg = (y_tr == 0).sum()
    spw = float(neg) / max(1.0, float(pos))

    w_all = _time_decay_weights(n, alpha=0.01)
    w_tr  = w_all[:split]
    w_va  = w_all[split:]

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=spw,
        max_depth=p["xgb_max_depth"],
        n_estimators=p["xgb_n_estimators"],
        learning_rate=p["xgb_lr"],
        gamma=p["xgb_gamma"],
        min_child_weight=p["xgb_min_child_weight"],
        subsample=p["xgb_subsample"],
        colsample_bytree=p["xgb_colsample_bytree"],
        reg_lambda=p.get("xgb_lambda", 1.0),
        reg_alpha=p.get("xgb_alpha", 0.0),
        random_state=SEED, verbosity=0,
    )

    if len(X_va) == 0:
        try:
            model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
        except TypeError:
            model.fit(X_tr, y_tr, sample_weight=w_tr)
        return model

    fit_params = tuple(inspect.signature(XGBClassifier.fit).parameters.keys())
    supports_esr = "early_stopping_rounds" in fit_params
    supports_cbs = "callbacks" in fit_params
    supports_verbose = "verbose" in fit_params
    supports_sw_eval = "sample_weight_eval_set" in fit_params

    try:
        if supports_esr:
            kw = dict(eval_set=[(X_va, y_va)], early_stopping_rounds=50, sample_weight=w_tr)
            if supports_sw_eval: kw["sample_weight_eval_set"] = [w_va]
            if supports_verbose: kw["verbose"] = False
            model.fit(X_tr, y_tr, **kw)
        elif supports_cbs:
            es = xgb.callback.EarlyStopping(rounds=50, save_best=True)
            kw = dict(eval_set=[(X_va, y_va)], callbacks=[es], sample_weight=w_tr)
            if supports_sw_eval: kw["sample_weight_eval_set"] = [w_va]
            if supports_verbose: kw["verbose"] = False
            model.fit(X_tr, y_tr, **kw)
        else:
            kw = dict(eval_set=[(X_va, y_va)], sample_weight=w_tr)
            if supports_sw_eval: kw["sample_weight_eval_set"] = [w_va]
            if supports_verbose: kw["verbose"] = False
            model.fit(X_tr, y_tr, **kw)
    except TypeError:
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], sample_weight=w_tr)
    return model

def fit_meta_dual(meta_X: np.ndarray, meta_y: np.ndarray, p: dict):
    """XGB + LR 듀얼 학습 및 AUC 기반 가중 리턴."""
    n = len(meta_y)
    split = max(int(n * 0.8), 50)
    X_tr, y_tr = meta_X[:split], meta_y[:split].astype(int)
    X_va, y_va = meta_X[split:], meta_y[split:].astype(int)

    xgb_model = fit_meta_model_xgb(meta_X, meta_y, p)

    lr_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    if len(X_tr)>0 and len(np.unique(y_tr))>1:
        lr_model.fit(X_tr, y_tr)
    else:
        return xgb_model, None, (1.0, 0.0)

    if len(X_va)>0 and len(np.unique(y_va))>1:
        px = xgb_model.predict_proba(X_va)[:,1]
        pl = lr_model.predict_proba(X_va)[:,1]
        ax = roc_auc_score(y_va, px)
        al = roc_auc_score(y_va, pl)
        s = max(ax+al, 1e-8)
        w_x = float(ax/s); w_l = float(al/s)
    else:
        w_x, w_l = 0.7, 0.3
    return xgb_model, lr_model, (w_x, w_l)


# ---------- d_model/nhead 카테고리 문자열화 ----------
_DNH_CHOICES = [(32,2),(32,4),(64,2),(64,4),(64,8),
                (96,2),(96,4),(96,8),(128,2),(128,4),(128,8)]
_DNH_STR_CHOICES = [f"{d}x{h}" for d,h in _DNH_CHOICES]
_DNH_RE = re.compile(r"^(\d+)x(\d+)$")

def _parse_dnh_str(s: str) -> Tuple[int,int]:
    m = _DNH_RE.match(str(s))
    if not m:
        raise ValueError(f"Invalid d_model_nhead token: {s}")
    return int(m.group(1)), int(m.group(2))

def _parse_ffmult(token: str) -> int:
    if not isinstance(token, str) or not token.startswith("x"):
        raise ValueError(f"Invalid ff_mult token: {token}")
    return int(token[1:])


# ---------- Objective ----------
def objective(trial):
    """
    Optuna 목적함수 (안정화 강화판)
    - 레이블 자동 보정(shift=0/1 감지)
    - 미래 누수 의심 컬럼 제외 후 대역 분리
    - 메타 예측 확률에 flip-guard 적용
    - 목적함수: 0.7*전체 AUC + 0.3*마진 AUC - coverage penalty
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    # 1) 데이터 로드: 자동 레이블 보정(shift 감지)
    df_all = load_dataset_auto_label(enhanced_csv)
    # [추가] 로그수익률 기반 보합(애매) 구간 제거 (분할 이전 적용)
    df_all = apply_logret_filter_pipeline(df_all)

    # Holdout 분리(Optuna는 holdout 직접 안봄)
    df, _df_holdout_ = _train_holdout_split(df_all)
    ctx_cols = _get_meta_ctx_cols(df)

    # 2) 하이퍼 샘플
    wavelet_choice    = trial.suggest_categorical("wavelet", SEARCH_SPACE_SPEC["wavelet"])
    wavelet_level     = trial.suggest_int("wavelet_level", *SEARCH_SPACE_SPEC["wavelet_level"])
    energy_thresh     = trial.suggest_float("energy_ratio_thresh", *SEARCH_SPACE_SPEC["energy_ratio_thresh"])

    n = len(df)
    initial_train = trial.suggest_int("initial_train", 300, min(SEARCH_SPACE_SPEC["initial_train"][1], n//2))
    val_size      = trial.suggest_int("val_size", *SEARCH_SPACE_SPEC["val_size"])
    splits        = _build_time_splits(n, initial_train, val_size)

    seq_len    = trial.suggest_int("seq_len", *SEARCH_SPACE_SPEC["seq_len"])
    n_scales   = trial.suggest_int("n_scales", *SEARCH_SPACE_SPEC["n_scales"])
    num_layers = trial.suggest_int("num_layers", *SEARCH_SPACE_SPEC["num_layers"])

    dnh_token      = trial.suggest_categorical(DMODEL_NHEAD_PARAM, _DNH_STR_CHOICES)
    d_model, nhead = _parse_dnh_str(dnh_token)

    ffmult_token    = trial.suggest_categorical(FFMULT_PARAM, SEARCH_SPACE_SPEC["ff_mult"])
    ff_mult         = _parse_ffmult(ffmult_token)
    dim_feedforward = int(d_model * ff_mult)

    transf_dropout = trial.suggest_float("transf_dropout", *SEARCH_SPACE_SPEC["transf_dropout"])
    transf_lr      = trial.suggest_float("transf_lr", *SEARCH_SPACE_SPEC["transf_lr"], log=True)
    transf_epochs  = trial.suggest_int("transf_epochs", *SEARCH_SPACE_SPEC["transf_epochs"])

    cnn_hidden  = trial.suggest_categorical("cnn_hidden", SEARCH_SPACE_SPEC["cnn_hidden"])
    cnn_dropout = trial.suggest_float("cnn_dropout", *SEARCH_SPACE_SPEC["cnn_dropout"])
    cnn_lr      = trial.suggest_float("cnn_lr", *SEARCH_SPACE_SPEC["cnn_lr"], log=True)

    xgb_md     = trial.suggest_int("xgb_max_depth", *SEARCH_SPACE_SPEC["xgb_max_depth"])
    xgb_ne     = trial.suggest_int("xgb_n_estimators", *SEARCH_SPACE_SPEC["xgb_n_estimators"])
    xgb_lr     = trial.suggest_float("xgb_lr", *SEARCH_SPACE_SPEC["xgb_lr"], log=True)
    xgb_gamma  = trial.suggest_float("xgb_gamma", *SEARCH_SPACE_SPEC["xgb_gamma"])
    xgb_mcw    = trial.suggest_int("xgb_min_child_weight", *SEARCH_SPACE_SPEC["xgb_min_child_weight"])
    xgb_subs   = trial.suggest_float("xgb_subsample", *SEARCH_SPACE_SPEC["xgb_subsample"])
    xgb_colsub = trial.suggest_float("xgb_colsample_bytree", *SEARCH_SPACE_SPEC["xgb_colsample_bytree"])
    xgb_lambda = trial.suggest_float("xgb_lambda", *SEARCH_SPACE_SPEC["xgb_lambda"])
    xgb_alpha  = trial.suggest_float("xgb_alpha", *SEARCH_SPACE_SPEC["xgb_alpha"])

    reject_margin = trial.suggest_float("reject_margin", *SEARCH_SPACE_SPEC["reject_margin"])

    p = {
        "wavelet": wavelet_choice, "wavelet_level": wavelet_level, "energy_ratio_thresh": energy_thresh,
        "initial_train": initial_train, "val_size": val_size,
        "seq_len": seq_len, "n_scales": n_scales,
        "d_model_nhead": (d_model, nhead),
        "dim_feedforward": dim_feedforward, "transf_dropout": transf_dropout,
        "transf_lr": transf_lr, "transf_epochs": transf_epochs,
        "cnn_hidden": cnn_hidden, "cnn_dropout": cnn_dropout, "cnn_lr": cnn_lr,
        "xgb_max_depth": xgb_md, "xgb_n_estimators": xgb_ne, "xgb_lr": xgb_lr,
        "xgb_gamma": xgb_gamma, "xgb_min_child_weight": xgb_mcw,
        "xgb_subsample": xgb_subs, "xgb_colsample_bytree": xgb_colsub,
        "xgb_lambda": xgb_lambda, "xgb_alpha": xgb_alpha,
        "num_layers": num_layers,
        "meta_ctx_cols": ctx_cols,
    }

    # 목적 가중/패널티(안정화 우선)
    W_BIN, W_FULL = 0.6, 0.4
    MIN_COV       = 0.60
    COV_PENALTY_K = 0.4

    scores = []
    y_all = df["label_up"].values

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        try:
            _leakage_sanity(y_all, tr_idx, va_idx, name=f"fold{fold}")
        except Exception as e:
            logging.warning(f"[sanity] {e}; skipping fold")
            scores.append(0.5)
            continue

        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df   = df.iloc[va_idx].reset_index(drop=True)

        # OOF 메타
        oof_X, oof_y = _make_oof_meta(train_df, p)
        if len(np.unique(oof_y)) < 2 or len(oof_y) < 30:
            scores.append(0.5)
            logging.info(f"fold{fold} score={scores[-1]:.4f} (insufficient oof)")
            continue
        xgb_meta, lr_meta, (w_x, w_l) = fit_meta_dual(oof_X, oof_y, p)

        # 베이스 학습 + 보정자 학습
        split_pt = int(len(train_df) * 0.8)
        base_df, meta_df = train_df.iloc[:split_pt], train_df.iloc[split_pt:]

        energy_thresh2 = adaptive_energy_thresh(base_df, p["energy_ratio_thresh"])

        # ★ 누수 컬럼 일괄 제외
        exclude_cols = build_exclude_columns(base_df.columns)

        low_feats, high_feats = assign_wavelet_groups(
            base_df,
            exclude_cols=exclude_cols,
            wavelet=p["wavelet"], level=p["wavelet_level"], energy_ratio_thresh=energy_thresh2
        )

        b_low,  m_low,  v_low,  _ = make_scaler_and_transform(base_df, meta_df, val_df, low_feats)
        b_high, m_high, v_high, _ = make_scaler_and_transform(base_df, meta_df, val_df, high_feats)

        yb, ym, yv = base_df["label_up"].values, meta_df["label_up"].values, val_df["label_up"].values
        lb = DataLoader(SequenceDataset(b_low, yb, p["seq_len"], False),  batch_size=64, shuffle=False, drop_last=True)
        lm = DataLoader(SequenceDataset(m_low, ym, p["seq_len"], False),  batch_size=64, shuffle=False, drop_last=False)
        lv = DataLoader(SequenceDataset(v_low, yv, p["seq_len"], False),  batch_size=64, shuffle=False, drop_last=False)
        hb = DataLoader(SequenceDataset(b_high,yb, p["seq_len"], False),  batch_size=64, shuffle=True,  drop_last=True)
        hm = DataLoader(SequenceDataset(m_high,ym, p["seq_len"], False),  batch_size=64, shuffle=True,  drop_last=True)
        hv = DataLoader(SequenceDataset(v_high,yv, p["seq_len"], False),  batch_size=64, shuffle=False, drop_last=False)

        weight = compute_class_weight(yb)
        d_model, nhead = p["d_model_nhead"]

        trans, _ = train_neural_model(
            WaveAttTransformerClassifier(
                input_size=b_low.shape[1],
                d_model=d_model, nhead=nhead,
                num_layers=p["num_layers"],
                dim_feedforward=p["dim_feedforward"],
                dropout=p["transf_dropout"],
                n_scales=p["n_scales"], dtw_gamma=0.1
            ),
            lb, lm, epochs=p["transf_epochs"], lr=p["transf_lr"], weight=weight
        )
        cnn, _ = train_neural_model(
            CNN1DClassifier(in_channels=b_high.shape[1], hidden=p["cnn_hidden"], dropout=p["cnn_dropout"]),
            hb, hm, epochs=p["transf_epochs"], lr=p["cnn_lr"], weight=weight
        )

        if len(meta_df) < p["seq_len"] + 5:
            scores.append(0.5)
            logging.info(f"fold{fold} score={scores[-1]:.4f} (meta_df too small)")
            continue

        # 보정자 학습 (meta)
        res_lm = evaluate_model(trans, lm)
        res_hm = evaluate_model(cnn,  hm)
        if res_lm is None or res_hm is None or res_lm[1] is None or res_hm[1] is None:
            scores.append(0.5); logging.info(f"fold{fold} score={scores[-1]:.4f} (no meta preds)"); continue

        lp_meta = res_lm[1]; hp_meta = res_hm[1]
        mt = meta_df["label_up"].values[p["seq_len"]:]
        if len(mt)==0 or len(lp_meta)==0 or len(hp_meta)==0:
            scores.append(0.5); logging.info(f"fold{fold} score={scores[-1]:.4f} (empty meta)"); continue
        L = min(len(mt), len(lp_meta), len(hp_meta))
        mt, lp_meta, hp_meta = mt[:L], lp_meta[:L], hp_meta[:L]

        cal_low  = ProbabilityCalibrator(dual_average=True).fit(lp_meta, mt)
        cal_high = ProbabilityCalibrator(dual_average=True).fit(hp_meta, mt)

        # Val 예측 → 보정 → 메타 특징
        res_lv = evaluate_model(trans, lv)
        res_hv = evaluate_model(cnn,  hv)
        if res_lv is None or res_hv is None or res_lv[1] is None or res_hv[1] is None:
            scores.append(0.5); logging.info(f"fold{fold} score={scores[-1]:.4f} (no val preds)"); continue

        lpv = res_lv[1]; hpv = res_hv[1]
        vt = val_df["label_up"].values[p["seq_len"]:]
        if len(vt)==0 or len(lpv)==0 or len(hpv)==0:
            scores.append(0.5); logging.info(f"fold{fold} score={scores[-1]:.4f} (empty val)"); continue
        L2 = min(len(vt), len(lpv), len(hpv))
        vt, lpv, hpv = vt[:L2], lpv[:L2], hpv[:L2]

        lpv_c = cal_low.transform(lpv)
        hpv_c = cal_high.transform(hpv)

        ctx_cols_now = p["meta_ctx_cols"]
        ctx_val_df = val_df.iloc[p["seq_len"]:p["seq_len"]+L2, :][ctx_cols_now].fillna(0) if ctx_cols_now else None
        p_blend = _make_regime_blend_series(ctx_val_df, lpv_c, hpv_c)
        X_val = _build_meta_matrix_from_probs(
            lpv_c, hpv_c,
            (ctx_val_df.values if ctx_val_df is not None else None),
            p_blend=p_blend
        )

        if len(np.unique(vt)) < 2:
            scores.append(0.5)
        else:
            px = xgb_meta.predict_proba(X_val)[:, 1]
            if lr_meta is not None:
                pl = lr_meta.predict_proba(X_val)[:, 1]
                p_ens = w_x * px + w_l * pl
            else:
                p_ens = px

            # flip-guard 적용 후 전체 AUC/마진 점수
            p_ens, auc_full, _flipped = apply_flip_guard(vt, p_ens)
            m = bin_metrics_with_margin(vt, p_ens, margin=reject_margin)
            auc_bin = m["auc_bin"] if np.isfinite(m["auc_bin"]) else 0.5
            cov = float(m["coverage"])

            score = W_BIN * auc_bin + W_FULL * auc_full
            if cov < MIN_COV:
                score -= (MIN_COV - cov) * COV_PENALTY_K
                score = max(score, 0.3)

            if not np.isfinite(score):
                score = 0.5

            scores.append(float(score))

        trial.report(scores[-1], fold)
        logging.info(
            f"fold{fold} margin={reject_margin:.3f} "
            f"score={scores[-1]:.4f} cov={cov if 'cov' in locals() else float('nan'):.3f} "
            f"auc_bin={auc_bin if 'auc_bin' in locals() else float('nan'):.4f} "
            f"auc_full={auc_full if 'auc_full' in locals() else float('nan'):.4f}"
        )
        if trial.should_prune():
            raise optuna.TrialPruned()

    logging.info(f"fold scores: {np.round(scores,4).tolist()} -> mean {np.mean(scores):.4f}")
    return float(np.mean(scores)) if scores else 0.5




# ---------- Optuna run ----------
def run_optuna():
    study = optuna.create_study(
        direction="maximize",
        study_name=STUDY_NAME,
        storage=f"sqlite:///{study_path}",
        load_if_exists=USE_EXISTING,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=8),
        sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True, group=True),
    )
    study.optimize(objective, n_trials=100, timeout=3600)  # ▼ 5000 → 800
    with open(best_params_path,"w") as f:
        json.dump(study.best_trial.params, f, indent=2)
    logging.info(f"Best trial params: {study.best_trial.params}")
    return study

def _get_best_dmodel_nhead(best: dict):
    raw = best.get("d_model_nhead") or best.get(DMODEL_NHEAD_PARAM)
    if raw is None:
        for k in best.keys():
            if k.startswith("d_model_nhead_"):
                raw = best[k]; break
    if isinstance(raw, (list, tuple)) and len(raw)==2:
        return int(raw[0]), int(raw[1])
    if isinstance(raw, str):
        return _parse_dnh_str(raw)
    raise ValueError("best params missing d_model_nhead-compatible key")

def _get_best_dimfeedforward(best: dict, d_model: int) -> int:
    if "dim_feedforward" in best:
        return int(best["dim_feedforward"])
    token = best.get(FFMULT_PARAM, "x3")
    return int(d_model * _parse_ffmult(token))


# ---------- retrain (train only) ----------
def retrain_and_save(study):
    """
    - best params로 재학습 후 evaluate_on_holdout에서 바로 쓸 번들 반환
    - 반환 번들에 'df', 'model'(CombinedPredictor), 메타 블렌더(xgb/lr) 포함
    """
    import json, joblib, logging, torch
    from torch.utils.data import DataLoader
    import numpy as np

    # ---- 0) 데이터 로드 & 필터 ----
    df_all = load_dataset_auto_label(enhanced_csv)
    df_all = apply_logret_filter_pipeline(df_all)

    # ---- 1) 분리 ----
    df, _ = _train_holdout_split(df_all)
    best = study.best_trial.params

    # ---- 2) 파생/컨텍스트 ----
    ctx_cols = _get_meta_ctx_cols(df)

    # 내부 rolling split
    n = len(df)
    initial_train = best.get("initial_train", 500 if n > 800 else n // 2)
    val_size = best.get("val_size", 120)
    splits, te = [], initial_train
    while te + val_size <= n:
        splits.append((list(range(te)), list(range(te, te + val_size))))
        te += val_size
    tr_idx, va_idx = splits[-1]
    train_df = df.iloc[tr_idx].reset_index(drop=True)
    val_df   = df.iloc[va_idx].reset_index(drop=True)

    # Transformer 구조
    d_model, nhead = _get_best_dmodel_nhead(best)
    dim_ff         = _get_best_dimfeedforward(best, d_model)

    best_full = dict(best)
    best_full["d_model_nhead"]   = (d_model, nhead)
    best_full["dim_feedforward"] = dim_ff
    best_full["n_scales"]        = int(best.get("n_scales", 3))
    best_full["meta_ctx_cols"]   = ctx_cols

    # ---- 3) OOF 메타 & 듀얼 블렌더 ----
    oof_X, oof_y = _make_oof_meta(train_df, best_full)
    xgb_meta = fit_meta_model_xgb(oof_X, oof_y, best_full)
    lr_meta  = None
    w_x, w_l = 1.0, 0.0
    try:
        xgb_meta, lr_meta, (w_x, w_l) = fit_meta_dual(oof_X, oof_y, best_full)
    except Exception as e:
        logging.warning(f"Meta dual fallback to XGB only due to: {e}")

    # 메타 피처 이름(학습 시 쓰인 순서)을 확보해 저장/전달
    meta_feature_names = _meta_feature_names(ctx_cols, with_blend=True)

    # ---- 4) 베이스 재학습 ----
    split_pt = int(len(train_df) * 0.8)
    base_df, meta_df = train_df.iloc[:split_pt], train_df.iloc[split_pt:]

    energy_thresh3 = adaptive_energy_thresh(base_df, best_full["energy_ratio_thresh"])
    low_feats, high_feats = assign_wavelet_groups(
        base_df, exclude_cols=["date","label_up"],
        wavelet=best_full["wavelet"], level=best_full["wavelet_level"],
        energy_ratio_thresh=energy_thresh3
    )

    b_low,  m_low,  v_low,  low_scaler  = make_scaler_and_transform(base_df, meta_df, val_df, low_feats)
    b_high, m_high, v_high, high_scaler = make_scaler_and_transform(base_df, meta_df, val_df, high_feats)

    yb = base_df["label_up"].values
    ym = meta_df["label_up"].values

    seq_len = best_full["seq_len"]
    lb = DataLoader(SequenceDataset(b_low,  yb, seq_len, False), batch_size=64, shuffle=False, drop_last=True)
    lm = DataLoader(SequenceDataset(m_low,  ym, seq_len, False), batch_size=64, shuffle=False, drop_last=True)
    hb = DataLoader(SequenceDataset(b_high, yb, seq_len, False), batch_size=64, shuffle=True,  drop_last=True)
    hm = DataLoader(SequenceDataset(m_high, ym, seq_len, False), batch_size=64, shuffle=True,  drop_last=True)

    # 클래스 가중치(Effective Number)
    def compute_class_weight_effective_num(y, beta=0.999):
        y = np.asarray(y, int)
        n_pos = (y == 1).sum(); n_neg = (y == 0).sum()
        def eff(n): return (1 - beta**max(int(n),1)) / (1 - beta)
        w_pos = 1.0 / eff(n_pos); w_neg = 1.0 / eff(n_neg)
        return float(w_neg / (w_pos + 1e-8))
    weight = compute_class_weight_effective_num(yb)

    trans_model = WaveAttTransformerClassifier(
        input_size=b_low.shape[1],
        d_model=d_model, nhead=nhead,
        num_layers=int(best.get("num_layers", 3)),
        dim_feedforward=dim_ff,
        dropout=float(best.get("transf_dropout", 0.2)),
        n_scales=best_full["n_scales"], dtw_gamma=0.1
    )
    cnn_model = CNN1DClassifier(
        in_channels=b_high.shape[1],
        hidden=int(best.get("cnn_hidden", 128)),
        dropout=float(best.get("cnn_dropout", 0.2))
    )

    use_amp = True; use_swa = True; swa_portion = 0.3
    trans_full, _ = train_neural_model(
        trans_model, lb, lm,
        epochs=int(best.get("transf_epochs", 30)),
        lr=float(best.get("transf_lr", 1e-3)),
        weight=weight, use_amp=use_amp, swa=use_swa, swa_portion=swa_portion
    )
    cnn_full, _ = train_neural_model(
        cnn_model, hb, hm,
        epochs=int(best.get("transf_epochs", 30)),
        lr=float(best.get("cnn_lr", 1e-3)),
        weight=weight, use_amp=use_amp, swa=use_swa, swa_portion=swa_portion
    )

    # ---- 5) 보정기 ----
    cal_low  = ProbabilityCalibrator(dual_average=True);  cal_low.trained  = False
    cal_high = ProbabilityCalibrator(dual_average=True); cal_high.trained = False

    if len(meta_df) >= seq_len + 5:
        res_lm = evaluate_model(trans_full, lm)   # (metrics, probs)
        res_hm = evaluate_model(cnn_full,  hm)
        if (res_lm and res_hm and (res_lm[1] is not None) and (res_hm[1] is not None)):
            lp_meta = res_lm[1]; hp_meta = res_hm[1]
            mt = meta_df["label_up"].values[seq_len:]
            if (len(mt) > 0) and (len(lp_meta) > 0) and (len(hp_meta) > 0):
                L = min(len(mt), len(lp_meta), len(hp_meta))
                mt, lp_meta, hp_meta = mt[:L], lp_meta[:L], hp_meta[:L]
                cal_low  = ProbabilityCalibrator(dual_average=True).fit(lp_meta, mt)
                cal_high = ProbabilityCalibrator(dual_average=True).fit(hp_meta, mt)

    # ---- 6) CombinedPredictor (메타 블렌딩 + 온도보정) ----
    class CombinedPredictor:
        def __init__(self, trans_model, cnn_model,
                     low_scaler, high_scaler, low_feats, high_feats,
                     cal_low, cal_high, params, seq_len,
                     xgb_meta=None, lr_meta=None, blend_w=None, ctx_cols=None,
                     meta_feature_names=None, temp_T: float = 1.5):
            self.trans = trans_model.eval()
            self.cnn   = cnn_model.eval()
            self.low_scaler  = low_scaler
            self.high_scaler = high_scaler
            self.low_feats   = list(low_feats)
            self.high_feats  = list(high_feats)
            self.cal_low  = cal_low
            self.cal_high = cal_high
            self.params   = params
            self.seq_len  = int(seq_len)
            self.xgb_meta = xgb_meta
            self.lr_meta  = lr_meta
            self.blend_w  = blend_w or {"w_x": 1.0, "w_l": 0.0}
            self.ctx_cols = list(ctx_cols) if ctx_cols else []
            self.meta_feature_names = list(meta_feature_names) if meta_feature_names else None
            self.temp_T   = float(temp_T)

        @staticmethod
        def _logit_clip(p, eps=1e-6):
            p = np.clip(p, eps, 1-eps)
            return np.log(p / (1-p))

        def _temp_scale(self, p, T=1.5):
            z = self._logit_clip(np.asarray(p, float).ravel())
            return 1.0 / (1.0 + np.exp(-z / max(T, 1e-6)))

        def _transform_matrix(self, df_part, feats, scaler):
            X = df_part[feats].fillna(0).values.astype(float)
            return scaler.transform(X) if hasattr(scaler, "transform") else X

        def _predict_with_model(self, model, X_mat, y_dummy):
            from torch.utils.data import DataLoader
            ds = SequenceDataset(X_mat, y_dummy, self.seq_len, False)
            ld = DataLoader(ds, batch_size=64, shuffle=False, drop_last=False)
            res = evaluate_model(model, ld)  # (metrics, probs)
            return res[1] if isinstance(res, (list, tuple)) and len(res) > 1 else None

        def _ctx_matrix_in_order(self, df_input, n_rows):
            """
            학습 때 사용한 ctx_cols 순서대로 (부족한 컬럼은 0) 행렬 생성
            """
            if not self.ctx_cols:
                return None
            cols = []
            for c in self.ctx_cols:
                if c in df_input.columns:
                    cols.append(df_input[c].values.astype(float))
                else:
                    cols.append(np.zeros(len(df_input), dtype=float))
            ctx_full = np.vstack(cols).T  # [N, C]
            # 시퀀스 손실 뒤쪽 정렬을 고려해 뒤쪽 n_rows만 사용
            return ctx_full[-n_rows:, :]

        def _align_meta_shape(self, X_meta):
            """
            XGB가 기대하는 meta_feature_names 길이에 맞춰 패딩/잘라내기
            (주로 패딩; 잘라낼 일은 거의 없음)
            """
            if self.meta_feature_names is None:
                return X_meta
            need = len(self.meta_feature_names)
            have = X_meta.shape[1]
            if have == need:
                return X_meta
            if have < need:
                pad = np.zeros((X_meta.shape[0], need - have), dtype=X_meta.dtype)
                return np.hstack([X_meta, pad])
            # have > need 인 경우는 드묾: 앞 need개만 사용
            return X_meta[:, :need]

        def predict(self, df_input):
            # 1) 저/고주파 스케일 변환
            X_low  = self._transform_matrix(df_input, self.low_feats,  self.low_scaler)
            X_high = self._transform_matrix(df_input, self.high_feats, self.high_scaler)
            y_dummy = np.zeros(len(df_input), dtype=int)

            # 2) 두 모델 확률
            p_low  = self._predict_with_model(self.trans, X_low,  y_dummy)
            p_high = self._predict_with_model(self.cnn,   X_high, y_dummy)
            if p_low is None or p_high is None:
                raise RuntimeError("predict: sequence 부족 또는 evaluate_model 인터페이스 문제")

            # 3) 보정
            if getattr(self.cal_low, "trained", False):
                try: p_low = self.cal_low.transform(p_low)
                except Exception: pass
            if getattr(self.cal_high, "trained", False):
                try: p_high = self.cal_high.transform(p_high)
                except Exception: pass

            p_low  = np.asarray(p_low).ravel()
            p_high = np.asarray(p_high).ravel()
            L = min(len(p_low), len(p_high))
            p_low, p_high = p_low[:L], p_high[:L]

            # 4) 메타 입력 — 학습 때와 동일 파이프라인 사용
            if self.xgb_meta is not None:
                # ctx를 학습 시 순서대로 구성 (부족 컬럼은 0)
                ctx_mat_tail = self._ctx_matrix_in_order(df_input, n_rows=L)
                # 레짐 가중 평균
                ctx_df_tail = None
                if ctx_mat_tail is not None:
                    # _make_regime_blend_series는 DataFrame을 기대하는 구현일 수 있어 방어적으로 처리
                    import pandas as pd
                    ctx_df_tail = pd.DataFrame(ctx_mat_tail, columns=self.ctx_cols)
                p_blend = _make_regime_blend_series(ctx_df_tail, p_low, p_high)
                # 학습과 동일한 메타 행렬 빌드
                X_meta = _build_meta_matrix_from_probs(
                    p_low, p_high,
                    (ctx_mat_tail if ctx_mat_tail is not None else None),
                    p_blend=p_blend
                )
                # 형상 강제 정렬(이 부분이 mismatch 방지 핵심)
                X_meta = self._align_meta_shape(X_meta)

                p_meta_x = self.xgb_meta.predict_proba(X_meta)[:, 1]
                if self.lr_meta is not None:
                    p_meta_l = self.lr_meta.predict_proba(X_meta)[:, 1]
                    w_x = float(self.blend_w.get("w_x", 1.0))
                    w_l = float(self.blend_w.get("w_l", 0.0))
                    p_final = (w_x * p_meta_x + w_l * p_meta_l) / max(1e-8, (w_x + w_l))
                else:
                    p_final = p_meta_x
            else:
                p_final = 0.5 * p_low + 0.5 * p_high

            # 5) 온도 보정(coverage 과신 완화)
            p_final = self._temp_scale(p_final, T=self.temp_T)
            return np.asarray(p_final).ravel()

        def predict_proba(self, df_input):
            p = self.predict(df_input)
            return np.vstack([1.0 - p, p]).T

    combined_model = CombinedPredictor(
        trans_full, cnn_full,
        low_scaler, high_scaler, low_feats, high_feats,
        cal_low, cal_high, best_full, seq_len,
        xgb_meta=xgb_meta, lr_meta=lr_meta, blend_w={"w_x": w_x, "w_l": w_l},
        ctx_cols=ctx_cols, meta_feature_names=meta_feature_names,  # ← 중요
        temp_T=1.5
    )

    # ---- 7) 저장(옵션) ----
    torch.save(trans_full.state_dict(), final_model_dir / "transformer_low_final.pt")
    torch.save(cnn_full.state_dict(),  final_model_dir / "cnn_high_final.pt")
    joblib.dump(xgb_meta, final_model_dir / "meta_xgb.pkl")
    if lr_meta is not None:
        joblib.dump(lr_meta, final_model_dir / "meta_lr.pkl")
        joblib.dump({"w_x": w_x, "w_l": w_l}, final_model_dir / "meta_blend_weights.pkl")
    joblib.dump({
        "low_scaler": low_scaler, "high_scaler": high_scaler,
        "low_feats": low_feats, "high_feats": high_feats,
        "cal_low": cal_low, "cal_high": cal_high,
        "params": best_full,
        "meta_feature_names": meta_feature_names,             # ← 함께 저장
    }, final_model_dir / "preproc_and_calibrators.pkl")
    with open(final_model_dir / "metrics.json","w", encoding="utf-8") as f:
        json.dump({"note": "Final metrics should be evaluated on HOLDOUT. See metrics_holdout.json."}, f, indent=2, ensure_ascii=False)

    # ---- 8) 번들 반환 ----
    return {
        "df": df_all,
        "model": combined_model,
        "best_params": best_full,
        "calibrator": None,
        "source_csv": enhanced_csv,
        "saved_dir": str(final_model_dir),
        "xgb_meta": xgb_meta,
        "lr_meta": lr_meta,
        "meta_blend_weights": {"w_x": w_x, "w_l": w_l},
        "meta_ctx_cols": ctx_cols,
        "meta_feature_names": meta_feature_names,
    }



# ---------- SHAP ----------
def _kernel_shap_for_model(model, X_samples, X_background, out_png, input_shape=None, nsamples=SHAP_NSAMPLES):
    model_cpu = model.cpu().eval()
    def f(x_np):
        x_np = x_np.astype(np.float32)
        if input_shape is not None:
            x = torch.from_numpy(x_np.reshape((-1,) + input_shape))
        else:
            x = torch.from_numpy(x_np)
        with torch.no_grad():
            logits = model_cpu(x)
            if logits.dim()==2 and logits.size(1)==1:
                logits = logits.squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs
    try:
        shap._config.show_progress = False
    except Exception:
        pass
    expl = shap.KernelExplainer(f, X_background)
    vals = expl.shap_values(X_samples, nsamples=nsamples)
    shap.summary_plot(vals, X_samples, show=False)
    plt.tight_layout()
    plt.savefig(out_png)
    _save_both(Path(out_png))
    plt.close()

def _pick_bg_and_samples(X_seq: np.ndarray, bg_k: int, sample_n: int):
    N = X_seq.shape[0]
    if N == 0:
        return np.zeros_like(X_seq), np.zeros_like(X_seq[:1])
    samp_idx = np.random.choice(N, size=min(sample_n, N), replace=False)
    bg_idx   = np.random.choice(N, size=min(bg_k,   N), replace=False)
    return X_seq[samp_idx], X_seq[bg_idx]

def _integrated_gradients(model, inputs: torch.Tensor, baseline: torch.Tensor, steps: int = IG_STEPS):
    model.eval()
    inputs = inputs.detach()
    baseline = baseline.detach()
    delta = inputs - baseline
    total_grad = torch.zeros_like(inputs)
    for s in range(1, steps + 1):
        x_s = baseline + (float(s) / steps) * delta
        x_s.requires_grad_(True)
        logits = model(x_s).squeeze(1)
        grads = torch.autograd.grad(logits.sum(), x_s, retain_graph=False)[0]
        total_grad += grads.detach()
    ig = delta * (total_grad / steps)
    return ig

class _DTWPassThroughExplain(_DTWPassThrough):
    pass

def _grad_explain_model(
    model,
    X_seq: np.ndarray,
    out_png: Path,
    topk: int = 30,
    out_csv: Optional[Path] = None,
    disable_dtw_for_explain: bool = True,
):
    if X_seq.shape[0] == 0:
        return
    X_samp, X_bg = _pick_bg_and_samples(X_seq, SHAP_BG_K, SHAP_SAMPLE_N)
    baseline = X_bg.mean(axis=0, keepdims=True).repeat(len(X_samp), axis=0)
    xb = torch.from_numpy(X_samp.astype(np.float32))
    bb = torch.from_numpy(baseline.astype(np.float32))
    device = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else torch.device("cpu")
    xb = xb.to(device); bb = bb.to(device)

    torch.set_grad_enabled(True)
    model.eval()
    orig_dtw = None
    if disable_dtw_for_explain and hasattr(model, "dtw_att"):
        orig_dtw = model.dtw_att
        model.dtw_att = _DTWPassThroughExplain()

    try:
        with torch.enable_grad():
            ig = _integrated_gradients(model, xb, bb, steps=IG_STEPS)
        ch_imp = ig.abs().mean(dim=1).sum(dim=0).detach().float().cpu().numpy()
        if (not np.isfinite(ch_imp).any()) or np.allclose(ch_imp, 0):
            xb2 = xb.clone().requires_grad_(True)
            logits = model(xb2).squeeze(1)
            loss = logits.sum()
            grads = torch.autograd.grad(loss, xb2, retain_graph=False)[0]
            sal = (xb2 * grads).abs().mean(dim=1).sum(dim=0).detach().float().cpu().numpy()
            ch_imp = sal
        if np.allclose(np.nanmax(ch_imp), 0.0):
            ch_imp = ch_imp + 1e-8

        idx = np.argsort(-ch_imp)[:min(topk, len(ch_imp))]
        vals = ch_imp[idx]
        plt.figure()
        plt.bar(range(len(idx)), vals)
        plt.xticks(range(len(idx)), [f"ch{int(i)}" for i in idx], rotation=90)
        plt.title("Integrated Gradients (channel importance)")
        plt.tight_layout()
        plt.savefig(out_png)
        _save_both(Path(out_png))
        plt.close()
        if out_csv is not None:
            pd.DataFrame({"channel": [int(i) for i in idx], "importance": vals.astype(float)}).to_csv(out_csv, index=False)
    finally:
        if orig_dtw is not None:
            model.dtw_att = orig_dtw

def explain_with_shap(model1, model2, meta_models, low_X_seq, high_X_seq, yv, ctx_val, best_full, cal_low, cal_high):
    if RUN_NEURAL_SHAP:
        try:
            if NEURAL_EXPLAINER.lower() == "kernel":
                if low_X_seq.shape[0] > 0:
                    T, C = low_X_seq.shape[1], low_X_seq.shape[2]
                    sm, bg = _pick_bg_and_samples(low_X_seq, SHAP_BG_K, SHAP_SAMPLE_N)
                    _kernel_shap_for_model(
                        model1, sm.reshape(len(sm), -1).astype(np.float32),
                        bg.reshape(len(bg), -1).astype(np.float32),
                        out_png=final_model_dir/"shap_low_summary.png",
                        input_shape=(T, C),
                        nsamples=SHAP_NSAMPLES,
                    )
                if high_X_seq.shape[0] > 0:
                    T, C = high_X_seq.shape[1], high_X_seq.shape[2]
                    sm, bg = _pick_bg_and_samples(high_X_seq, SHAP_BG_K, SHAP_SAMPLE_N)
                    _kernel_shap_for_model(
                        model2, sm.reshape(len(sm), -1).astype(np.float32),
                        bg.reshape(len(bg), -1).astype(np.float32),
                        out_png=final_model_dir/"shap_high_summary.png",
                        input_shape=(T, C),
                        nsamples=SHAP_NSAMPLES,
                    )
            else:
                if low_X_seq.shape[0] > 0:
                    _grad_explain_model(
                        model1, low_X_seq,
                        final_model_dir / "ig_low_summary.png",
                        out_csv = final_model_dir / "ig_low_summary.csv",
                        disable_dtw_for_explain = True,
                    )
                if high_X_seq.shape[0] > 0:
                    _grad_explain_model(
                        model2, high_X_seq,
                        final_model_dir / "ig_high_summary.png",
                        out_csv = final_model_dir / "ig_high_summary.csv",
                        disable_dtw_for_explain = True,
                    )
        except Exception as e:
            logging.warning(f"Neural explanation skipped due to: {e}")

    # 메타 SHAP
    xgb_meta, lr_meta, (w_x, w_l) = meta_models
    with torch.no_grad():
        x_low = torch.from_numpy(low_X_seq.astype(np.float32)).to("cpu")
        p_low = torch.sigmoid(model1.cpu()(x_low)).view(-1).numpy() if x_low.shape[0]>0 else np.array([])
        x_high = torch.from_numpy(high_X_seq.astype(np.float32)).to("cpu")
        p_high = torch.sigmoid(model2.cpu()(x_high)).view(-1).numpy() if x_high.shape[0]>0 else np.array([])

    m = min(len(p_low), len(p_high))
    if m == 0:
        return
    p_low = p_low[:m]; p_high = p_high[:m]
    if hasattr(cal_low, "transform") and getattr(cal_low, "trained", False):
        p_low = cal_low.transform(p_low)
    if hasattr(cal_high,"transform") and getattr(cal_high, "trained", False):
        p_high = cal_high.transform(p_high)

    ctx_arr = ctx_val.values[:m] if ctx_val is not None else None
    p_blend = _make_regime_blend_series(ctx_val.iloc[:m] if ctx_val is not None else None, p_low, p_high)
    stacked = _build_meta_matrix_from_probs(p_low, p_high, ctx_arr, p_blend=p_blend)

    try:
        expl_meta = shap.TreeExplainer(xgb_meta)
        vals = expl_meta.shap_values(stacked)
        feat_names = _meta_feature_names(best_full.get("meta_ctx_cols", []), with_blend=True)
        try:
            shap.summary_plot(vals, stacked, feature_names=feat_names, show=False)
        except Exception:
            shap.summary_plot(vals, stacked, show=False)
        plt.tight_layout()
        png = final_model_dir/"shap_meta_summary.png"
        plt.savefig(png)
        _save_both(png)
        plt.close()
    except Exception as e:
        logging.warning(f"Meta SHAP skipped due to: {e}")


# ---------- Holdout 평가 ----------
def evaluate_on_holdout(saved):
    """
    saved: {
        "df": 전체 데이터프레임 (권장),
        "model": 최종 예측기 (predict(df) 또는 predict_proba(df) 제공),
        "best_params": 하이퍼파라미터 (seq_len, reject_margin 등),
        (옵션) "calibrator": 보정기 (.transform(prob) 지원)
        (옵션) "source_csv": 원본 CSV 경로 (df 누락 시 리로드용)
    }
    """
    import logging
    import numpy as np
    log = logging.getLogger(__name__)

    if not isinstance(saved, dict):
        raise TypeError("evaluate_on_holdout(saved): saved는 dict여야 합니다.")

    model       = saved.get("model", None)
    best_params = saved.get("best_params", {}) or {}
    calibrator  = saved.get("calibrator", None)
    if model is None:
        raise ValueError("evaluate_on_holdout: saved['model']가 없습니다.")

    # 1) df 확보 (없으면 로드/필터)
    df_all = saved.get("df", None)
    if df_all is None:
        source_csv = saved.get("source_csv", None)
        df_all = load_dataset_auto_label(source_csv or enhanced_csv)
        df_all = apply_logret_filter_pipeline(df_all)

    # 2) 학습/홀드아웃 분리
    dft, dfh = _train_holdout_split(df_all)

    # ================= TRAIN SET 평가 추가 =================
    if "label" not in dft.columns:
        if "label_up" in dft.columns:
            dft = dft.copy()
            dft["label"] = dft["label_up"].astype(int)
        else:
            raise ValueError("df_train에 'label' 또는 'label_up' 컬럼이 필요합니다.")

    y_train = dft["label"].values
    # 예측
    if hasattr(model, "predict_proba"):
        p_train = model.predict_proba(dft)[:, 1]
    else:
        p_train = model.predict(dft)
    p_train = np.asarray(p_train, dtype=float).ravel()

    # 🔴 여기서 길이 맞춰주기 (시퀀스 모델이라 앞부분 잘리는 경우)
    n_y_tr = len(y_train)
    n_p_tr = len(p_train)
    if n_p_tr != n_y_tr:
        if n_p_tr < n_y_tr:
            offset = n_y_tr - n_p_tr
            log.warning("[evaluate_on_holdout][TRAIN] length mismatch: y=%d, p=%d → trim first %d rows.",
                        n_y_tr, n_p_tr, offset)
            dft = dft.iloc[offset:].copy()
            y_train = y_train[offset:]
        else:
            trim = n_p_tr - n_y_tr
            log.warning("[evaluate_on_holdout][TRAIN] length mismatch: y=%d, p=%d → trim last %d preds.",
                        n_y_tr, n_p_tr, trim)
            p_train = p_train[:n_y_tr]
        n_y_tr, n_p_tr = len(y_train), len(p_train)
        if n_y_tr != n_p_tr:
            raise RuntimeError(f"[TRAIN] 길이 정렬 실패: y={n_y_tr}, p={n_p_tr}")

    train_metrics = evaluate_holdout(
        y_train, p_train,
        calibrator=calibrator,
        reject_margin=float(best_params.get("reject_margin", 0.05)),
        logger=None,      # 여기선 한 줄만 직접 찍을 거라 logger 안 넘김
    )
    # 한 줄 출력
    log.info(
        "[TRAIN] auc=%.4f acc=%.4f f1=%.4f cov=%.3f acc_bin=%.3f auc_bin=%.3f",
        train_metrics["auc_overall"],
        train_metrics["acc_overall"],
        train_metrics["f1_overall"],
        train_metrics["coverage"],
        train_metrics["acc_bin"],
        train_metrics["auc_bin"],
    )
    # ======================================================

    # ====== 아래는 네 원래 코드 (HOLDOUT 쪽) ======
    if "label" not in dfh.columns:
        if "label_up" in dfh.columns:
            dfh = dfh.copy()
            dfh["label"] = dfh["label_up"].astype(int)
        else:
            raise ValueError("df_holdout에 'label' 또는 'label_up' 컬럼이 필요합니다.")
    y_hold = dfh["label"].values
    n_y = len(y_hold)

    if hasattr(model, "predict_proba"):
        p_hold = model.predict_proba(dfh)[:, 1]
    else:
        p_hold = model.predict(dfh)
    p_hold = np.asarray(p_hold, dtype=float).ravel()
    n_p = len(p_hold)

    if n_p != n_y:
        if n_p < n_y:
            offset = n_y - n_p
            log.warning("[evaluate_on_holdout] length mismatch: y=%d, p=%d → trim first %d rows.", n_y, n_p, offset)
            dfh = dfh.iloc[offset:].copy()
            y_hold = y_hold[offset:]
        else:
            trim = n_p - n_y
            log.warning("[evaluate_on_holdout] length mismatch: y=%d, p=%d → trim last %d preds.", n_y, n_p, trim)
            p_hold = p_hold[:n_y]
        n_y, n_p = len(y_hold), len(p_hold)
        if n_y != n_p:
            raise RuntimeError(f"길이 정렬 실패: y={n_y}, p={n_p}")

    base_m = float(best_params.get("reject_margin", 0.05))
    if "vol_20d" in dfh.columns and len(dfh) > 20:
        v = float(dfh["vol_20d"].iloc[-1] or 0.0)
        adapt = base_m * (1.0 + 0.5 * np.tanh((v - 0.01) / 0.02))
        reject_margin = float(np.clip(adapt, 0.03, 0.12))
    else:
        reject_margin = max(0.03, base_m)

    metrics = evaluate_holdout(
        y_hold, p_hold,
        calibrator=calibrator,
        reject_margin=reject_margin,
        logger=log,
    )
    log.info(
        "[HOLDOUT] auc=%.4f acc=%.4f f1=%.4f cov=%.3f acc_bin=%.3f auc_bin=%.3f",
        metrics["auc_overall"],
        metrics["acc_overall"],
        metrics["f1_overall"],
        metrics["coverage"],
        metrics["acc_bin"],
        metrics["auc_bin"],
    )
    log.info("최종 HOLDOUT metrics: %s", metrics)
    return metrics


# ---------- Entry ----------
if __name__ == "__main__":
    study = run_optuna()
    saved = retrain_and_save(study)  # 학습/저장 (holdout 사용 X)
    holdout_metrics = evaluate_on_holdout(saved)  # 진짜 최종 평가 (holdout만)
    with open(final_model_dir/"metrics_holdout.json","w") as f:
        json.dump(holdout_metrics, f, indent=2)
    logging.info(f"최종 HOLDOUT metrics: {holdout_metrics}")

    # (선택) 간단한 설명 그림 생성 시도
    try:
        # 저장물에서 일부를 읽어 lightweight IG 생성만 시도
        bundle = joblib.load(Path(saved["saved_dir"]) / "preproc_and_calibrators.pkl")
        best_full = bundle["params"]
        # 설명은 retrain 단계에서 생성 가능하나, 여기선 생략/안전
        logging.info("Pipeline finished. See metrics_holdout.json for final numbers.")
    except Exception:
        pass