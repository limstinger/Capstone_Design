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

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, classification_report
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
SPACE_VERSION   = "v6_waveatt"   # ← 기존과 겹치지 않게 버전 업데이트
USE_EXISTING    = True
OOF_TRAIN_FRAC  = 0.6
OOF_VAL_BLOCKS  = 3
MIN_VAL_BLOCK   = 60                # ▲ 30 → 60 (검증 블록 최소 길이 확대)
EMBARGO_STEPS   = 15                # ▲ 3 → 15  (정보 전염 완충)
HOLDOUT_FRAC    = 0.15              # ▲ 마지막 15%는 절대 건드리지 않는 최종 평가 구간
OBJ_MIN_COVERAGE = 0.20

# ---- 탐색공간 명세(해시) ----
SEARCH_SPACE_SPEC = {
    "wavelet": ["db1","db4","coif1","coif3"],
    "wavelet_level": [2, 5],
    "energy_ratio_thresh": [0.3, 0.7],  # 기본값(레짐으로 가감)
    "initial_train": [300, 800],
    "val_size": [120, 300],  # ▲ 60~240 → 120~300
    "seq_len": [5, 120],
    "n_scales": [2, 4],
    "d_model_nhead": [(32,2),(32,4),(64,2),(64,4),(64,8),
                      (96,2),(96,4),(96,8),(128,2),(128,4),(128,8)],
    "ff_mult": ["x2","x3","x4"],
    "transf_dropout": [0.1, 0.4],
    "transf_lr": [1e-4, 1e-2],
    "transf_epochs": [20, 50],
    "cnn_hidden": [64, 128],
    "cnn_dropout": [0.1, 0.4],
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

    "reject_margin": [0.0, 0.2],
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


# ---------- Wavelet-based feature grouping ----------
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

def apply_margin_mask(p: np.ndarray, margin: float) -> np.ndarray:
    """
    확률 p에 대해 |p-0.5| >= margin 인 샘플만 True.
    (보합/유보 구간은 False로 제외)
    """
    if margin <= 0:
        return np.ones_like(p, dtype=bool)
    return (p < 0.5 - margin) | (p > 0.5 + margin)

def bin_metrics_with_margin(y_true: np.ndarray, p: np.ndarray, margin: float) -> Dict[str, float]:
    """
    보합(유보) 제외 구간에서의 이진 성능 지표 리턴.
    - coverage: 평가에 사용된 비율
    - acc_bin / auc_bin: 보합 제외 샘플의 정확도/ AUC
    빈 마스크면 coverage=0, acc_bin/auc_bin=nan
    """
    p = np.asarray(p).reshape(-1)
    y = np.asarray(y_true).astype(int).reshape(-1)
    mask = apply_margin_mask(p, margin)
    cov = float(mask.mean()) if len(mask) else 0.0
    if not mask.any():
        return {"coverage": cov, "acc_bin": float("nan"), "auc_bin": float("nan")}
    p_m = p[mask]
    y_m = y[mask]
    acc = accuracy_score(y_m, (p_m > 0.5).astype(int))
    auc = roc_auc_score(y_m, p_m) if len(np.unique(y_m)) > 1 else float("nan")
    return {"coverage": cov, "acc_bin": float(acc), "auc_bin": float(auc)}


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
def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    """단일 모델 실험용: date/label_up 등 제외하고 전 특징 사용."""
    exclude = {"date", "label_up"}
    return [c for c in df.columns if c not in exclude]

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

def train_neural_model(model, train_loader, val_loader,
                       epochs=500, lr=1e-3, weight=None,
                       early_stopping_patience=5):
    model = model.to(DEVICE)
    criterion = (nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight,device=DEVICE))
                 if weight is not None else nn.BCEWithLogitsLoss())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    best_auc, best_state, no_improve = -np.inf, None, 0

    for epoch in range(1, epochs+1):
        model.train()
        for xb,yb in train_loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            if yb.dim()==2 and yb.size(1)==1: yb=yb.squeeze(1)
            logits = model(xb)
            if logits.dim()==2 and logits.size(1)==1: logits=logits.squeeze(1)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        probs, targs = [], []
        with torch.no_grad():
            for xb,yb in val_loader:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                if logits.dim()==2 and logits.size(1)==1: logits=logits.squeeze(1)
                p = torch.sigmoid(logits)
                probs.append(p.cpu().numpy()); targs.append(yb.cpu().numpy())
        if not targs:
            continue
        all_p = np.concatenate(probs).ravel()
        all_t = np.concatenate(targs).ravel()
        val_auc = roc_auc_score(all_t, all_p) if len(np.unique(all_t))>1 else 0.5

        if val_auc > best_auc + 1e-4:
            best_auc, best_state, no_improve = val_auc, model.state_dict(), 0
        else:
            no_improve += 1
            if no_improve>=early_stopping_patience: break

    if best_state: model.load_state_dict(best_state)
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


def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    exclude = {"date", "label_up"}
    return [c for c in df.columns if c not in exclude]


# ---------- Objective ----------
# ---------- Objective (margin-aware AUC) ----------
def objective(trial):
    # ───────── 데이터 로드 & 홀드아웃 분리 ─────────
    df_all = (
        pd.read_csv(enhanced_csv, parse_dates=["date"])
          .sort_values("date").reset_index(drop=True)
    )
    # Optuna 동안 holdout은 절대 사용 안 함
    df, _df_holdout_ = _train_holdout_split(df_all)

    # ───────── 하이퍼파라미터 샘플 ─────────
    # (WaveAtt 관련 및 시퀀스/학습 파라미터만 사용)
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

    # ★ 튜닝 대상: margin (거부 마진)
    reject_margin  = trial.suggest_float("reject_margin", *SEARCH_SPACE_SPEC["reject_margin"])

    # coverage-penalized 점수
    LAMBDA = 0.2
    TAU    = 0.5

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

        # ── Train을 base/meta로 쪼개 meta 구간에서 보정자 학습
        split_pt = int(len(train_df) * 0.8)
        base_df, meta_df = train_df.iloc[:split_pt], train_df.iloc[split_pt:]

        # ── 전체 피처 사용 (date/label_up 제외)
        feat_cols = _get_feature_cols(train_df)

        # 스케일링 및 로더 구성
        b_all, m_all, v_all, _ = make_scaler_and_transform(base_df, meta_df, val_df, feat_cols)
        yb, ym, yv = base_df["label_up"].values, meta_df["label_up"].values, val_df["label_up"].values

        lb = DataLoader(SequenceDataset(b_all,yb,seq_len,False), batch_size=64, shuffle=False, drop_last=True)
        lm = DataLoader(SequenceDataset(m_all,ym,seq_len,False), batch_size=64, shuffle=False, drop_last=False)
        lv = DataLoader(SequenceDataset(v_all,yv,seq_len,False), batch_size=64, shuffle=False, drop_last=False)

        # ── WaveAtt 단일 모델 학습
        weight = compute_class_weight(yb)
        trans, _ = train_neural_model(
            WaveAttTransformerClassifier(
                input_size=b_all.shape[1],
                d_model=d_model, nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=transf_dropout,
                n_scales=n_scales, dtw_gamma=0.1
            ),
            lb, lm, epochs=transf_epochs, lr=transf_lr, weight=weight
        )

        # ── meta 구간에서 확률 보정자 학습
        res_lm = evaluate_model(trans, lm)
        if (res_lm is None) or (res_lm[1] is None):
            scores.append(0.5); continue
        p_meta = res_lm[1]
        mt = meta_df["label_up"].values[seq_len:]
        if len(mt)==0 or len(p_meta)==0:
            scores.append(0.5); continue
        Lm = min(len(mt), len(p_meta))
        calibr = ProbabilityCalibrator(dual_average=True).fit(p_meta[:Lm], mt[:Lm])

        # ── 외부 Val 예측 → 보정 → 마진 지표
        res_lv = evaluate_model(trans, lv)
        if (res_lv is None) or (res_lv[1] is None):
            scores.append(0.5); continue
        p_val = res_lv[1]
        vt    = val_df["label_up"].values[seq_len:]
        if len(vt)==0 or len(p_val)==0:
            scores.append(0.5); continue
        L2 = min(len(vt), len(p_val))
        vt, p_val = vt[:L2], p_val[:L2]
        p_val_c   = calibr.transform(p_val) if getattr(calibr, "trained", False) else p_val

        met = bin_metrics_with_margin(vt, p_val_c, margin=reject_margin)
        score_auc = met["auc_bin"] if np.isfinite(met["auc_bin"]) else 0.5
        score = score_auc - LAMBDA * max(0.0, TAU - met["coverage"])
        if met["coverage"] <= 1e-6:
            score = 0.45

        scores.append(float(score))
        trial.report(scores[-1], fold)
        logging.info(f"[fold{fold}] margin={reject_margin:.3f} auc_bin={met['auc_bin']} coverage={met['coverage']:.3f} score={scores[-1]:.4f}")
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
    study.optimize(objective, n_trials=800, timeout=3600)  # ▼ 5000 → 800
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
    df_all = (
        pd.read_csv(enhanced_csv, parse_dates=["date"])
          .sort_values("date")
          .reset_index(drop=True)
    )
    # train/holdout 분리
    df, _ = _train_holdout_split(df_all)
    best = study.best_trial.params

    # train 내부 마지막 분할 사용
    n = len(df)
    initial_train = best.get("initial_train", 500 if n>800 else n//2)
    val_size      = best.get("val_size", 120)
    splits = []
    te = initial_train
    while te + val_size <= n:
        splits.append((list(range(te)), list(range(te, te + val_size))))
        te += val_size
    tr_idx, va_idx = splits[-1]
    train_df = df.iloc[tr_idx].reset_index(drop=True)
    val_df   = df.iloc[va_idx].reset_index(drop=True)

    # 전체 피처
    feat_cols = _get_feature_cols(train_df)

    # 하이퍼파라미터 복원
    d_model, nhead = _get_best_dmodel_nhead(best)
    dim_ff = _get_best_dimfeedforward(best, d_model)
    best_full = dict(best)
    best_full["d_model_nhead"]  = (d_model, nhead)
    best_full["dim_feedforward"] = dim_ff
    best_full["n_scales"] = int(best.get("n_scales", 3))

    # meta 보정 학습용 분할
    split_pt = int(len(train_df)*0.8)
    base_df, meta_df = train_df.iloc[:split_pt], train_df.iloc[split_pt:]

    b_all, m_all, v_all, scaler = make_scaler_and_transform(base_df, meta_df, val_df, feat_cols)
    yb, ym, yv = base_df["label_up"].values, meta_df["label_up"].values, val_df["label_up"].values

    lb = DataLoader(SequenceDataset(b_all,yb,best_full["seq_len"],False), batch_size=64, shuffle=False, drop_last=True)
    lm = DataLoader(SequenceDataset(m_all,ym,best_full["seq_len"],False), batch_size=64, shuffle=False, drop_last=False)
    lv = DataLoader(SequenceDataset(v_all,yv,best_full["seq_len"],False), batch_size=64, shuffle=False, drop_last=False)

    weight = compute_class_weight(yb)

    model = WaveAttTransformerClassifier(
        input_size=b_all.shape[1],
        d_model=d_model, nhead=nhead,
        num_layers=int(best.get("num_layers", 3)),
        dim_feedforward=dim_ff,
        dropout=float(best.get("transf_dropout", 0.2)),
        n_scales=best_full["n_scales"], dtw_gamma=0.1
    )
    model,_ = train_neural_model(
        model, lb, lm,
        epochs=int(best.get("transf_epochs", 30)),
        lr=float(best.get("transf_lr", 1e-3)),
        weight=weight
    )

    # 보정자 학습 (meta 구간)
    calibr = ProbabilityCalibrator(dual_average=True); calibr.trained = False
    if len(meta_df) >= best_full["seq_len"] + 5:
        res_meta = evaluate_model(model, lm)
        if res_meta and (res_meta[1] is not None):
            p_meta = res_meta[1]
            mt = meta_df["label_up"].values[best_full["seq_len"]:]
            if len(mt)>0 and len(p_meta)>0:
                Lm = min(len(mt), len(p_meta))
                calibr = ProbabilityCalibrator(dual_average=True).fit(p_meta[:Lm], mt[:Lm])

    # 저장 (WaveAtt 단일 포맷)
    torch.save(model.state_dict(), final_model_dir/"transformer_final.pt")
    joblib.dump({
        "scaler": scaler,
        "feat_cols": feat_cols,
        "calibrator": calibr,
        "params": best_full
    }, final_model_dir/"preproc_and_calibrator.pkl")

    with open(final_model_dir/"metrics.json","w") as f:
        json.dump({"note": "Final metrics should be evaluated on HOLDOUT. See metrics_holdout.json."}, f, indent=2)

    return {
        "params": best_full,
        "saved_dir": str(final_model_dir)
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
# ---------- Holdout 평가 (margin-aware AUC 리포트) ----------
def evaluate_on_holdout(saved):
    # 전체 데이터 로드 & 홀드아웃 분리
    df_all = (
        pd.read_csv(enhanced_csv, parse_dates=["date"])
          .sort_values("date").reset_index(drop=True)
    )
    _, df_holdout = _train_holdout_split(df_all)

    # ── 단일 번들 로드 (스케일러/특징/보정자/파라미터)
    bundle_path = Path(saved["saved_dir"]) / "preproc_and_calibrator.pkl"  # 단수형으로 변경
    bundle = joblib.load(bundle_path)
    scaler    = bundle["scaler"]
    feat_cols = bundle["feat_cols"]
    calibr    = bundle["calibrator"]
    best_full = bundle["params"]

    # ── 모델 로드 (WaveAtt 단일)
    trans = WaveAttTransformerClassifier(
        input_size=len(feat_cols),
        d_model=best_full["d_model_nhead"][0],
        nhead=best_full["d_model_nhead"][1],
        num_layers=int(best_full.get("num_layers", 3)),
        dim_feedforward=int(best_full["dim_feedforward"]),
        dropout=float(best_full.get("transf_dropout", 0.2)),
        n_scales=int(best_full.get("n_scales", 3)),
        dtw_gamma=0.1
    )
    state_path = Path(saved["saved_dir"]) / "transformer_final.pt"  # 단일 파일명
    trans.load_state_dict(torch.load(state_path, map_location=DEVICE))
    trans.to(DEVICE).eval()

    # ── 입력 구성
    dfh = df_holdout.reset_index(drop=True)
    if len(feat_cols) > 0:
        X = scaler.transform(dfh[feat_cols].fillna(0).values.astype(float))
    else:
        X = np.zeros((len(dfh), 0), dtype=float)
    y = dfh["label_up"].values

    seq_len = int(best_full["seq_len"])
    dl = DataLoader(SequenceDataset(X, y, seq_len, False), batch_size=128, shuffle=False)

    # ── 예측
    _, p, _ = evaluate_model(trans, dl)
    L = min(len(p), len(y) - seq_len)
    vt = y[seq_len:seq_len + L]
    if len(vt) == 0:
        logging.warning("[HOLDOUT] not enough length for evaluation; returning neutral metrics.")
        return {"auc": 0.5, "accuracy": 0.0, "f1": 0.0}

    # ── 보정(캘리브레이션)
    p_c = p[:L]
    if getattr(calibr, "trained", False):
        p_c = calibr.transform(p_c)

    # ── 전체 지표
    auc = roc_auc_score(vt, p_c) if len(np.unique(vt)) > 1 else 0.5
    pred = (p_c > 0.5).astype(int)
    acc = accuracy_score(vt, pred) if len(vt) == len(pred) else 0.0
    f1  = f1_score(vt, pred) if len(np.unique(vt)) > 1 and len(vt) == len(pred) else 0.0

    # ── 마진 기반 지표
    reject_margin = float(best_full.get("reject_margin", 0.0))
    m = bin_metrics_with_margin(vt, p_c, margin=reject_margin)

    logging.info(
        f"[HOLDOUT] overall auc={auc:.4f}, acc={acc:.4f} | "
        f"margin={reject_margin:.3f} coverage={m['coverage']:.3f} "
        f"acc_bin={m['acc_bin'] if np.isfinite(m['acc_bin']) else float('nan'):.4f} "
        f"auc_bin={m['auc_bin'] if np.isfinite(m['auc_bin']) else float('nan'):.4f}"
    )

    out = {
        "auc_overall": auc,
        "acc_overall": acc,
        "f1_overall": f1,
        "reject_margin": reject_margin,
        "coverage": m["coverage"],
        "acc_bin": m["acc_bin"],
        "auc_bin": m["auc_bin"],
    }
    return out



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
