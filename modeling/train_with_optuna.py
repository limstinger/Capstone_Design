import numpy as np
import pandas as pd
import ta
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import optuna
import json


# —————————————————————
# 1. 이진 분류 라벨링 함수
# —————————————————————
def label_binary(returns: pd.Series) -> np.ndarray:
    return (returns > 0).astype(int).to_numpy()

# 프로젝트 루트(=Capstone_Design) 경로
root = Path(__file__).resolve().parents[2]

# —————————————————————
# 2. prepare_data 함수 (스케일링 없이 raw 값만 반환)
# —————————————————————
def prepare_data(seq_len: int, feature_cols: list) -> tuple[np.ndarray, np.ndarray]:
    """
    - merged_macro_dataset.csv와 daily_kfdeberta_sentiment.csv를 병합
    - 문자열로 된 숫자 컬럼(예: "1,234.56")에서 따옴표/쉼표 제거 후 float으로 변환
    - sentiment 컬럼이 결측이면 0으로 채우기
    - 종가 수익률 반환(ret)과 하루 전 수익률(ret_lag1) 생성
    - label_binary()로 상승(1)/하락(0) 라벨링
    - dropna() 후, 시퀀스 길이(seq_len) 단위로 X, y 생성 (스케일 없이 raw 값)
    """
    # (1) CSV 로드
    df = pd.read_csv(root / 'data' / 'raw' / 'merged_macro_dataset.csv', parse_dates=['date'])
    df_sent = pd.read_csv(root / 'data' / 'raw' / 'daily_kfdeberta_sentiment.csv', parse_dates=['date'])
    df = df.merge(df_sent, on='date', how='left')

    # (2) 문자열 숫자 컬럼 → float (따옴표/쉼표 제거)
    for col in ['kospi', 'usd_krw', 'eur_krw', 'cny_krw', 'jpy_krw',
                'usd_jpy', 'gold_krw', 'wti_krw', 'kosdaq']:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in merged_macro_dataset.csv")
        # 문자열로 된 숫자 예: '"1,234.56"' → '1234.56' → float
        df[col] = (df[col]
                   .astype(str)
                   .str.replace('"', '', regex=False)
                   .str.replace(',', '', regex=False)
                   .astype(float))

    # (3) Sentiment 결측치 0으로 대체
    df[['sent_pos', 'sent_neu', 'sent_neg']] = df[['sent_pos', 'sent_neu', 'sent_neg']].fillna(0)

    # (4) 수익률 계산 및 라벨링
    df['ret'] = df['kospi'].pct_change()
    df['ret_lag1'] = df['ret'].shift(1)
    df['target'] = label_binary(df['ret'])

    df['rsi_14'] = ta.momentum.RSIIndicator(close=df['kospi'], window=14).rsi()
    df['macd'] = ta.trend.MACD(close=df['kospi'], window_slow=26, window_fast=12).macd()
    df['boll_upper'] = ta.volatility.BollingerBands(close=df['kospi']).bollinger_hband()
    df['boll_lower'] = ta.volatility.BollingerBands(close=df['kospi']).bollinger_lband()
    df['volatility_std10'] = df['kospi'].rolling(10).std()

    df.dropna(inplace=True)

    # (5) 최소 데이터 수 검사
    if len(df) < seq_len + 1:
        raise ValueError("데이터가 너무 작아서 시퀀스를 구성할 수 없습니다. (len < seq_len+1)")

    # (6) 시퀀스 구성 (스케일링은 하지 않음)
    X, y = [], []
    vals = df[feature_cols].values   # (N, num_features)
    labs = df['target'].values       # (N,)
    for i in range(len(df) - seq_len):
        X.append(vals[i : i + seq_len])      # shape = (seq_len, num_features)
        y.append(labs[i + seq_len])          # 다음 시점의 라벨

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

# —————————————————————
# 3. 모델 정의: Fusion LSTM-Transformer
# —————————————————————
class TransformerLayerWithAttn(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model,
                                               num_heads=nhead,
                                               dropout=dropout,
                                               batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # src: (batch, seq_len, d_model)
        attn_output, attn_weights = self.self_attn(src, src, src, need_weights=True)
        # attn_weights: (batch, nhead, seq_len, seq_len)

        src2 = self.norm1(src + self.dropout1(attn_output))
        ff = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src3 = self.norm2(src2 + self.dropout2(ff))
        return src3, attn_weights


# ② FusionLSTMTransformerWithAttn: LSTM → Transformer(어텐션) → Classifier
class FusionLSTMTransformerWithAttn(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, nhead):
        super().__init__()
        self.lstm = nn.LSTM(input_dim,
                            hidden_dim,
                            num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)
        d_model = hidden_dim * 2
        self.transformer_layer = TransformerLayerWithAttn(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)   # 이진 분류 기준
        )

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        h, _ = self.lstm(x)                     # (batch, seq_len, hidden_dim*2)
        out_t, attn_weights = self.transformer_layer(h)
        # out_t: (batch, seq_len, hidden_dim*2)
        # attn_weights: (batch, nhead, seq_len, seq_len)

        pooled = out_t.mean(dim=1)              # (batch, hidden_dim*2)
        logits = self.classifier(pooled)        # (batch, 2)
        return logits, attn_weights

# —————————————————————
# 4. Objective 함수 (Optuna + TimeSeriesSplit)
# —————————————————————
def objective(trial: optuna.trial.Trial) -> float:
    # 4-1) Hyperparameter 탐색 공간 정의
    seq_len    = trial.suggest_int('seq_len',     5, 20)            # 시퀀스 길이
    hidden_dim = trial.suggest_int('hidden_dim',  32, 128, step=32) # LSTM hidden 차원
    num_layers = trial.suggest_int('num_layers',   2,   6)          # LSTM 레이어 수
    dropout    = trial.suggest_float('dropout',   0.0,  0.5)        # Dropout 비율
    lr         = trial.suggest_float('lr',      1e-4,  1e-2, log=True)  # 학습률
    nhead      = trial.suggest_categorical('nhead',       [2, 4])    # Transformer 헤드 수
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])# 배치 크기
    n_epochs   = trial.suggest_int('n_epochs',     5, 20)           # 에폭 수

    # 4-2) 사용할 피처 컬럼 목록
    feature_cols = [
        'kospi', 'ret_lag1',
        'usd_krw', 'cny_krw', 'eur_krw', 'jpy_krw', 'usd_jpy', 'gold_krw', 'wti_krw',
        'sent_pos', 'sent_neu', 'sent_neg',
        'rsi_14', 'macd', 'boll_upper', 'boll_lower', 'volatility_std10'
    ]

    # 4-3) 전체 데이터를 로드 (raw 값)
    X_all, y_all = prepare_data(seq_len, feature_cols)
    #    X_all.shape = (n_samples, seq_len, num_features)
    #    y_all.shape = (n_samples,)

    # 4-4) TimeSeriesSplit 객체 생성 (3-fold)
    tscv = TimeSeriesSplit(n_splits=3)
    fold_metrics = []

    # 4-5) 각 Fold마다 “스케일링 → 학습 → 검증” 반복
    for train_idx, val_idx in tscv.split(X_all):
        X_tr_raw, X_va_raw = X_all[train_idx], X_all[val_idx]
        y_tr,     y_va     = y_all[train_idx], y_all[val_idx]

        # (A) Fold별 스케일러: train 데이터만 fit → train/val transform
        ns_tr, sl, nf = X_tr_raw.shape  # ns_tr=#train samples, sl=seq_len, nf=#features
        scaler = StandardScaler().fit(X_tr_raw.reshape(-1, nf))
        X_tr_scaled = scaler.transform(X_tr_raw.reshape(-1, nf)).reshape(ns_tr, sl, nf)

        ns_va, _, _  = X_va_raw.shape
        X_va_scaled = scaler.transform(X_va_raw.reshape(-1, nf)).reshape(ns_va, sl, nf)

        # (B) DataLoader 준비 (shuffle=False → 시계열 순서 유지)
        train_dataset = TensorDataset(torch.from_numpy(X_tr_scaled), torch.from_numpy(y_tr))
        val_dataset   = TensorDataset(torch.from_numpy(X_va_scaled), torch.from_numpy(y_va))
        train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

        # (C) 모델, 옵티마이저, 손실함수 초기화
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = FusionLSTMTransformerWithAttn(
            input_dim = X_tr_scaled.shape[2],
            hidden_dim= hidden_dim,
            num_layers= num_layers,
            dropout   = dropout,
            nhead     = nhead
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # (D) Fold별 학습 (n_epochs 동안)
        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits, _ = model(xb)          # (batch_size, 2)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(train_loader.dataset)
            # (선택) print(f"Fold {len(fold_metrics)+1} Epoch {epoch+1}/{n_epochs} Loss: {epoch_loss:.4f}")

        # (E) Fold별 검증
        model.eval()
        preds_probs, true_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits, _ = model(xb)                    # (batch_size, 2)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # 상승(1) 확률
                preds_probs.extend(probs)
                true_labels.extend(yb.numpy())

        fold_auc = roc_auc_score(true_labels, preds_probs)
        fold_metrics.append(fold_auc)

    # 4-6) 3개 Fold AUC 평균 반환
    return float(np.mean(fold_metrics))


# —————————————————————
# 5. 최적화 & 최종 학습
# —————————————————————
if __name__ == '__main__':
    # (1) GPU/CPU 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # (2) Optuna 스터디 생성 (maximize: ROC-AUC가 클수록 좋음)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=1000)

    print("Best params:", study.best_params)

    # (3) 최적 파라미터로 전체 데이터(80%→Train, 20%→Test) 재학습
    params = study.best_params
    feature_cols = [
        'kospi', 'ret_lag1',
        'usd_krw', 'cny_krw', 'eur_krw', 'jpy_krw', 'usd_jpy', 'gold_krw', 'wti_krw',
        'sent_pos', 'sent_neu', 'sent_neg',
        'rsi_14', 'macd', 'boll_upper', 'boll_lower', 'volatility_std10'
    ]

    # 3-1) 전체 raw 데이터 로드
    X_all, y_all = prepare_data(params['seq_len'], feature_cols)

    # 3-2) 80/20 split
    split = int(len(X_all) * 0.8)
    X_tr_raw, X_te_raw = X_all[:split], X_all[split:]
    y_tr, y_te         = y_all[:split],     y_all[split:]

    # 3-3) Train 데이터 스케일링 → 학습
    ns_tr, sl, nf = X_tr_raw.shape
    scaler = StandardScaler().fit(X_tr_raw.reshape(-1, nf))
    X_tr_scaled = scaler.transform(X_tr_raw.reshape(-1, nf)).reshape(ns_tr, sl, nf)

    ns_te, _, _   = X_te_raw.shape
    X_te_scaled   = scaler.transform(X_te_raw.reshape(-1, nf)).reshape(ns_te, sl, nf)

    train_dataset = TensorDataset(torch.from_numpy(X_tr_scaled), torch.from_numpy(y_tr))
    train_loader  = DataLoader(train_dataset,
                               batch_size=params.get('batch_size', 32),
                               shuffle=False)

    # 3-4) 모델 재초기화 → 학습
    model = FusionLSTMTransformerWithAttn(
        input_dim = X_tr_scaled.shape[2],
        hidden_dim= params['hidden_dim'],
        num_layers= params['num_layers'],
        dropout   = params['dropout'],
        nhead     = params['nhead']
    ).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(params.get('n_epochs', 10)):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    # (4) 최종 모델 저장
    (root / 'models').mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), root / 'models' / 'fusion_model_techind_0612.pth')
    print("✅ 최종 모델 (binary classifier) 학습 및 저장 완료")

    # (5) 최적 파라미터 JSON으로 저장
    with open(root / 'training' / 'modeling' / 'best_params_techind.json', 'w') as f:
        json.dump(study.best_params, f)
    print("✅ Best params saved to best_params.json")