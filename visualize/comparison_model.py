import sys, json, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             precision_score, recall_score,
                             precision_recall_curve)
# -----------------------------
# 0. 경로 & 공통
# -----------------------------
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root / "training" / "modeling"))

from model import FusionLSTMTransformerWithAttn          # 기존 모델
from prepare_data import prepare_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_cols = [
    'kospi','ret_lag1','usd_krw','cny_krw','eur_krw','jpy_krw','usd_jpy',
    'gold_krw','wti_krw',
    'sent_pos','sent_neu','sent_neg',
    'rsi_14','macd','boll_upper','boll_lower','volatility_std10'
]

# -----------------------------
# 1. 동적 임계치 평가 함수
# -----------------------------
def evaluate_and_tune(model, X, y):
    """PR-curve에서 F1이 최대인 cut-off로 다시 지표 산출"""
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X),
                                      torch.from_numpy(y)),
                        batch_size=256)
    prob = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            out, _ = model(xb)
            prob.extend(torch.softmax(out,1)[:,1].cpu().numpy())
    prob = np.asarray(prob)

    # --- best-F1 임계치 산출 ---
    prec, rec, thr = precision_recall_curve(y, prob)
    f1s            = 2*prec*rec / (prec+rec+1e-8)
    best_idx       = np.argmax(f1s)
    best_thr       = thr[max(best_idx-1,0)]      # thr 길이 = prec-1
    pred           = (prob > best_thr).astype(int)

    return dict(
        auc   = roc_auc_score(y, prob),
        acc   = accuracy_score(y, pred),
        f1    = f1_score(y, pred),
        prec  = precision_score(y, pred),
        rec   = recall_score(y, pred),
        thr   = best_thr,
        prob  = prob,
        y_true= y
    )

def backtest(prob, price, thr):
    """오늘 prob>thr → 내일 종가 매수, 아니면 관망"""
    ret = [(price[i+1]-price[i])/price[i] if prob[i] > thr else 0
           for i in range(len(prob)-1)]
    ret = np.asarray(ret)
    return np.cumprod(1+ret), ret.mean()/(ret.std()+1e-8)*np.sqrt(252)

# -----------------------------
# 2. 비교할 모델 정보
# -----------------------------
model_info = {
    "Base": {  # 헤드라인+거시(기본)
        "param": root/"training"/"modeling"/"best_params.json",
        "weight":root/"models"/"fusion_model_2.pth",
        "cols": [
            'kospi','ret_lag1','usd_krw','cny_krw','eur_krw','jpy_krw','usd_jpy',
            'gold_krw','wti_krw','sent_pos','sent_neu','sent_neg'
        ]
    },
    "TechInd": {  # 기술지표 포함
        "param": root/"training"/"modeling"/"best_params_techind.json",
        "weight":root/"models"/"fusion_model_techind_0612.pth",
        "cols"  : feature_cols
    }
}

# -----------------------------
# 3. 실행
# -----------------------------
def main():
    # KOSPI 가격
    kospi = (pd.read_csv(root/"data"/"raw"/"merged_macro_dataset.csv",
                         parse_dates=['date'])
               .assign(kospi=lambda d: d['kospi'].astype(str)
                        .str.replace(',','').astype(float))
               .dropna(subset=['kospi']))
    price_arr = kospi['kospi'].values

    results, hists = {}, {}

    for tag, meta in model_info.items():
        best = json.load(open(meta["param"]))
        seq_len = best["seq_len"]

        # ----- 데이터 -----
        X, y = prepare_data(seq_len, meta["cols"])
        split = int(len(X)*0.8)
        X_te, y_te = X[split:], y[split:]

        # ----- 모델 -----
        model = FusionLSTMTransformerWithAttn(
            input_dim = len(meta["cols"]),
            hidden_dim= best["hidden_dim"],
            num_layers= best["num_layers"],
            dropout   = best["dropout"],
            nhead     = best["nhead"]
        ).to(device)
        model.load_state_dict(torch.load(meta["weight"], map_location=device))

        # ----- 평가 -----
        m = evaluate_and_tune(model, X_te, y_te)
        hist_prob = m["prob"];  hists[tag] = hist_prob

        # ----- 백테스트 -----
        cum, sharpe = backtest(hist_prob, price_arr[-len(hist_prob):], m["thr"])

        results[tag] = {
            "AUC":m["auc"], "Acc":m["acc"], "F1":m["f1"],
            "Prec":m["prec"], "Rec":m["rec"],
            "Sharpe": sharpe, "Cum": cum, "Thr": m["thr"]
        }

    # ----------------- 시각화 -----------------
    # (1) 확률 히스토그램
    plt.figure(figsize=(7,4))
    for tag,p in hists.items():
        plt.hist(p, bins=40, alpha=.5, density=True, label=tag)
    plt.title("Predicted-probability distribution"); plt.legend(); plt.grid()
    plt.tight_layout(); plt.show()

    # (2) 누적 수익
    plt.figure(figsize=(8,4))
    for tag in results:
        plt.plot(results[tag]["Cum"], label=f"{tag} (Sharpe {results[tag]['Sharpe']:.2f})")
    plt.title("Cumulative Return (best-F1 threshold)")
    plt.grid(); plt.legend(); plt.tight_layout(); plt.show()

    # (3) 성능 바 차트
    df = (pd.DataFrame(results).T
            [["AUC","Acc","F1","Prec","Rec","Sharpe"]].round(3))
    print(df, "\n")
    df.plot(kind="bar", rot=0, colormap="viridis", figsize=(10,5))
    plt.title("Performance metrics (dynamic threshold)"); plt.grid(axis="y")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()