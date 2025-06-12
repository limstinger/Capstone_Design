import sys, json, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             precision_score, recall_score,
                             precision_recall_curve, roc_curve)

# --------------------------------------------------
# 0. 경로·공통
# --------------------------------------------------
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root / "training" / "modeling"))

from model import FusionLSTMTransformerWithAttn, FusionLSTMPerformer
from prepare_data import prepare_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_cols = [
    'kospi','ret_lag1','usd_krw','cny_krw','eur_krw','jpy_krw','usd_jpy',
    'gold_krw','wti_krw',
    'sent_pos','sent_neu','sent_neg',
    'rsi_14','macd','boll_upper','boll_lower','volatility_std10'
]

model_info = {
    "Transformer": {
        "cls"       : FusionLSTMTransformerWithAttn,
        "param_json": root / "training" / "modeling" / "best_params_techind.json",
        "weight"    : root / "models" / "fusion_model_techind_0612.pth"
    },
    "Performer"  : {
        "cls"       : FusionLSTMPerformer,
        "param_json": root / "training" / "modeling" / "best_params_lineared.json",
        "weight"    : root / "models" / "fusion_model_lineared_0613.pth"
    }
}

# --------------------------------------------------
# 1. 유틸
# --------------------------------------------------
def evaluate_and_tune(model, X, y):
    """F1이 최대가 되는 cut-off를 탐색해 모든 지표를 계산"""
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X),
                                      torch.from_numpy(y)),
                        batch_size=256)
    prob = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            out, _ = model(xb)
            prob.extend(torch.softmax(out, 1)[:,1].cpu().numpy())
    prob = np.asarray(prob)

    # ---- 임계치 탐색 (PR-curve) ----
    prec, rec, thr = precision_recall_curve(y, prob)
    f1s            = 2*prec*rec/(prec+rec+1e-8)
    best_idx       = np.argmax(f1s)
    best_thr       = thr[max(best_idx-1,0)]        # thr 길이는 prec-1
    pred           = (prob > best_thr).astype(int)

    return {
        "auc"  : roc_auc_score(y, prob),
        "acc"  : accuracy_score(y, pred),
        "f1"   : f1_score(y, pred),
        "prec" : precision_score(y, pred),
        "rec"  : recall_score(y, pred),
        "prob" : prob,
        "y"    : y,
        "thr"  : best_thr
    }

def backtest(prob, price, thr):
    """단순 다음-날 전략·Sharpe"""
    ret = [(price[i+1]-price[i])/price[i] if prob[i] > thr else 0
           for i in range(len(prob)-1)]
    ret = np.array(ret)
    return np.cumprod(1+ret), ret.mean()/(ret.std()+1e-8)*np.sqrt(252)

# --------------------------------------------------
# 2. 평가 & 비교
# --------------------------------------------------
scores, roc_curves, hist_data = {}, {}, {}

for tag, info in model_info.items():
    best    = json.load(open(info["param_json"]))
    seq_len = best["seq_len"]

    # ----- 데이터 분할 -----
    X, y      = prepare_data(seq_len, feature_cols)
    cut       = int(len(X)*0.8)
    X_te, y_te = X[cut:], y[cut:]

    # ----- 모델 로드 -----
    kwargs = dict(input_dim=len(feature_cols),
                  hidden_dim=best["hidden_dim"],
                  num_layers=best["num_layers"],
                  dropout   =best["dropout"])
    if tag == "Performer":
        kwargs.update(heads=best.get("heads", best.get("nhead",4)),
                      depth=best.get("depth",2),
                      max_seq_len=seq_len)
    else:
        kwargs.update(nhead=best["nhead"])

    model = info["cls"](**kwargs).to(device)
    model.load_state_dict(torch.load(info["weight"], map_location=device))

    # ----- 평가 -----
    res = evaluate_and_tune(model, X_te, y_te)
    fpr, tpr, _ = roc_curve(res["y"], res["prob"])
    roc_curves[tag] = (fpr, tpr, res["auc"])
    hist_data[tag]  = res["prob"]

    # ----- 백테스트 -----
    kospi = (pd.read_csv(root/"data"/"raw"/"merged_macro_dataset.csv",
                         parse_dates=["date"])
               .assign(kospi=lambda d: d["kospi"].astype(str)
                        .str.replace(",","").astype(float))
               .dropna(subset=["kospi"]))
    cum, sharpe = backtest(res["prob"],
                           kospi["kospi"].values[-len(res["prob"]):],
                           res["thr"])

    scores[tag] = {"ROC-AUC":res["auc"], "Accuracy":res["acc"],
                   "F1":res["f1"], "Prec":res["prec"], "Recall":res["rec"],
                   "Sharpe":sharpe, "CumRet":cum, "BestThr":res["thr"]}

# --------------------------------------------------
# 3. 시각화
# --------------------------------------------------
# 3-1 ROC
plt.figure(figsize=(7,6))
for tag,(fpr,tpr,auc) in roc_curves.items():
    plt.plot(fpr, tpr, label=f"{tag} (AUC={auc:.3f})")
plt.plot([0,1],[0,1],'k--'); plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curve"); plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

# 3-2 확률 분포
plt.figure(figsize=(7,4))
for tag,p in hist_data.items():
    plt.hist(p, bins=40, alpha=.5, label=tag, density=True)
plt.title("Predicted-Probability Distribution"); plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

# 3-3 누적 수익
plt.figure(figsize=(8,4))
for tag in scores:
    plt.plot(scores[tag]["CumRet"], label=f"{tag} (sharpe {scores[tag]['Sharpe']:.2f})")
plt.title("Cumulative Return (next-day strategy)")
plt.ylabel("×"); plt.grid(); plt.legend(); plt.tight_layout(); plt.show()

# 3-4 표 & 막대
df = (pd.DataFrame(scores)
        .T[["ROC-AUC","Accuracy","F1","Prec","Recall","Sharpe","BestThr"]]
        .round(3))
print(df,"\n")
df.drop(columns=["BestThr"]).plot(kind="bar", rot=0, colormap="viridis",
                                  figsize=(10,5))
plt.title("Performance (dynamic threshold)"); plt.grid(axis="y"); plt.tight_layout(); plt.show()