import sys, json, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve

root = Path(__file__).resolve().parent.parent
modeling_dir = root/"training"/"modeling"
sys.path.append(str(modeling_dir))

from model import FusionLSTMTransformerWithAttn
from prepare_data import prepare_data

def evaluate_model(model, X, y, device):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=64)
    probs, preds = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits, _ = model(xb)
            p = torch.softmax(logits,1)[:,1].cpu().numpy()
            probs.extend(p)
            preds.extend((p>0.5).astype(int))
    return roc_auc_score(y, probs), accuracy_score(y, preds), f1_score(y, preds), np.array(probs), y

def simple_backtest(prob, prices, threshold=0.5):
    returns = [(prices[i+1]-prices[i])/prices[i] if prob[i]>threshold else 0 for i in range(len(prob)-1)]
    cum = np.cumprod([1+r for r in returns])
    sharpe = np.mean(returns)/(np.std(returns)+1e-8)*np.sqrt(252)
    return cum, sharpe

def compare_and_plot(model_info, prices):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results, roc_data = {}, {}

    for label, info in model_info.items():
        best = json.load(open(info["param_path"]))
        feature_cols = info["feature_cols"]

        model = FusionLSTMTransformerWithAttn(
            input_dim=len(feature_cols),
            hidden_dim=best["hidden_dim"],
            num_layers=best["num_layers"],
            dropout=best["dropout"],
            nhead=best["nhead"]
        ).to(device)
        model.load_state_dict(torch.load(info["model_path"], map_location=device))

        X_all, y_all = prepare_data(best["seq_len"], feature_cols)
        split = int(len(X_all)*0.8)
        X_te, y_te = X_all[split:], y_all[split:]

        auc, acc, f1, probs, y_true = evaluate_model(model, X_te, y_te, device)
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_data[label] = (fpr, tpr, auc)

        cum, sharpe = simple_backtest(probs, prices[-len(probs):])
        results[label] = {"ROC-AUC": auc, "Accuracy": acc, "F1": f1, "Cumulative": cum}

    # ROC 곡선 비교
    plt.figure(figsize=(8,6))
    for label,(fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    plt.plot([0,1], [0,1], 'k--', label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid(); plt.tight_layout()
    plt.show()

    return results

if __name__=="__main__":
    kospi = pd.read_csv(root/"data"/"raw"/"merged_macro_dataset.csv", parse_dates=["date"])
    kospi["kospi"] = kospi["kospi"].str.replace(",", "").astype(float)
    kospi = kospi.dropna(subset=["kospi"])
    prices = kospi["kospi"].values

    model_info = {
        "Base": {
            "param_path": root/"training"/"modeling"/"best_params.json",
            "model_path": root/"models"/"fusion_model_2.pth",
            "feature_cols": [
                'kospi','ret_lag1',
                'usd_krw','cny_krw','eur_krw','jpy_krw','usd_jpy',
                'gold_krw','wti_krw',
                'sent_pos','sent_neu','sent_neg'
            ]
        },
        "TechInd": {
            "param_path": root/"training"/"modeling"/"best_params_techind.json",
            "model_path": root/"models"/"fusion_model_techind_0612.pth",
            "feature_cols": [
                'kospi','ret_lag1',
                'usd_krw','cny_krw','eur_krw','jpy_krw','usd_jpy',
                'gold_krw','wti_krw',
                'sent_pos','sent_neu','sent_neg',
                'rsi_14','macd','boll_upper','boll_lower','volatility_std10'
            ]
        }
    }

    results = compare_and_plot(model_info, prices)
    df = pd.DataFrame(results).T.round(4)
    print(df)

    df.drop(columns=["Cumulative"]).plot(kind="bar", figsize=(10,5), colormap="viridis", rot=0)
    plt.title("Performance Comparison")
    plt.ylabel("Score")
    plt.grid(axis="y"); plt.tight_layout()
    plt.show()