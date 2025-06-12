import sys, json, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# 경로 설정
root = Path(__file__).resolve().parent.parent
modeling_dir = root / "training" / "modeling"
sys.path.append(str(modeling_dir))

from model import FusionLSTMTransformerWithAttn
from prepare_data import prepare_data  # 사용자 정의 데이터 준비 함수

# ——————————————————————————————
# 평가 함수: ROC‑AUC, Accuracy, F1, 예측 확률 반환
def evaluate(model, X, y, device):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=64, shuffle=False)
    y_true, preds, probs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits, _ = model(xb)
            p = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            probs.extend(p)
            preds.extend((p > 0.5).astype(int))
            y_true.extend(yb.numpy())
    return (
        roc_auc_score(y_true, probs),
        accuracy_score(y_true, preds),
        f1_score(y_true, preds),
        np.array(probs)
    )

# ——————————————————————————————
# 백테스트: 상승 예측 시 종가 비중=1 진입, 추후 수익률 계산
def backtest(pred_probs, prices, threshold=0.5):
    returns = [(prices[i+1] - prices[i])/prices[i] if pred_probs[i] > threshold else 0
               for i in range(len(pred_probs)-1)]
    cumulative = np.cumprod([1 + r for r in returns])
    sharpe = np.mean(returns) / (np.std(returns)+1e-8) * np.sqrt(252)
    return cumulative, sharpe

# ——————————————————————————————
# 모델별 성능 비교
def compare_models(model_info, kospi_prices):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    for label, info in model_info.items():
        best = json.load(open(info["param_path"], 'r'))
        feature_cols = info["feature_cols"]

        # 모델 로드
        model = FusionLSTMTransformerWithAttn(
            input_dim=len(feature_cols),
            hidden_dim=best["hidden_dim"],
            num_layers=best["num_layers"],
            dropout=best["dropout"],
            nhead=best["nhead"]
        ).to(device)
        model.load_state_dict(torch.load(info["model_path"], map_location=device))

        # 데이터 준비 (prepare_data 활용)
        X, y = prepare_data(best["seq_len"], feature_cols)
        split = int(len(X)*0.8)
        X_test, y_test = X[split:], y[split:]
        test_prices = kospi_prices[split + best["seq_len"]:]

        # 평가 및 백테스트
        auc, acc, f1, probs = evaluate(model, X_test, y_test, device)
        cumulative, sharpe = backtest(probs, test_prices)

        results[label] = {
            "ROC-AUC": auc,
            "Accuracy": acc,
            "F1-score": f1,
            "Sharpe": sharpe,
            "Cumulative": cumulative
        }

    return results

# ——————————————————————————————
# 실행
if __name__ == "__main__":
    model_info = {
        "TechInd": {
            "param_path": root/"training"/"modeling"/"best_params_techind.json",
            "model_path": root/"models"/"fusion_model_techind_0612.pth",
            "feature_cols": [
                'kospi', 'ret_lag1', 'usd_krw', 'cny_krw', 'eur_krw',
                'jpy_krw', 'usd_jpy', 'gold_krw', 'wti_krw',
                'sent_pos', 'sent_neu', 'sent_neg',
                'rsi_14', 'macd', 'boll_upper', 'boll_lower', 'volatility_std10'
            ]
        }
    }

    # KOSPI 종가 로드 (백테스트 용)
    kospi_df = pd.read_csv(root/"data"/"raw"/"merged_macro_dataset.csv", parse_dates=["date"])
    kospi_df["kospi"] = kospi_df["kospi"].str.replace(",", "").astype(float)
    kospi_prices = kospi_df.dropna(subset=["kospi"])["kospi"].values

    results = compare_models(model_info, kospi_prices)

    # 출력
    for label, res in results.items():
        print(f"\n=== {label} ===")
        print(f"ROC-AUC: {res['ROC-AUC']:.4f}")
        print(f"Accuracy: {res['Accuracy']:.4f}")
        print(f"F1-score: {res['F1-score']:.4f}")
        print(f"Sharpe: {res['Sharpe']:.4f}")

        plt.figure(figsize=(8,4))
        plt.plot(res["Cumulative"], label=label)
        plt.title(f"{label} Cumulative Return")
        plt.xlabel("Days")
        plt.ylabel("Cumulative Return")
        plt.legend(); plt.tight_layout(); plt.show()