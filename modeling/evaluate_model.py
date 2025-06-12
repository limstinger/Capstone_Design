import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

# prepare_data 함수는 “스케일 없이 raw X, y”를 반환하도록 정의되어 있어야 합니다.
from prepare_data import prepare_data
from model import FusionLSTMTransformer

# 프로젝트 최상위 디렉토리 (evaluate_model.py가 있는 위치에서 두 단계 위)
root = Path(__file__).resolve().parents[2]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Optuna가 저장한 최적 파라미터 불러오기
    best_path = root / "training" / "modeling" / "best_params.json"
    with open(best_path, "r") as f:
        best = json.load(f)

    # 2) Feature 컬럼 리스트 (학습 때와 동일해야 합니다)
    feature_cols = [
        'kospi','ret_lag1',
        'usd_krw','cny_krw','eur_krw','jpy_krw','usd_jpy','gold_krw','wti_krw',
        'sent_pos','sent_neu','sent_neg'
    ]
    seq_len = best['seq_len']

    # 3) 전체 데이터(raw X, y) 불러오기
    X_all, y_all = prepare_data(seq_len, feature_cols)
    n_samples = X_all.shape[0]

    # 4) 앞 80%를 train, 뒤 20%를 validation/test로 분할
    split_idx = int(n_samples * 0.8)
    X_tr_raw, X_te_raw = X_all[:split_idx], X_all[split_idx:]
    y_tr,     y_te     = y_all[:split_idx], y_all[split_idx:]

    # 5) Train 구간에서만 StandardScaler를 fit → train/test 모두 transform
    ns_tr, sl, nf = X_tr_raw.shape
    scaler = StandardScaler().fit(X_tr_raw.reshape(-1, nf))
    X_tr_scaled = scaler.transform(X_tr_raw.reshape(-1, nf)).reshape(ns_tr, sl, nf)

    ns_te = X_te_raw.shape[0]
    X_te_scaled = scaler.transform(X_te_raw.reshape(-1, nf)).reshape(ns_te, sl, nf)

    # 6) validation(Test) 데이터셋 및 DataLoader 준비
    test_dataset = TensorDataset(torch.from_numpy(X_te_scaled), torch.from_numpy(y_te))
    test_loader  = DataLoader(test_dataset, batch_size=best.get('batch_size', 32), shuffle=False)

    # 7) 모델 생성 및 학습된 weight 로드
    model = FusionLSTMTransformer(
        input_dim = X_tr_scaled.shape[2],
        hidden_dim= best['hidden_dim'],
        num_layers= best['num_layers'],
        dropout   = best['dropout'],
        nhead     = best['nhead']
    ).to(device)

    model_file = root / "models" / "fusion_model.pth"
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    # 8) Test셋으로 예측 수행
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # “상승” 클래스 확률
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 9) 성능 지표 계산 및 출력
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    print("===== Test Set Performance =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC : {auc:.4f}\n")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    # 10) 혼동 행렬 시각화
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.title("Confusion Matrix (Test Set)")
    plt.colorbar()
    plt.xticks([0, 1], ["Down (0)", "Up (1)"])
    plt.yticks([0, 1], ["Down (0)", "Up (1)"])
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # 셀 값 표시
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='red')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    