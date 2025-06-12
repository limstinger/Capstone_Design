import sys
import pandas as pd
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 프로젝트 루트 설정
root = Path(__file__).resolve().parent.parent
modeling_dir  = root / "training" / "modeling"
sys.path.append(str(modeling_dir))

from model import FusionLSTMTransformerWithAttn
from prepare_data import prepare_data

def visualize_one_sample(seq_len, feature_cols, sample_index=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 최적 파라미터 불러오기
    with open(root / "training" / "modeling" / "best_params_techind.json", "r") as f:
        best = json.load(f)

    # 2) 데이터 준비
    X_all, y_all = prepare_data(best["seq_len"], feature_cols)
    split = int(len(X_all) * 0.8)
    X_val = X_all[split:]
    sample = X_val[sample_index : sample_index + 1]  # (1, seq_len, n_features)
    sample_tensor = torch.from_numpy(sample).float().to(device)

    # 3) 모델 로딩
    model = FusionLSTMTransformerWithAttn(
        input_dim=sample.shape[2],
        hidden_dim=best["hidden_dim"],
        num_layers=best["num_layers"],
        dropout=best["dropout"],
        nhead=best["nhead"],
    ).to(device)

    model_file = root / "models" / "fusion_model_techind_0612.pth"
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    # 4) Forward → Attention 추출
    with torch.no_grad():
        logits, attn_w = model(sample_tensor)
        # attn_w: (1, nhead, seq_len, seq_len) 예상
        print("attn_w.shape:", attn_w.shape)

    nhead = attn_w.shape[1]
    seq_len_actual = attn_w.shape[-1]

    # 5) 헤드 평균 어텐션 맵 생성
    attn_np = attn_w.squeeze(0).cpu().numpy()  # shape = (seq_len, seq_len)

    attn_avg = attn_w.mean(dim=1)[0]  # (seq_len, seq_len)
    importance = attn_avg.sum(dim=0).cpu().numpy()  # key 방향으로 누적
    top_k_indices = importance.argsort()[::-1][:3]  # 가장 주목한 시점 3개
    print("모델이 가장 집중한 시점:", top_k_indices)

    df_input = pd.DataFrame(sample.squeeze(0), columns=feature_cols)
    print(df_input.iloc[top_k_indices])

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        attn_np,
        cmap="viridis",
        xticklabels=list(range(attn_np.shape[1])),
        yticklabels=list(range(attn_np.shape[0])),
    )
    plt.xlabel("Key position (j)")
    plt.ylabel("Query position (i)")
    plt.title(f"Average Attention Map (Head-avg) (nhead={nhead})")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    feature_cols = [
        "kospi", "ret_lag1", "usd_krw", "cny_krw", "eur_krw", "jpy_krw",
        "usd_jpy", "gold_krw", "wti_krw",
        "sent_pos", "sent_neu", "sent_neg",
        'rsi_14', 'macd', 'boll_upper', 'boll_lower', 'volatility_std10'
    ]
    with open(root / "training" / "modeling" / "best_params_techind.json", "r") as f:
        best_params = json.load(f)

    visualize_one_sample(seq_len=best_params["seq_len"], feature_cols=feature_cols, sample_index=0)

    