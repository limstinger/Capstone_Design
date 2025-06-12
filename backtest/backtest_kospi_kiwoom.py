import os, sys, json, requests
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

# 프로젝트 구조 기준 경로
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root/"training"/"modeling"))
from model import FusionLSTMTransformerWithAttn

load_dotenv()
APP_KEY    = os.getenv("KIWOOM_KEY")
APP_SECRET = os.getenv("KIWOOM_SECRET")

params_path = root/"training"/"modeling"/"best_params_techind.json"
with open(params_path, 'r') as f:
    best = json.load(f)

def get_access_token():
    """토큰 발급 (모의 투자)"""
    url = "https://mockapi.kiwoom.com/oauth2/token"
    headers = {"Content-Type": "application/json;charset=UTF-8"}
    body = {
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "secretkey": APP_SECRET
    }
    r = requests.post(url, headers=headers, json=body)
    print("🔁 토큰 요청:", r.status_code, r.text)
    data = r.json()
    token = data.get("token") or data.get("access_token")
    if not token:
        raise RuntimeError(f"토큰 발급 실패: {data}")
    return token

def get_kospi_daily(token, cont_yn="N", next_key="", base_dt=None):
    url = "https://mockapi.kiwoom.com/api/dostk/chart"
    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "authorization": f"Bearer {token}",
        "cont-yn": cont_yn,
        "next-key": next_key,
        "api-id": "ka20006"
    }
    body = {
        "inds_cd": "001",  # 종합 KOSPI
        "base_dt": base_dt or datetime.today().strftime("%Y%m%d")
    }
    resp = requests.post(url, headers=headers, json=body)
    j = resp.json()
    if j.get("return_code") != 0:
        raise RuntimeError("업종일봉 조회 오류: " + j.get("return_msg", ""))
    
    df = pd.DataFrame(j["inds_dt_pole_qry"])
    df["date"] = pd.to_datetime(df["dt"], format="%Y%m%d")
    df["close"] = df["cur_prc"].astype(float)
    # df["open"] = df["open_pric"].astype(float)
    # df["high"] = df["high_pric"].astype(float)
    # df["low"]  = df["low_pric"].astype(float)

    return df[["date", "close"]].sort_values("date").reset_index(drop=True)

def make_signals(df, model, scaler, seq_len):
    df = df.copy().dropna().reset_index(drop=True)
    df["signal"] = 0
    for i in range(len(df)-seq_len):
        seq = df["close"].iloc[i:i+seq_len].values.reshape(-1,1)
        seq_scaled = scaler.transform(seq)
        x = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(x)
            df.at[i+seq_len, "signal"] = int(torch.argmax(logits,1).item())
    return df

# ➎ 백테스트
def backtest(df):
    ret = df["close"].pct_change().shift(-1).fillna(0)
    strat = ret.values * df["signal"].values
    cum_s = (1+strat).cumprod()-1
    cum_b = (1+ret).cumprod()-1
    sharpe = np.mean(strat)/np.std(strat)*np.sqrt(252) if np.std(strat)!=0 else 0
    mdd = (cum_s/np.maximum.accumulate(cum_s)-1).min()
    return cum_s, cum_b, sharpe, mdd

def main():
    token = get_access_token()
    df = get_kospi_daily(token)

    # 모델 초기화 (best_params.json 기준)
    model = FusionLSTMTransformerWithAttn(
        input_dim=1, 
        hidden_dim=best['hidden_dim'],
        num_layers=best['num_layers'],
        dropout=best['dropout'],
        nhead=best['nhead']
    )
    model.load_state_dict(torch.load(root/"models"/"fusion_model_techind_0612.pth", map_location="cpu"))
    model.eval()

    scaler = StandardScaler().fit(df["close"].pct_change().dropna().values.reshape(-1,1))
    seq_len = best['seq_len']
    df2 = make_signals(df, model, scaler, seq_len)

    cum_s, cum_b, sharpe, mdd = backtest(df2)
    print(f"📊 샤프: {sharpe:.3f}, MDD: {mdd:.2%}")

    plt.figure(figsize=(10,5))
    plt.plot(df2["date"].iloc[:-1], cum_s, label="Strategy")
    plt.plot(df2["date"].iloc[:-1], cum_b, label="Benchmark")
    plt.legend(); plt.xticks(rotation=45); plt.tight_layout(); plt.show()

if __name__=="__main__":
    main()