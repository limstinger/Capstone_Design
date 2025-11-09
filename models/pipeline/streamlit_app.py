# streamlit_app.py
# -*- coding: utf-8 -*-
import os, re, math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# -------------------- 기본 설정 --------------------
pd.set_option("future.no_silent_downcasting", True)
st.set_page_config(page_title="📈 다음 날 예측", layout="centered")
st.title("📈 다음 날 주가 예측")
st.caption("※ 내부 모델 예측값입니다. 실제 투자판단에는 참고용으로만 사용하세요.")

# -------------------- 유틸 --------------------
def _strip_bom_zwsp(s: str) -> str:
    return re.sub(r"[\u200b\ufeff]", "", str(s)).strip()

def safe_parse_ts(x):
    """문자/NaN 안전 파싱 (YYYY-MM-DD 같은 일반 문자열 우선)"""
    if x is None or x == "" or (isinstance(x, float) and math.isnan(x)):
        return pd.NaT
    s = _strip_bom_zwsp(x)
    return pd.to_datetime(s, errors="coerce")

def feature_to_category(feat: str) -> str:
    if not feat:
        return "기타"
    f = str(feat).upper()
    CATEGORY_SPECS = [
        ("환율",      [r"(USD|EUR|JPY|CNY|CNH|GBP|AUD|CAD|CHF|SGD|HKD).*_KRW", r"^FX_", r"_FX$", r"KRW$", r"^USD_", r"^EUR_", r"^JPY_"]),
        ("금리/채권", [r"(KTB|TBOND|BOND|YIELD|IR_|KORIBOR|CD_|UST|US10Y|US2Y)"]),
        ("원자재",    [r"(WTI|BRENT|XAU|XAG|GOLD|SILVER|COPPER|NICKEL|ALUMINIUM|GAS|LNG|COAL|OIL)"]),
        ("주가지수",  [r"(KOSPI|KOSDAQ|^SPX$|S&P|^NDX$|IXIC|^DJI$|HSI|SSE|NIKKEI|NKY|DAX|FTSE)"]),
        ("변동성",    [r"(VIX|VKOSPI|VOL_)"]),
        ("거시지표",  [r"(CPI|PPI|PMI|ISM|UNEMP|GDP|RETAIL|HOUSING|INDUSTRIAL|M2|EXPORT|IMPORT)"]),
        ("뉴스심리",  [r"(SENTIMENT|BERT|ENTROPY|TOPIC|NEWS|POLARITY)"]),
    ]
    for cat, pats in CATEGORY_SPECS:
        if any(re.search(p, f) for p in pats):
            return cat
    return "기타"

# -------------------- 경로 해결 --------------------
def resolve_pred_csv() -> Path | None:
    """
    Streamlit Cloud에서 파일을 찾는 순서:
    1) ENV: PRED_CSV
    2) 앱 파일 옆(models/pipeline/next_day_predictions.csv)
    3) repo 루트의 data/predictions/next_day_predictions.csv
    4) CWD 기준 data/predictions/next_day_predictions.csv
    (없으면 None 반환)
    """
    env = os.getenv("PRED_CSV", "").strip()
    if env:
        p = Path(env)
        if p.exists():
            return p

    here = Path(__file__).resolve()
    candidates = [
        here.parent / "next_day_predictions.csv",
        here.parents[2] / "data" / "predictions" / "next_day_predictions.csv",
        Path("data/predictions/next_day_predictions.csv"),  # CWD 대비
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

@st.cache_data(show_spinner=False)
def load_predictions_df() -> pd.DataFrame:
    # 1) 리포 내 파일 우선
    p = resolve_pred_csv()
    if p is not None:
        try:
            return pd.read_csv(p, dtype=str, encoding="utf-8-sig", encoding_errors="ignore")
        except Exception as e:
            st.warning(f"로컬 CSV 읽기 실패: {e}")

    # 2) 원격 URL (Secrets)
    url = st.secrets.get("PRED_CSV_URL")
    if url:
        try:
            return pd.read_csv(url, dtype=str)
        except Exception as e:
            st.error(f"원격 CSV 읽기 실패: {e}")

    return pd.DataFrame()

# -------------------- 상단 버튼: 새로고침 (좌측, 한 줄 고정) --------------------
st.markdown("""
    <style>
    .stButton > button { white-space: nowrap; }
    </style>
""", unsafe_allow_html=True)

col_btn, col_spacer = st.columns([1, 9])
with col_btn:
    if st.button("🔄 새로고침", use_container_width=False, key="refresh_btn"):
        st.rerun()

# -------------------- 데이터 로드 --------------------
df_raw = load_predictions_df()

# 경로/원천 표시
p = resolve_pred_csv()
if p is not None and p.exists():
    st.caption(f"읽는 CSV: `{p}`")
else:
    src = st.secrets.get("PRED_CSV_URL")
    st.caption(f"읽는 CSV: `{src or '찾을 수 없음'}`")

if df_raw.empty:
    st.warning("⚠ 예측 결과 파일이 없습니다.\n"
               "리포에 `data/predictions/next_day_predictions.csv` 또는 "
               "`models/pipeline/next_day_predictions.csv`를 넣거나, "
               "App Secrets에 `PRED_CSV_URL`을 설정하세요.")
    st.stop()

# 전처리(헤더/문자열 정리)
df_raw.columns = [_strip_bom_zwsp(c).lower() for c in df_raw.columns]
for c in df_raw.columns:
    if df_raw[c].dtype == object:
        df_raw[c] = df_raw[c].astype(str).map(_strip_bom_zwsp)

with st.expander("🧪 CSV 진단"):
    st.write("헤더:", list(df_raw.columns))
    st.write("앞부분:", df_raw.head(5))
    st.write("열별 결측수:", df_raw.replace({"": np.nan}).isna().sum())

df = df_raw.copy()

# 필수 컬럼 확인
required = ["asof_date", "pred_for"]
miss = [c for c in required if c not in df.columns]
if miss:
    st.error(f"필수 컬럼 누락: {miss} (현재 컬럼: {list(df.columns)})")
    st.stop()

# 날짜/숫자/불리언 파싱
for col in ["asof_date", "pred_for", "created_at"]:
    if col in df.columns:
        df[col] = df[col].apply(safe_parse_ts)

for col in ["proba_up_low", "proba_up_high", "proba_up", "reject_margin", "driver_feature_scaled"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].replace({"": np.nan}), errors="coerce")

if "is_confident" in df.columns:
    df["is_confident"] = df["is_confident"].astype(str).str.lower().map({"true": True, "false": False})

# 유효한 최신행 (asof_date 있는 마지막)
df = df.sort_values("asof_date")
if df["asof_date"].notna().any():
    last = df[df["asof_date"].notna()].iloc[-1]
else:
    st.error("모든 asof_date가 NaT입니다. CSV의 날짜 값을 확인하세요.")
    st.stop()

# -------------------- 최신 결과 카드 --------------------
st.subheader("📊 최신 예측 결과")

# proba 보정
proba = last.get("proba_up")
proba = float(proba) if pd.notna(proba) else np.nan
if np.isnan(proba):
    p_low  = float(last.get("proba_up_low"))  if pd.notna(last.get("proba_up_low"))  else np.nan
    p_high = float(last.get("proba_up_high")) if pd.notna(last.get("proba_up_high")) else np.nan
    proba  = (p_low + p_high) / 2 if not np.isnan(p_low) and not np.isnan(p_high) else 0.5

margin = float(last.get("reject_margin")) if pd.notna(last.get("reject_margin")) else 0.0

# is_confident 보정
if "is_confident" in last.index and pd.notna(last["is_confident"]):
    confident = bool(last["is_confident"])
else:
    confident = abs(proba - 0.5) > margin

decision = str(last.get("decision", "")).strip().upper()
pred_txt = "📈 상승 예상" if decision == "UP" else ("📉 하락 예상" if decision == "DOWN" else "🤔 보류")

def fmt_date(ts):
    return ts.strftime("%Y-%m-%d") if (isinstance(ts, pd.Timestamp) and pd.notna(ts)) else "-"

colA, colB = st.columns(2)
with colA:
    st.metric("기준일(asof_date)", fmt_date(last.get("asof_date")))
    st.metric("예측일(pred_for)",  fmt_date(last.get("pred_for")))
    if "created_at" in df.columns:
        st.metric("생성 시각", fmt_date(last.get("created_at")))
with colB:
    st.metric("예측", pred_txt)
    st.metric("상승 확률", f"{proba*100:.1f}%")
    st.metric("신뢰 여부", "✅ 신뢰" if confident else "⚠️ 불확실")

# 대표 요인
driver_side = str(last.get("driver_side", "") or "").strip().lower()
driver_feat = str(last.get("driver_feature", "") or "")
if driver_feat:
    cat = feature_to_category(driver_feat)
    side_txt = {"low":"(저)", "high":"(고)"}.get(driver_side, "")
    st.markdown("### 🧭 오늘 예측에 영향을 준 대표 요인")
    st.success(f"**{cat}** {side_txt} · {driver_feat}", icon="📂")

# -------------------- 히스토리 시각화 --------------------
if len(df) > 1 and "proba_up" in df.columns:
    st.markdown("### 📈 최근 예측 추세")
    plot = df.tail(60).copy()
    plot["upper_band"] = 0.5 + plot["reject_margin"].fillna(0)
    plot["lower_band"] = 0.5 - plot["reject_margin"].fillna(0)

    base = alt.Chart(plot).encode(x=alt.X("asof_date:T", title="기준일"))
    band = base.mark_area(opacity=0.15, color="#bbbbbb").encode(y="lower_band:Q", y2="upper_band:Q")
    line = base.mark_line(point=True).encode(
        y=alt.Y("proba_up:Q", title="상승 확률"),
        color=alt.Color("decision:N",
                        scale=alt.Scale(domain=["UP","DOWN","HOLD"], range=["#2ca02c","#d62728","#999999"]),
                        legend=alt.Legend(title="판정")),
        tooltip=["asof_date:T","pred_for:T",
                 alt.Tooltip("proba_up:Q", title="확률", format=".3f"),
                 "decision:N",
                 alt.Tooltip("reject_margin:Q", title="margin", format=".3f")],
    )
    st.altair_chart((band + line).properties(height=320), use_container_width=True)

# -------------------- 최근 10건 표 (확인용) --------------------
with st.expander("🧪 최근 10건 원본 보기"):
    cols_show = [c for c in [
        "asof_date","pred_for","proba_up_low","proba_up_high","proba_up",
        "reject_margin","is_confident","pred_label","decision",
        "driver_side","driver_feature","driver_feature_raw","driver_feature_scaled",
        "model_dir","created_at"
    ] if c in df.columns]
    st.dataframe(df.tail(10)[cols_show])
