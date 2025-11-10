# streamlit_app.py
# -*- coding: utf-8 -*-
import os, re, math
from io import StringIO
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# -------------------- 기본 설정 --------------------
pd.set_option("future.no_silent_downcasting", True)
st.set_page_config(page_title="📈 다음 날 예측", layout="centered")
st.title("📈 다음 개장일 코스피 예측")
st.caption("※ 내부 모델 예측값입니다. 실제 투자판단에는 참고용으로만 사용하세요.")

# -------------------- 도우미 --------------------
def _strip_bom_zwsp(s: str) -> str:
    return re.sub(r"[\u200b\ufeff]", "", s).strip()

def safe_parse_ts(x):
    """문자/NaN 안전 파싱 (YYYY-MM-DD 같은 일반 문자열 우선)"""
    if x is None or x == "" or (isinstance(x, float) and math.isnan(x)):
        return pd.NaT
    s = str(x).strip()
    s = _strip_bom_zwsp(s)
    return pd.to_datetime(s, errors="coerce")

# -------------------- 파일 찾기: repo상대/ENV/secrets/URL/업로드 --------------------
def candidate_paths() -> list[Path]:
    here = Path(__file__).parent.resolve()
    repo_root = here  # Streamlit Cloud는 보통 app 파일이 repo 루트에 있음
    return [
        repo_root / "data" / "predictions" / "next_day_predictions.csv",
        repo_root / "data" / "next_day_predictions.csv",
        repo_root / "next_day_predictions.csv",
    ]

def get_secret_path() -> Optional[Path]:
    # secrets.toml에 [paths] PRED_CSV="data/predictions/xxx.csv" 식으로 넣으면 사용
    try:
        p = st.secrets["paths"].get("PRED_CSV", "")
        if p:
            return Path(p)
    except Exception:
        pass
    # 환경변수 PRED_CSV 로컬/클라우드 공통 지원
    env_p = os.environ.get("PRED_CSV", "")
    return Path(env_p) if env_p else None

def get_secret_url() -> Optional[str]:
    # secrets.toml에 [urls] PRED_CSV_URL="https://..." 넣으면 원격에서 읽음
    try:
        url = st.secrets["urls"].get("PRED_CSV_URL", "")
        return url if isinstance(url, str) and url.startswith(("http://", "https://")) else None
    except Exception:
        return None

@st.cache_data(ttl=60)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig", encoding_errors="ignore")

@st.cache_data(ttl=60)
def load_csv_url(url: str) -> pd.DataFrame:
    # 공개 URL이면 pandas가 직접 읽을 수 있습니다(GitHub raw, S3 presigned 등)
    return pd.read_csv(url, dtype=str)

def find_and_load_predictions() -> Tuple[pd.DataFrame, str]:
    # 1) secrets/ENV 지정 경로
    sp = get_secret_path()
    if sp and Path(sp).exists():
        return load_csv(Path(sp)), f"file:{Path(sp)}"

    # 2) repo 상대 경로 후보들
    for p in candidate_paths():
        if p.exists():
            return load_csv(p), f"file:{p}"

    # 3) URL (예: GitHub raw/S3)
    url = get_secret_url()
    if url:
        try:
            return load_csv_url(url), f"url:{url}"
        except Exception as e:
            st.warning(f"원격 CSV 로드 실패: {e}")

    # 4) 업로드로 대체
    up = st.file_uploader("CSV 업로드(임시 대체)", type=["csv"])
    if up is not None:
        data = up.read().decode("utf-8", errors="ignore")
        return pd.read_csv(StringIO(data), dtype=str), "upload:<session>"

    # 전부 실패
    raise FileNotFoundError("CSV를 찾을 수 없습니다.")

# ───────────────────────────────────────────────────────────
# 피처 설명 사전 + 패턴 기반 설명기 (원문 그대로 유지)
# ───────────────────────────────────────────────────────────
FEATURE_EXPLAIN = {
    "date": "거래 기준일.",
    "close": "기준일 종가(KOSPI 지수).",
    "next_close": "다음 거래일 종가(타깃 계산용).",
    "logret_next": "다음 거래일 종가의 로그수익률(=ln(next_close/close)).",
    "label_up": "다음 날 상승(1) / 하락(0) 레이블.",
    "is_pre_holiday": "연휴/공휴일 직전 거래일 여부(1이면 직전).",
    "is_post_holiday": "연휴/공휴일 직후 거래일 여부(1이면 직후).",
    "days_since_last_trade": "직전 거래일까지 경과 일수(연휴·주말 길이 반영).",
    "entropy_full": "그날 뉴스 토픽 분포의 엔트로피(높을수록 이슈가 분산).",
    "entropy_excl_unassigned": "‘미할당(unassigned)’ 토픽 제외한 엔트로피.",
    "hhi_full": "토픽 집중도(HHI, 높을수록 특정 토픽에 쏠림).",
    "hhi_excl_unassigned": "‘미할당’ 제외한 HHI.",
    "kl_topic_shift": "전일 대비 토픽 분포의 변화(=KL divergence).",
    "top1_topic": "그날 1순위 토픽 ID.",
    "top1_soft_prop": "1순위 토픽 확률 비중.",
    "top2_topic": "그날 2순위 토픽 ID.",
    "top2_soft_prop": "2순위 토픽 확률 비중.",
    "top3_topic": "그날 3순위 토픽 ID.",
    "top3_soft_prop": "3순위 토픽 확률 비중.",
    "is_top1_unassigned": "1순위 토픽이 ‘미할당’인지 여부.",
    "top1_topic_change_flag": "1순위 토픽이 전일과 달라졌는지 여부.",
    "top1_soft_prop_diff": "1순위 토픽 비중의 전일 대비 변화량.",
    "top2_soft_prop_diff": "2순위 토픽 비중의 전일 대비 변화량.",
    "top3_soft_prop_diff": "3순위 토픽 비중의 전일 대비 변화량.",
    "top1_new_emerge": "1순위 토픽이 새롭게 등장했는지 여부.",
    "top2_new_emerge": "2순위 토픽 새로 등장 여부.",
    "top3_new_emerge": "3순위 토픽 새로 등장 여부.",
    "entropy_trend": "엔트로피 추세(증가: 이슈 분산, 감소: 이슈 집중).",
    "hhi_trend": "집중도(HHI) 추세.",
    "concentration_momentum": "토픽 집중도의 단기 모멘텀.",
    "pos_prob": "뉴스 긍정 확률 평균.",
    "neg_prob": "뉴스 부정 확률 평균.",
    "sentiment_score": "감성 스코어(보통 pos-neg 또는 모델 출력 기반).",
    "sentiment_momentum_1d": "감성 점수의 1일 모멘텀(전일 대비 변화).",
    "sentiment_momentum_3d": "감성 점수의 3일 모멘텀.",
    "sentiment_roll5_mean_prev": "직전 5일 평균 감성.",
    "sentiment_surprise_5d": "최근 5일 평균 대비 ‘감성 서프라이즈’.",
    "sentiment_vol_5d": "감성의 5일 변동성(표준편차).",
    "sentiment_z_30": "감성 점수의 30일 Z-점수(장기 평균/표준편차 표준화).",
    "rate_announce": "금리 발표 당일(또는 직후) 더미/지표.",
    "rate_announce_lag_1d": "금리 발표 후 1영업일 경과 효과.",
    "rate_announce_lag_2d": "금리 발표 후 2영업일 경과 효과.",
    "rate_announce_lag_3d": "금리 발표 후 3영업일 경과 효과.",
    "rate_announce_lag_4d": "금리 발표 후 4영업일 경과 효과.",
    "rate_announce_lag_5d": "금리 발표 후 5영업일 경과 효과.",
    "days_since_rate_announce": "마지막 금리 발표 이후 경과 영업일.",
    "rate_announce_decay": "금리 이벤트 효과의 시간감쇠(가중치).",
    "logret_KOSPI_Close": "KOSPI 지수의 일간 로그수익률.",
    "KOSPI_Volatility": "KOSPI 변동성(예: VKOSPI 등 또는 자체 지표).",
    "logret_KOSPI_Volatility": "KOSPI 변동성 지표의 로그수익률.",
    "USD_KRW": "원/달러 환율(원화 약세↑: 수치↑).",
    "logret_USD_KRW": "원/달러 환율의 로그수익률.",
    "USD_JPY": "달러/엔 환율.",
    "logret_USD_JPY": "달러/엔 환율의 로그수익률.",
    "JPY_KRW": "원/엔 환율.",
    "logret_JPY_KRW": "원/엔 환율의 로그수익률.",
    "EUR_KRW": "원/유로 환율.",
    "logret_EUR_KRW": "원/유로 환율의 로그수익률.",
    "CNY_KRW": "원/위안 환율.",
    "logret_CNY_KRW": "원/위안 환율의 로그수익률.",
    "USD_KRW_derived": "원/달러 파생 지표(예: 크로스·가중·정규화).",
    "logret_next_momentum_3d": "타깃(logret_next)의 3일 모멘텀(학습에서 누수 방지 처리).",
    "logret_next_vol_5d": "타깃(logret_next)의 5일 변동성.",
    "logret_KOSPI_Close_momentum_3d": "KOSPI 로그수익률 3일 모멘텀.",
    "logret_KOSPI_Close_vol_5d": "KOSPI 로그수익률 5일 변동성.",
    "logret_KOSPI_Volatility_momentum_3d": "KOSPI 변동성 3일 모멘텀.",
    "logret_KOSPI_Volatility_vol_5d": "KOSPI 변동성 5일 변동성.",
    "logret_USD_KRW_momentum_3d": "원/달러 환율 3일 모멘텀.",
    "logret_USD_KRW_vol_5d": "원/달러 환율 5일 변동성.",
    "logret_USD_JPY_momentum_3d": "달러/엔 3일 모멘텀.",
    "logret_USD_JPY_vol_5d": "달러/엔 5일 변동성.",
    "logret_JPY_KRW_momentum_3d": "원/엔 3일 모멘텀.",
    "logret_JPY_KRW_vol_5d": "원/엔 5일 변동성.",
    "logret_EUR_KRW_momentum_3d": "원/유로 3일 모멘텀.",
    "logret_EUR_KRW_vol_5d": "원/유로 5일 변동성.",
    "logret_CNY_KRW_momentum_3d": "원/위안 3일 모멘텀.",
    "logret_CNY_KRW_vol_5d": "원/위안 5일 변동성.",
    "ma_5": "5일 이동평균.",
    "ma_20": "20일 이동평균.",
    "ma_diff_5_20": "MA(5)-MA(20) 차이(골든/데드 크로스 감지 보조).",
    "ma_ratio_5_20": "MA(5)/MA(20) 비율.",
    "macd_line": "MACD 라인(12-26 지수이평 차).",
    "signal_line": "MACD 시그널(9일).",
    "macd_histogram": "MACD-시그널(양수↑ 상승 모멘텀).",
    "price_momentum_3d": "가격(또는 지수) 3일 모멘텀.",
    "sentiment_x_price_logret": "감성 × KOSPI 수익률 상호작용.",
    "sentiment_x_usdkrw_logret": "감성 × 환율(logret_USD_KRW) 상호작용.",
    "top1_topic_change_flag_x_sentiment_surprise": "메인 토픽 교체 × 감성 서프라이즈 결합효과.",
    "relative_currency_momentum": "주요 통화 간 상대적 모멘텀(달러 강세/약세 반영).",
    "kospi_vol_5d": "KOSPI 5일 변동성(표준편차).",
    "vol_ratio_kospi_usdkrw": "KOSPI 변동성 대비 환율 변동성 비율.",
    "dow_sin": "요일(0~6)을 Sine로 임베딩(주기성 반영).",
    "dow_cos": "요일(0~6)을 Cosine으로 임베딩.",
    "rsi_14": "14일 RSI(70↑ 과열·30↓ 침체의 단순 신호).",
    "bb_width_20": "볼린저 밴드폭(가격 변동성의 상대적 크기).",
    "return_z_20": "수익률의 20일 Z-점수(극단값 탐지).",
    "vol_5d": "수익률 5일 변동성.",
    "vol_20d": "수익률 20일 변동성.",
    "vol_ratio_5_20": "5일/20일 변동성 비율(단기 급등락 민감).",
}

def _pattern_explain(key: str) -> Optional[str]:
    if key.startswith("logret_"):
        base = key.replace("logret_", "")
        return f"{base}의 일간 로그수익률(ln(today/yesterday))."
    if "_momentum_3d" in key:
        base = key.replace("_momentum_3d", "")
        return f"{base}의 3일 모멘텀(최근 흐름)."
    if key.endswith("_vol_5d"):
        base = key.replace("_vol_5d", "")
        return f"{base}의 5일 변동성(단기 표준편차)."
    if key.startswith("ma_"):
        return "이동평균(MA). 기간 숫자는 일 수."
    if key in ("macd_line", "signal_line", "macd_histogram"):
        return FEATURE_EXPLAIN.get(key)
    return None

def explain_feature(name: str) -> str:
    k = (name or "").strip()
    if k in FEATURE_EXPLAIN:
        return FEATURE_EXPLAIN[k]
    pat = _pattern_explain(k)
    if pat:
        return pat
    return "설명 준비 중인 피처입니다. 컬럼명만으로는 맥락이 불분명할 수 있어요."

def feature_to_category(feat: str) -> str:
    if not feat:
        return "기타"
    f = str(feat).upper()
    CATEGORY_SPECS = [
        ("환율",      [r"(USD|EUR|JPY|CNY|CNH|GBP|AUD|CAD|CHF|SGD|HKD).*_KRW", r"^USD_", r"^EUR_", r"^JPY_", r"KRW$"]),
        ("기술적 지표", [r"(KTB|TBOND|BOND|YIELD|IR_|KORIBOR|CD_|UST|US10Y|US2Y)"]),
        ("원자재",    [r"(WTI|BRENT|XAU|XAG|GOLD|SILVER|COPPER|NICKEL|ALUMINIUM|GAS|LNG|COAL|OIL)"]),
        ("주가지수",  [r"(KOSPI|KOSDAQ|^SPX$|S&P|^NDX$|IXIC|^DJI$|HSI|SSE|NIKKEI|NKY|DAX|FTSE)"]),
        ("변동성",    [r"(VIX|VKOSPI|VOL_)"]),
        ("거시지표",  [r"(CPI|PPI|PMI|ISM|UNEMP|GDP|RETAIL|HOUSING|INDUSTRIAL|M2|EXPORT|IMPORT)"]),
        ("뉴스심리",  [r"(SENTIMENT|BERT|ENTROPY|TOPIC|NEWS|POLARITY)"]),
    ]
    for cat, pats in CATEGORY_SPECS:
        if any(re.search(p, f) for p in pats):
            return cat
    return "복합"

# -------------------- 상단 버튼: 새로고침 (좌측, 한 줄 고정) --------------------
st.markdown("""
    <style>
    .stButton > button { white-space: nowrap; }
    </style>
""", unsafe_allow_html=True)

col_btn, col_spacer = st.columns([1, 9])
with col_btn:
    if st.button("🔄 새로고침", use_container_width=False, key="refresh_btn"):
        st.cache_data.clear()
        st.rerun()

# -------------------- 데이터 로드 --------------------
try:
    df_raw, source_info = find_and_load_predictions()
    st.caption(f"읽는 소스: `{source_info}`")
except FileNotFoundError as e:
    st.warning("⚠ 예측 결과 CSV를 찾지 못했습니다. 아래 중 한 가지를 설정해 주세요.\n"
               "1) repo에 `data/predictions/next_day_predictions.csv` 추가\n"
               "2) `secrets.toml`에 `[paths] PRED_CSV=\"data/...csv\"` 또는 `[urls] PRED_CSV_URL=\"https://...\"`\n"
               "3) 위의 업로더로 CSV 업로드")
    st.stop()

# 전처리(컬럼 정리/트림)
df_raw.columns = [_strip_bom_zwsp(c).lower() for c in df_raw.columns]
for c in df_raw.columns:
    if df_raw[c].dtype == object:
        df_raw[c] = df_raw[c].astype(str).map(_strip_bom_zwsp)

with st.expander("🧪 CSV 진단"):
    st.write("헤더:", list(df_raw.columns))
    st.write("앞부분:", df_raw.head(5))
    st.write("열별 결측수:", df_raw.replace({"": np.nan}).isna().sum())

df = df_raw.copy()

# 필수 컬럼
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

# 유효한 최신행
df = df.sort_values("asof_date")
if df["asof_date"].notna().any():
    last = df[df["asof_date"].notna()].iloc[-1]
else:
    st.error("모든 asof_date가 NaT입니다. CSV의 날짜 값을 확인하세요.")
    st.stop()

# -------------------- 최신 결과 카드 --------------------
st.subheader("📊 최신 예측 결과")

proba = last.get("proba_up")
proba = float(proba) if pd.notna(proba) else np.nan
if np.isnan(proba):
    p_low  = float(last.get("proba_up_low"))  if pd.notna(last.get("proba_up_low"))  else np.nan
    p_high = float(last.get("proba_up_high")) if pd.notna(last.get("proba_up_high")) else np.nan
    proba  = (p_low + p_high) / 2 if not np.isnan(p_low) and not np.isnan(p_high) else 0.5

margin = float(last.get("reject_margin")) if pd.notna(last.get("reject_margin")) else 0.0

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
def show_driver():
    driver_feat = str(last.get("driver_feature", "") or "")
    if not driver_feat:
        return
    cat = feature_to_category(driver_feat)
    st.markdown("### 🧭 오늘 예측에 영향을 준 대표 요인")
    st.success(f"**{cat}** · {driver_feat}", icon="📂")
    st.caption(f"설명 · {explain_feature(driver_feat)}")

show_driver()

# -------------------- 히스토리 시각화 --------------------
if len(df) > 1:
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
