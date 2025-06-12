import pandas as pd
from pathlib import Path

def main():
    # 1) 프로젝트 루트와 데이터 폴더
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR     = PROJECT_ROOT / "data" / "raw"
    OUT_DIR      = DATA_DIR 
    OUT_DIR.mkdir(exist_ok=True)

    # 2) 파일명과 사용할 종가 컬럼명을 지정
    #    gold_wti 파일에는 'gold_krw'와 'wti_krw' 두 칼럼을 모두 사용
    files = {
        "usd_krw":   ("USD_KRW Historical Data.csv",      "Price"),
        "cny_krw":   ("CNY_KRW Historical Data.csv",      "Price"),
        "eur_krw":   ("EUR_KRW Historical Data.csv",      "Price"),
        "jpy_krw":   ("JPY_KRW Historical Data.csv",      "Price"),
        "usd_jpy":   ("USD_JPY Historical Data.csv",      "Price"),
        "gold_wti":  ("gold_wti_prices_krw.csv",          ["gold_krw", "wti_krw"]),
        "kospi" :    ("KOSPI Historical Data.csv",        "Price"),
        "kosdaq":    ("KOSDAQ Historical Data.csv",        "Price")
        # 예시: "fed_rate": ("fed_funds_rate.csv", "Close"),
    }

    dfs = []
    for key, spec in files.items():
        fname, price_cols = spec
        path = DATA_DIR / fname

        if not path.exists():
            print(f"⚠️ 파일이 없습니다: {path}")
            continue

        # CSV 로드
        df = pd.read_csv(path, parse_dates=[0])
        date_col = df.columns[0]  # 첫 번째 컬럼은 날짜라고 가정

        # gold_wti만 다중 가격 컬럼 처리
        if isinstance(price_cols, list):
            use_cols = [date_col] + price_cols
            tmp = df[use_cols].copy()
            # 칼럼명 통일
            tmp.columns = ["date"] + price_cols
        else:
            # 단일 price 컬럼 처리
            tmp = df[[date_col, price_cols]].copy()
            tmp.columns = ["date", key]

        dfs.append(tmp)

    if not dfs:
        print("❌ 병합할 데이터가 하나도 없습니다.")
        return

    # 3) 순차 병합 (inner join on date)
    merged = dfs[0]
    for df_ in dfs[1:]:
        merged = pd.merge(merged, df_, on="date", how="inner")

    # 4) 정렬 및 저장
    merged = merged.sort_values("date").reset_index(drop=True)

    # merged['usd_krw'] = pd.to_numeric(merged['usd_krw'].str.replace('"', ''), errors='coerce')
    # merged['eur_krw'] = pd.to_numeric(merged['eur_krw'].str.replace('"', ''), errors='coerce')
    # merged['kospi']   = pd.to_numeric(merged['kospi'].str.replace('"', ''), errors='coerce')

    out_path = OUT_DIR / "merged_macro_dataset.csv"
    merged.to_csv(out_path, index=False)
    print(f"✔️ 병합 완료: {out_path} (shape: {merged.shape})")

if __name__ == "__main__":
    main()

    