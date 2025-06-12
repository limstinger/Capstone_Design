import pandas as pd

# 1) CSV 파일 불러오기 (parse_dates 로 날짜 타입 지정)
gold_df = pd.read_csv(
    'XAU_USD Historical Data.csv',
    parse_dates=['Date'],
    thousands=',',       # 천 단위 구분자 제거
    usecols=['Date', 'Price']
).rename(columns={'Price': 'gold_usd'})

wti_df = pd.read_csv(
    'WTI_USD Historical Data.csv',
    parse_dates=['Date'],
    thousands=',',
    usecols=['Date', 'Price']
).rename(columns={'Price': 'wti_usd'})

fx_df = pd.read_csv(
    'USD_KRW Historical Data.csv',
    parse_dates=['Date'],
    thousands=',',
    usecols=['Date', 'Price']
).rename(columns={'Price': 'usd_krw_rate'})

# 3) 세 DataFrame을 날짜 기준으로 병합
#    how='inner' 로 공통 날짜만 남기거나, how='outer' 로 전체 날짜를 보존할 수 있습니다.
df = pd.merge(gold_df, fx_df, on='Date', how='inner')
df = pd.merge(df,      wti_df, on='Date', how='inner')

# 4) KRW 기준 가격 계산
df['gold_krw'] = df['gold_usd'] * df['usd_krw_rate']
df['wti_krw']  = df['wti_usd']  * df['usd_krw_rate']

# 5) 결과 확인 및 저장
print(df[['Date','gold_krw','wti_krw']].head())
df.to_csv('gold_wti_prices_krw.csv', index=False)