# ✅ 1. 운영환경(Environment)
■ OS 및 실행 환경

* `Windows 10/11 또는 Linux(Ubuntu 20.04 이상)`

* `Python 3.9 권장`

* GPU 사용 시: `NVIDIA GPU + CUDA 11.x 또는 12.x`

* PyTorch CUDA 빌드 필요

■ 필수 외부 실행환경

* Chrome 브라우저

* `ChromeDriver (selenium 자동 제어)`

* DuckDB 데이터베이스(파일 기반, 설치 불필요)

* 한국은행 ECOS API (경제지표 수집용)

* 네이버 뉴스 페이지 접근 가능해야 함


## 📂 2. 프로젝트 구조(Directory Structure)

```
Capstone_Design/
 ├─ models/
 │   └─ pipeline/
 │        economic_indicators.py         # 경제·시장지표 수집
 │        news_indicators.py             # 뉴스 크롤링 + 감성 분석
 │        preprocess.py                  # 통합 전처리
 │        predict_next_day.py            # 다음 영업일 방향성 예측
 │        run_pipeline.py                # 전체 자동 파이프라인 실행
 │        model_train_margin_log.py      # 예측 모델 학습
 │
 ├─ data/
 │   ├─ raw/                             # 원천 CSV (KOSPI/환율/VKOSPI/WTI 등)
 │   ├─ lake/
 │   │   └─ bronze/                      # 날짜 기반 파티션 Parquet
 │   ├─ processed/
 │   │   └─ training_with_refined_features.csv
 │   └─ predictions/
 │       └─ next_day_predictions.csv     # 예측 결과 (누적)
 │

```
## 🔐 3. .env 설정
* ECOS_KEY=한국은행_API_KEY
* EIA_KEY=EIA_API_KEY

## 📦 4. 필수 라이브러리 (requirements.txt)


## ⚙ 5. 실행 순서(전체 파이프라인) : run_pipeline.py
### 🔹 (1) 경제 지표 수집 -> ``python models/pipeline/economic_indicators.py``


* 환율(USD/KRW 등)

* VKOSPI (KOPSI 변동성)

* 금/WTI

* 한국은행 기준금리 등

### 🔹 (2) 뉴스 데이터 수집 + 감성 분석 -> ``python models/pipeline/news_indicators.py``


* 네이버 뉴스 크롤링

* 본문/헤드라인 정제

* KR-FinBERT 기반 감성 분석

* BERTopic 기반 키워드/토픽 추출

### 🔹 (3) 통합 전처리(Feature Engineering) -> ``python models/pipeline/preprocess.py``


### 🔹 (4) 다음 영업일 예측 -> ``python models/pipeline/predict_next_day.py``

### 🔹 (5) 예측 확인(시각화) -> ``streamlit run models/pipeline/streamlit_app.py``
