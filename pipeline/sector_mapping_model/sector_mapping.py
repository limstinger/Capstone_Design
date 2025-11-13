import os
import logging
import json
import pandas as pd

from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic

# ————————— Logger 세팅 —————————
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    return logging.getLogger(__name__)

logger = setup_logger()

# ————————— 전역 Okt & tokenizer —————————
okt = Okt()
def okt_tokenizer(text: str):
    """
    Okt 형태소 분석기를 사용해 두 글자 이상의 토큰만 리턴.
    pickle 직렬화를 위해 반드시 전역 함수로 정의합니다.
    """
    tokens = okt.morphs(text)
    return [t for t in tokens if len(t) > 1]

def korean_vectorizer():
    """
    CountVectorizer 에 전역 토크나이저를 연결.
    ngram=(1,2), 최소 문서 빈도 5, 최대 문서 비율 80% 로 설정.
    """
    return CountVectorizer(
        tokenizer=okt_tokenizer,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8
    )

# ————————— 데이터 로드 함수 —————————
def load_data(base_dir):
    """
    data/raw 아래의 kospi_news_2023.csv, kospi_news_2024.csv를 읽어 합칩니다.
    """
    raw_dir = os.path.abspath(os.path.join(base_dir, "../../data/raw"))
    paths = [os.path.join(raw_dir, f"kospi_news_{y}.csv") for y in (2021, 2022, 2023, 2024, 2025)]
    dfs = []
    for p in paths:
        logger.info(f"Loading news data from {p} …")
        dfs.append(pd.read_csv(p))
    df = pd.concat(dfs, ignore_index=True)
    df['headline'] = df['headline'].fillna("").astype(str)
    logger.info(f"총 {len(df):,}건의 뉴스 로드 완료")
    return df

# ————————— 토픽 모델링 함수 —————————
def run_topic_modeling(docs, model_path=None):
    """
    BERTopic 모델 학습 또는 저장된 모델 로드.
    - model_path가 존재하면 load 후 transform
    - 아니면 fit_transform 후 save
    """
    # 이미 저장된 모델이 있으면 로드
    if model_path and os.path.isfile(model_path):
        logger.info(f"Saved model found at {model_path}, loading...")
        tm = BERTopic.load(model_path)
        topics, probs = tm.transform(docs)
        return tm, topics, probs

    # 신규 학습 세팅
    vectorizer = korean_vectorizer()
    umap_model = UMAP(n_neighbors=15, n_components=2, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=50, prediction_data=True)

    tm = BERTopic(
        embedding_model="sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens",
        vectorizer_model=vectorizer,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True,
        verbose=True
    )

    # fit & transform
    topics, probs = tm.fit_transform(docs)

    # 모델 저장
    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        tm.save(model_path)
        logger.info(f"✅ 모델 저장 완료 → {model_path}")

    return tm, topics, probs

# ————————— 토픽 매핑 저장 함수 —————————
def save_topic_mapping(topic_model, base_dir):
    """
    토픽 번호와 이름이 담긴 매핑을 CSV/JSON으로 저장합니다.
    """
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # 토픽 정보 DataFrame 생성
    topic_info = topic_model.get_topic_info()[['Topic', 'Name']]
    topic_info = topic_info.rename(columns={'Topic':'topic_id', 'Name':'topic_name'})

    # CSV로 저장
    csv_path = os.path.join(output_dir, "topic_mapping.csv")
    topic_info.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"✅ 토픽 매핑 저장 → {csv_path}")

    # JSON으로 저장
    json_path = os.path.join(output_dir, "topic_mapping.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(topic_info.to_dict(orient='records'), f, ensure_ascii=False, indent=2)
    logger.info(f"✅ 토픽 매핑 JSON 저장 → {json_path}")

# ————————— 시각화 & 저장 함수 —————————
def visualize_and_save(topic_model, probs, base_dir):
    """
    토픽 간 거리 맵, 상위 토픽 바 차트, 첫 문서 분포 차트까지
    모두 HTML로 output 폴더에 저장합니다.
    """
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # 1) 인터토픽 거리 맵
    fig1 = topic_model.visualize_topics()
    fig1.write_html(
        os.path.join(output_dir, "intertopic_map.html"),
        full_html=True, include_plotlyjs="cdn"
    )
    logger.info("✅ 인터토픽 맵 저장 → output/intertopic_map.html")

    # 2) 상위 10개 토픽 바 차트
    fig2 = topic_model.visualize_barchart(top_n_topics=10)
    fig2.write_html(
        os.path.join(output_dir, "topic_barchart.html"),
        full_html=True, include_plotlyjs="cdn"
    )
    logger.info("✅ 토픽 바 차트 저장 → output/topic_barchart.html")

    # 3) 첫 문서에 대한 토픽 분포 차트
    fig3 = topic_model.visualize_distribution(probs[0])
    fig3.write_html(
        os.path.join(output_dir, "example_distribution.html"),
        full_html=True, include_plotlyjs="cdn"
    )
    logger.info("✅ 예시 분포 차트 저장 → output/example_distribution.html")

# ————————— 메인 스크립트 —————————
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1) 데이터 로드
    df = load_data(base_dir)
    docs = df['headline'].tolist()

    # 2) 토픽 모델링
    model_path = os.path.join(base_dir, "output", "sector_model.pkl")
    topic_model, topics, probs = run_topic_modeling(docs, model_path=model_path)

    # 3) DataFrame에 태깅
    df['topic']      = topics
    df['topic_prob'] = [max(p) if p is not None else 0 for p in probs]

    # 4) 토픽 매핑 저장
    save_topic_mapping(topic_model, base_dir)

    # 5) 토픽 요약 정보 출력
    logger.info("=== 토픽 요약 정보 (상위 10개) ===")
    print(topic_model.get_topic_info().head(10))

    # 6) 시각화 및 HTML 저장
    visualize_and_save(topic_model, probs, base_dir)

    # 7) CSV로 결과 저장
    csv_path = os.path.join(base_dir, "output", "news_with_sectors.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"✅ 섹터(토픽) 태깅 데이터 저장 → {csv_path}")

if __name__ == "__main__":
    main()
