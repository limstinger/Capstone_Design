import os
import time
import schedule
import requests  
from datetime import datetime
from urllib.parse import quote
from newspaper import Article, Config
from pymongo import MongoClient, errors
from dotenv import load_dotenv

# ─── 환경 변수 로드 ─────────────────────────
load_dotenv()
CLIENT_ID     = os.getenv("NAVER_CLIENT_ID")
CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
MONGO_URI     = os.getenv("MONGO_URI")

if not CLIENT_ID or not CLIENT_SECRET:
    raise RuntimeError("환경변수 NAVER_CLIENT_ID/SECRET 값을 확인하세요")
if not MONGO_URI:
    raise RuntimeError("환경변수 MONGO_URI를 .env에 설정하세요")

# ─── MongoDB 연결 & 컬렉션 준비 ────────────
client = MongoClient(MONGO_URI)
db     = client["newsdb"]
col    = db.articles
col.create_index("url", unique=True)

# ─── 수집할 키워드 목록 ─────────────────────
KEYWORDS = [
    "정치", "의료", "반도체", "2차 전지",
    "금융정책", "주식", "환율", "국제무역", "로봇", "방산"
]

# ─── 총 수집건수 ───────────────────────────
TOTAL_PER_RUN = 1000
base = TOTAL_PER_RUN // len(KEYWORDS)
remainder = TOTAL_PER_RUN % len(KEYWORDS)

# ─── newspaper3k 설정 ──────────────────────
config = Config()
config.request_timeout = 20  # 타임아웃 20초

def fetch_meta(query: str, display: int):
    url = (
        "https://openapi.naver.com/v1/search/news.json"
        f"?query={quote(query)}&display={display}&sort=date"
    )
    headers = {
        "X-Naver-Client-Id":     CLIENT_ID,
        "X-Naver-Client-Secret": CLIENT_SECRET,
    }
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r.json().get("items", [])

def fetch_full_article(url: str, retries: int = 2):
    art = Article(url, language="ko", config=config)
    for attempt in range(1, retries+1):
        try:
            art.download(); art.parse()
            return art.text
        except Exception:
            if attempt < retries:
                time.sleep(2)
    return None

def save_to_mongo(item, content, source_tag, lang):
    url = item.get("originallink") or item.get("link")
    doc = {
        "source":      source_tag,
        "language":    lang,
        "title":       item.get("title"),
        "description": item.get("description"),
        "url":         url,
        "urlToImage":  None,
        "pubDate":     item.get("pubDate"),
        "content":     content,
        "fetchedAt":   datetime.utcnow().isoformat()
    }
    try:
        col.insert_one(doc)
        return True
    except errors.DuplicateKeyError:
        return False

def run_naver_pipeline():
    """2시간마다 호출될 네이버 뉴스 수집 파이프라인"""
    print(f"[{datetime.now()}] >>> 네이버 뉴스 수집 시작 (총 {TOTAL_PER_RUN}건)")
    total_saved = 0
    total_dup   = 0

    for i, kw in enumerate(KEYWORDS):
        per_kw = base + (remainder if i == len(KEYWORDS)-1 else 0)
        print(f"\n[{datetime.now()}] --- 키워드 `{kw}`: 목표 {per_kw}건 ---")
        metas = fetch_meta(kw, per_kw)
        print(f"[{datetime.now()}] ▶ 메타 가져옴: {len(metas)}건")

        saved = 0
        dup   = 0
        for idx, m in enumerate(metas, start=1):
            print(f"[{datetime.now()}] [{kw} {idx}/{len(metas)}] 크롤링...", end=" ")
            link = m.get("originallink") or m.get("link")
            full = fetch_full_article(link)
            if full:
                print("성공 / 저장→", end=" ")
                if save_to_mongo(m, full, source_tag=kw, lang="ko"):
                    print("OK")
                    saved += 1
                else:
                    print("중복")
                    dup += 1
            else:
                print("⚠️ 실패(본문 없음)")
            time.sleep(0.1)

        total_saved += saved
        total_dup   += dup
        print(f"[{datetime.now()}] --- `{kw}` 완료: 저장 {saved}건, 중복 {dup}건 ---")

    print(f"\n[{datetime.now()}] ✅ 전체 완료: 저장 {total_saved}건, 중복 {total_dup}건\n")

# ─── 스케줄링 ──────────────────────────────
schedule.every(2).hours.do(run_naver_pipeline)

# 첫 실행
run_naver_pipeline()

# 백그라운드에서 계속 실행
while True:
    schedule.run_pending()
    time.sleep(30)