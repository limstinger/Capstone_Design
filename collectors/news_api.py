import os
import time
import schedule
from datetime import datetime
from newsapi import NewsApiClient
from pymongo import MongoClient, errors
from dotenv import load_dotenv

# ─── 환경 변수 로드 ─────────────────────────
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
MONGO_URI   = os.getenv("MONGO_URI")

if not NEWSAPI_KEY or not MONGO_URI:
    raise RuntimeError("NEWSAPI_KEY와 MONGO_URI를 .env에 설정하세요")

# ─── MongoDB 연결 & 컬렉션 준비 ────────────
client = MongoClient(MONGO_URI)
db     = client["newsdb"]
col    = db.articles
col.create_index("url", unique=True)  # url 기준 중복 방지

# ─── NewsAPI 클라이언트 초기화 ─────────────
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

# ─── Daily State ────────────────────────────
DAILY_LIMIT    = 100
FETCH_PER_RUN  = 8     # 2시간마다 8건
state = {
    "remaining": DAILY_LIMIT,
    "last_reset": datetime.utcnow().date()
}

def reset_daily_counter():
    state["remaining"] = DAILY_LIMIT
    state["last_reset"] = datetime.utcnow().date()
    print(f"[{datetime.now()}] Daily counter reset → {DAILY_LIMIT}")

def save_to_mongo(art: dict):
    """NewsAPI용 doc 생성 및 저장 (네이버와 같은 스키마)"""
    doc = {
        "source":      art.get("source", {}).get("name"),
        "language":    "en",
        "title":       art.get("title"),
        "description": art.get("description"),
        "url":         art.get("url"),
        "urlToImage":  art.get("urlToImage"),
        "pubDate":     art.get("publishedAt"),
        "content":     art.get("content"),
        "fetchedAt":   datetime.utcnow().isoformat()
    }
    try:
        col.insert_one(doc)
        return True
    except errors.DuplicateKeyError:
        return False

def fetch_and_store_chunk():
    # 하루 한도 리셋 체크
    if datetime.utcnow().date() != state["last_reset"]:
        reset_daily_counter()

    if state["remaining"] <= 0:
        print(f"[{datetime.now()}] ⛔ 오늘 한도({DAILY_LIMIT}) 소진, 스킵")
        return

    to_fetch = min(FETCH_PER_RUN, state["remaining"])
    print(f"[{datetime.now()}] ▶ Fetching {to_fetch} headlines (remaining {state['remaining']})")

    articles = newsapi.get_top_headlines(
        category="general",  # 글로벌 일반 속보
        language="en",
        page_size=to_fetch
    ).get("articles", [])

    saved, dup = 0, 0
    for art in articles:
        if save_to_mongo(art):
            saved += 1
        else:
            dup += 1

    state["remaining"] -= saved
    print(f"[{datetime.now()}] ✔ Saved {saved} (dup {dup}), remaining → {state['remaining']}")

# ─── 스케줄러 등록 ──────────────────────────
schedule.every().day.at("00:00").do(reset_daily_counter)
schedule.every(2).hours.do(fetch_and_store_chunk)

print("🗓 NewsAPI Scheduler Started (2h × 8건, 일일 100건 한도)")
fetch_and_store_chunk()  # 첫 실행

while True:
    schedule.run_pending()
    time.sleep(30)