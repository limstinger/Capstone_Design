import os
import time
import math
import requests
from datetime import datetime, timedelta
from urllib.parse import quote
from newspaper import Article, Config
from bs4 import BeautifulSoup
from pymongo import MongoClient, errors
from dotenv import load_dotenv

# ─── 환경 변수 로드 ─────────────────────────
load_dotenv()
CLIENT_ID     = os.getenv("NAVER_CLIENT_ID")
CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
MONGO_URI     = os.getenv("MONGO_URI")

if not CLIENT_ID or not CLIENT_SECRET:
    raise RuntimeError("NAVER_CLIENT_ID/SECRET 환경변수 필요")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI 환경변수 필요")

# ─── MongoDB 연결 ──────────────────────────
client = MongoClient(MONGO_URI)
db     = client["newsdb"]
col    = db.articles
col.create_index("url", unique=True)

# ─── 키워드 목록 ────────────────────────────
KEYWORDS = [
    "LG", "KOSPI", "KOSDAQ"
]

# KEYWORDS = ["금리", "완화","환율", "물가", "CPI", "GDP","연준", "일본은행","코스피", "코스닥", "반도체", "AI", "제약","바이오","삼성전자", "SK하이닉스", "현대차", "한화","전쟁", "무역전쟁", "금융", "경제"]

# ─── newspaper3k 설정 ──────────────────────
config = Config()
config.request_timeout = 20

# ─── 수집 기간 설정 ─────────────────────────
START_DATE = datetime(2023, 1, 1)
END_DATE   = datetime(2024, 12, 31)

# ─── API 호출 제한 설정 ────────────────────
MAX_REQUESTS   = 25000
request_count  = 0
limit_exceeded = False

# ─── 페이징 & 슬라이스 설정 ─────────────────
PER_PAGE   = 100   # 한 페이지당 100건
PAGES      = 10    # start=1,101,…,901 → 10페이지 → 최대 1,000건
SLICES     = 10    # 기간을 10조각으로 분할 → 총 10×1,000건 = 10,000건

slice_days = math.ceil((END_DATE - START_DATE).days / SLICES)

def fetch_meta_page(query: str, start: int, display: int = PER_PAGE):
    """네이버 뉴스 메타 수집(페이지 단위)"""
    global request_count, limit_exceeded
    if request_count >= MAX_REQUESTS:
        limit_exceeded = True
        return []
    url = (
        "https://openapi.naver.com/v1/search/news.json"
        f"?query={quote(query)}&display={display}&start={start}&sort=date"
    )
    headers = {
        "X-Naver-Client-Id":     CLIENT_ID,
        "X-Naver-Client-Secret": CLIENT_SECRET,
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 429:
            limit_exceeded = True
            print("🛑 HTTP 429: API 한도 초과")
            return []
        resp.raise_for_status()
        request_count += 1
        return resp.json().get("items", [])
    except requests.exceptions.RequestException as e:
        print("⚠️ fetch_meta_page 오류:", e)
        return []

def fetch_headline(url: str, retries: int = 2):
    """1) newspaper3k로 <title> 시도
       2) 이상하면 BeautifulSoup <h1> fallback"""
    for _ in range(retries):
        try:
            art = Article(url, language="ko", config=config)
            art.download(); art.parse()
            title = art.title.strip()
            # 사이트명만 나올 경우 예외 처리
            if len(title) > 5 and "Capital Markets" not in title:
                return title
        except Exception:
            pass
        time.sleep(1)
    # fallback: <h1> 태그 추출
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            return h1.get_text(strip=True)
    except Exception:
        pass
    return None

def save_to_mongo(item, headline, kw, adjusted_date):
    url = item.get("originallink") or item.get("link")
    doc = {
        "source":       kw,
        "url":          url,
        "pubDate":      item.get("pubDate"),
        "adjustedDate": adjusted_date.isoformat(),
        "headline":     headline,
        "fetchedAt":    datetime.utcnow().isoformat()
    }
    try:
        col.insert_one(doc)
        return True
    except errors.DuplicateKeyError:
        return False

def run_news_collector():
    global request_count, limit_exceeded
    total_saved = 0

    # 1) 날짜 범위를 SLICES개로 쪼갭니다
    slices = []
    for i in range(SLICES):
        s = START_DATE + timedelta(days=i*slice_days)
        e = s + timedelta(days=slice_days-1)
        if e > END_DATE: e = END_DATE
        slices.append((s.date(), e.date()))

    # 2) 각 키워드마다 10,000건 시도
    for kw in KEYWORDS:
        if limit_exceeded: break
        print(f"\n▶ 키워드 '{kw}' 최대 10,000건 수집 시작")
        saved = 0
        for (sd, ed) in slices:
            if limit_exceeded: break
            date_filter = f"{sd}..{ed}"
            query = f"{kw} {date_filter}"
            for p in range(PAGES):
                if limit_exceeded: break
                start_idx = p * PER_PAGE + 1
                items = fetch_meta_page(query, start_idx)
                print(f"[{kw}] {sd}~{ed} 페이지 {p+1}/{PAGES} → {len(items)}건")
                for item in items:
                    try:
                        pub_dt = datetime.strptime(
                            item["pubDate"], "%a, %d %b %Y %H:%M:%S %z"
                        )
                    except Exception:
                        continue
                    market_close = pub_dt.replace(hour=15, minute=30, second=0)
                    adjusted = (pub_dt + timedelta(days=1)).date() \
                                if pub_dt > market_close else pub_dt.date()
                    hl = fetch_headline(item.get("link"))
                    if hl and save_to_mongo(item, hl, kw, adjusted):
                        saved += 1
                        total_saved += 1
                time.sleep(1)
        print(f"✔ 키워드 '{kw}' 저장 완료: {saved}건")

    print(f"\n✅ 전체 저장된 뉴스: {total_saved}건  (요청 횟수: {request_count})")

if __name__ == "__main__":
    run_news_collector()