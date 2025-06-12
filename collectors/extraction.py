import os
import csv
from pymongo import MongoClient
from dotenv import load_dotenv

# ─── 환경 변수 로드 ─────────────────────────
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI 환경변수 필요")

# ─── MongoDB 연결 ──────────────────────────
client = MongoClient(MONGO_URI)
db     = client["newsdb"]
col    = db.articles

# ─── CSV로 내보낼 필드 순서 지정 ────────────
fields = [
    "source", "title", "url",
    "pubDate", "adjustedDate", "headline", "fetchedAt"
]

# ─── CSV 쓰기 ──────────────────────────────
with open("articles_export.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for doc in col.find({}, {k: 1 for k in fields}):
        # ObjectId _id 필드 제거
        row = {k: doc.get(k, "") for k in fields}
        writer.writerow(row)

print("✅ articles_export.csv 생성 완료")