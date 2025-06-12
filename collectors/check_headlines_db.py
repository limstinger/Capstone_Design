import os
from pymongo import MongoClient
from dotenv import load_dotenv
import pprint

load_dotenv()
uri = os.getenv("MONGO_URI")
if not uri:
    raise RuntimeError("환경변수 MONGO_URI를 .env에 설정하세요")

client = MongoClient(uri)
db     = client["newsdb"]
col    = db.headlines

# 1) 전체 문서 수
total = col.count_documents({})
print(f"Total headlines stored: {total}")

# 2) 최근 5건 샘플 출력
print("\nSample 5 documents:")
for doc in col.find().sort("fetchedAt", -1).limit(5):
    # 필요한 필드만 보기
    pprint.pprint({
        "title": doc.get("title"),
        "source": doc.get("source"),
        "publishedAt": doc.get("publishedAt"),
        "fetchedAt": doc.get("fetchedAt"),
        "url": doc.get("url")
    })
