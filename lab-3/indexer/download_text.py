"""
Download and filter Russian Wikipedia articles on math/ML/statistics topics.
Uses streaming mode — downloads only what's needed, not the full 5 GB dump.
Saves filtered articles to /data/articles.jsonl.
"""
import json
import os
from datasets import load_dataset

DATA_DIR = os.environ.get("DATA_DIR", "/data")
OUT_FILE = os.path.join(DATA_DIR, "articles.jsonl")

MAX_ARTICLES = 10_000

KEYWORDS = [
    "матем", "статист", "машинн", "нейрон", "вероятн",
    "алгебр", "регресс", "оптимиз", "кластер", "классифик",
    "нейросет", "линейн", "градиент", "теорем", "функци",
    "дискретн", "вычислит", "алгоритм", "случайн", "дисперси",
    "ковариац", "байесов", "марков", "энтропи", "логистич",
    "метрик", "граф ", "деревь", "лес (", "бустинг",
]


def matches(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in KEYWORDS)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(OUT_FILE):
        print(f"[text] {OUT_FILE} already exists, skipping.")
        return

    print("[text] Streaming Russian Wikipedia (no full download)...")
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.ru",
        split="train",
        streaming=True,
    )

    count = 0
    scanned = 0
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for article in ds:
            scanned += 1
            if scanned % 10_000 == 0:
                print(f"[text] scanned={scanned}  found={count}")

            if not matches(article["title"]):
                continue

            record = {
                "id": article["id"],
                "title": article["title"],
                "text": article["text"][:5000],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

            if count >= MAX_ARTICLES:
                print(f"[text] Reached limit of {MAX_ARTICLES} articles.")
                break

    print(f"[text] Saved {count} articles (scanned {scanned}) to {OUT_FILE}")


if __name__ == "__main__":
    main()
