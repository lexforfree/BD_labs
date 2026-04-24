"""
Download and parse the Russian Wikipedia page-links SQL dump.
Keeps only links where both endpoints are in our article corpus.
Saves to /data/links.jsonl.
"""
import gzip
import io
import json
import os
import re
import urllib.request

DATA_DIR = os.environ.get("DATA_DIR", "/data")
ARTICLES_FILE = os.path.join(DATA_DIR, "articles.jsonl")
LINKS_FILE = os.path.join(DATA_DIR, "links.jsonl")

DUMP_URL = (
    "https://dumps.wikimedia.org/ruwiki/latest/ruwiki-latest-pagelinks.sql.gz"
)

# INSERT INTO `pagelinks` VALUES (from_id,...,target_title,...)
INSERT_RE = re.compile(
    r"\((\d+),\d+,'([^']*)',\d+\)"
)


def load_corpus_titles(path: str) -> dict[str, str]:
    """Returns {title: id} for all articles in the corpus."""
    title_to_id: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            title_to_id[rec["title"]] = rec["id"]
    return title_to_id


def load_from_id_to_title(path: str) -> dict[int, str]:
    """Returns {numeric_wiki_id: title} for articles in the corpus."""
    id_to_title: dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            try:
                id_to_title[int(rec["id"])] = rec["title"]
            except (ValueError, KeyError):
                pass
    return id_to_title


def main():
    if os.path.exists(LINKS_FILE):
        print(f"[links] {LINKS_FILE} already exists, skipping.")
        return

    title_to_id = load_corpus_titles(ARTICLES_FILE)
    from_id_map = load_from_id_to_title(ARTICLES_FILE)
    corpus_titles = set(title_to_id.keys())

    print(f"[links] Corpus: {len(corpus_titles)} articles.")
    print(f"[links] Downloading {DUMP_URL} ...")

    links: dict[str, list[str]] = {}  # from_title → [to_title, ...]
    count = 0

    with urllib.request.urlopen(DUMP_URL) as resp:
        with gzip.open(resp, "rt", encoding="utf-8", errors="replace") as gz:
            for line in gz:
                if not line.startswith("INSERT INTO"):
                    continue
                for m in INSERT_RE.finditer(line):
                    from_num = int(m.group(1))
                    to_title = m.group(2).replace("\\'", "'")

                    from_title = from_id_map.get(from_num)
                    if from_title is None or to_title not in corpus_titles:
                        continue

                    links.setdefault(from_title, [])
                    if to_title not in links[from_title]:
                        links[from_title].append(to_title)
                        count += 1

    print(f"[links] Found {count} intra-corpus links across {len(links)} source articles.")

    with open(LINKS_FILE, "w", encoding="utf-8") as f:
        for from_title, targets in links.items():
            record = {
                "from_id": title_to_id[from_title],
                "from_title": from_title,
                "to_ids": [title_to_id[t] for t in targets],
                "to_titles": targets,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[links] Saved to {LINKS_FILE}")


if __name__ == "__main__":
    main()
