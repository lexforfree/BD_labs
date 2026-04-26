"""
Extract named entities and key noun phrases from articles using spaCy.
Saves: entities.pkl  {doc_id: [entity_text, ...]}
"""
import json
import os
import pickle

DATA_DIR = os.environ.get("DATA_DIR", "/data")
ARTICLES_FILE = os.path.join(DATA_DIR, "articles.jsonl")
ENTITIES_FILE = os.path.join(DATA_DIR, "entities.pkl")


def main():
    if os.path.exists(ENTITIES_FILE):
        print("[entities] entities.pkl already exists, skipping.")
        return

    import spacy
    print("[entities] Loading spaCy ru_core_news_md...")
    nlp = spacy.load("ru_core_news_md")
    nlp.max_length = 6_000_000

    articles = []
    with open(ARTICLES_FILE, encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line))
    print(f"[entities] Processing {len(articles)} articles...")

    texts = [a["title"] + ". " + a["text"][:2000] for a in articles]
    ids = [a["id"] for a in articles]

    entities: dict[str, list[str]] = {}
    for doc_id, doc in zip(ids, nlp.pipe(texts, batch_size=64, n_process=1)):
        ents: list[str] = []

        # named entities
        for ent in doc.ents:
            text = ent.text.strip().lower()
            if len(text) > 2:
                ents.append(text)

        # multi-token NOUN sequences as pseudo-phrases (noun_chunks unavailable for ru)
        i = 0
        while i < len(doc):
            if doc[i].pos_ in ("NOUN", "PROPN"):
                j = i + 1
                while j < len(doc) and doc[j].pos_ in ("NOUN", "PROPN", "ADJ"):
                    j += 1
                if j - i >= 2:
                    phrase = " ".join(t.lemma_ for t in doc[i:j]).lower()
                    if 4 < len(phrase) < 60:
                        ents.append(phrase)
                i = j
            else:
                i += 1

        entities[doc_id] = list(set(ents))

        if len(entities) % 500 == 0:
            print(f"[entities] {len(entities)}/{len(ids)}")

    with open(ENTITIES_FILE, "wb") as f:
        pickle.dump(entities, f)
    print(f"[entities] Saved {len(entities)} docs to {ENTITIES_FILE}")


if __name__ == "__main__":
    main()
