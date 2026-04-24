"""
Full-text search over an inverted index using two traversal strategies:

DAAT (Document-At-A-Time): iterate candidate documents, check all query terms
TAAT (Term-At-A-Time):     iterate query terms, accumulate scores per document
"""
import re
from collections import defaultdict

TOKEN_RE = re.compile(r"[а-яёa-z]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def daat(
    query: str,
    index: dict[str, list[tuple[str, float]]],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """
    Document-At-A-Time: find docs that contain ALL query terms,
    then rank by sum of TF scores.
    """
    terms = list(set(tokenize(query)))
    if not terms:
        return []

    # postings lists for each query term
    postings = [dict(index.get(t, [])) for t in terms]

    # candidate docs: intersection of all postings
    candidate_sets = [set(p.keys()) for p in postings]
    candidates = candidate_sets[0]
    for s in candidate_sets[1:]:
        candidates &= s

    results: list[tuple[str, float]] = []
    for doc_id in candidates:
        score = sum(p.get(doc_id, 0.0) for p in postings)
        results.append((doc_id, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def taat(
    query: str,
    index: dict[str, list[tuple[str, float]]],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """
    Term-At-A-Time: iterate query terms one by one,
    accumulate TF contributions per document.
    """
    terms = list(set(tokenize(query)))
    if not terms:
        return []

    scores: dict[str, float] = defaultdict(float)
    for term in terms:
        for doc_id, tf in index.get(term, []):
            scores[doc_id] += tf

    results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return results[:top_k]
