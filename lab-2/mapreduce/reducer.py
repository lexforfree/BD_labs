#!/usr/bin/env python3
"""
MapReduce Reducer — aggregate wins/total per game situation bucket.

Input:  sorted tab-separated lines  →  "{key}\t{win}\t1"
Output: one JSON object per bucket  →  {"qtr":1, "down":1, ..., "win_probability":0.72}

No pandas, no scikit-learn.
"""
import sys
import json

MIN_SAMPLE = 10  # discard buckets with fewer plays (low statistical confidence)


def parse_key(key: str) -> dict:
    qtr, down, score_bucket, field_bucket = key.split("|")
    return {
        "qtr": int(qtr),
        "down": int(down),
        "score_diff_bucket": score_bucket,
        "field_bucket": field_bucket,
    }


def emit(key: str, wins: int, total: int) -> None:
    if total < MIN_SAMPLE:
        return
    out = {
        **parse_key(key),
        "wins": wins,
        "total": total,
        "win_probability": round(wins / total, 4),
    }
    print(json.dumps(out))


current_key = None
total_wins = 0
total_count = 0

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    try:
        key, win_str, count_str = line.split("\t")
        win = int(win_str)
        count = int(count_str)
    except (ValueError, IndexError):
        continue

    if key == current_key:
        total_wins += win
        total_count += count
    else:
        if current_key is not None:
            emit(current_key, total_wins, total_count)
        current_key = key
        total_wins = win
        total_count = count

if current_key is not None:
    emit(current_key, total_wins, total_count)
