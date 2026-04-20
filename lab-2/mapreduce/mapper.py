#!/usr/bin/env python3
"""
MapReduce Mapper — Win Probability by game situation.

Input:  one CSV row per line (columns: qtr,down,ydstogo,yardline_100,score_differential,win)
Output: tab-separated key/value  →  "{qtr}|{down}|{score_bucket}|{field_bucket}\t{win}\t1"

No pandas, no scikit-learn.
"""
import sys

# Column indices in the preprocessed CSV (no header in streaming input)
QTR_IDX = 0
DOWN_IDX = 1
YARD_IDX = 3       # yardline_100
SCORE_IDX = 4      # score_differential
WIN_IDX = 5


def score_diff_bucket(diff: int) -> str:
    if diff <= -14:
        return "le-14"
    if diff <= -7:
        return "-13to-7"
    if diff <= -1:
        return "-6to-1"
    if diff == 0:
        return "0"
    if diff <= 6:
        return "1to6"
    if diff <= 13:
        return "7to13"
    return "ge14"


def field_bucket(yard: int) -> str:
    if yard <= 25:
        return "0-25"
    if yard <= 50:
        return "26-50"
    if yard <= 75:
        return "51-75"
    return "76-100"


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    parts = line.split(",")
    if len(parts) < 6:
        continue  # skip header / malformed

    try:
        qtr = int(parts[QTR_IDX])
        down = int(parts[DOWN_IDX])
        yard = int(float(parts[YARD_IDX]))
        score = int(float(parts[SCORE_IDX]))
        win = int(parts[WIN_IDX])

        # Filter: only regulation plays (qtr 1-4, down 1-4)
        if not (1 <= qtr <= 4 and 1 <= down <= 4):
            continue

        key = f"{qtr}|{down}|{score_diff_bucket(score)}|{field_bucket(yard)}"
        print(f"{key}\t{win}\t1")

    except (ValueError, IndexError):
        # Header line ("qtr,down,...") and malformed rows fall here
        continue
