#!/usr/bin/env python3
"""
Preprocess NFL play-by-play CSV files for Win Probability analysis.

Extracts 6 columns from raw nflverse data (no pandas, no scikit-learn):
  qtr, down, ydstogo, yardline_100, score_differential, win

The `win` column is derived from `posteam_type` and `result`:
  - posteam_type="home" + result>0  → win=1
  - posteam_type="away" + result<0  → win=1
  - tie (result=0) or NA values     → row is skipped
"""
import sys
import csv
import gzip
import os

OUTPUT_COLUMNS = ["qtr", "down", "ydstogo", "yardline_100", "score_differential", "win"]


def process_file(input_path: str, writer: csv.DictWriter) -> tuple[int, int]:
    """Read one season CSV (possibly gzipped), write clean rows. Returns (written, skipped)."""
    written = 0
    skipped = 0

    open_fn = gzip.open if input_path.endswith(".gz") else open

    with open_fn(input_path, "rt", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                qtr = row.get("qtr", "").strip()
                down = row.get("down", "").strip()
                ydstogo = row.get("ydstogo", "").strip()
                yardline_100 = row.get("yardline_100", "").strip()
                score_diff = row.get("score_differential", "").strip()
                posteam_type = row.get("posteam_type", "").strip()
                result = row.get("result", "").strip()

                # Drop rows with missing or NA values
                if not all([qtr, down, ydstogo, yardline_100, score_diff, posteam_type, result]):
                    skipped += 1
                    continue
                if any(v == "NA" for v in [qtr, down, ydstogo, yardline_100, score_diff, result]):
                    skipped += 1
                    continue

                result_val = float(result)
                if result_val == 0.0:  # skip ties
                    skipped += 1
                    continue

                if posteam_type == "home":
                    win = 1 if result_val > 0 else 0
                elif posteam_type == "away":
                    win = 1 if result_val < 0 else 0
                else:
                    skipped += 1
                    continue

                writer.writerow(
                    {
                        "qtr": int(float(qtr)),
                        "down": int(float(down)),
                        "ydstogo": float(ydstogo),
                        "yardline_100": int(float(yardline_100)),
                        "score_differential": int(float(score_diff)),
                        "win": win,
                    }
                )
                written += 1

            except (ValueError, TypeError):
                skipped += 1
                continue

    return written, skipped


def main():
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "/data/raw"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/data/processed/nfl_all.csv"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    total_written = 0
    total_skipped = 0

    files = sorted(
        f for f in os.listdir(raw_dir) if f.endswith(".csv.gz") or f.endswith(".csv")
    )
    if not files:
        print(f"No CSV files found in {raw_dir}", file=sys.stderr)
        sys.exit(1)

    with open(out_path, "w", newline="") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for fname in files:
            in_path = os.path.join(raw_dir, fname)
            print(f"Processing {fname}...", flush=True)
            written, skipped = process_file(in_path, writer)
            total_written += written
            total_skipped += skipped
            print(f"  → written: {written:,}  skipped: {skipped:,}", flush=True)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\nOutput: {out_path} ({size_mb:.1f} MB)")
    print(f"Total rows: {total_written:,}  skipped: {total_skipped:,}")


if __name__ == "__main__":
    main()
