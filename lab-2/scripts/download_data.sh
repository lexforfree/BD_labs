#!/bin/bash
# Download NFL play-by-play data (2014-2023) from nflverse GitHub releases.
# Output: data/raw/play_by_play_{year}.csv.gz
set -e

RAW_DIR="${1:-$(dirname "$0")/../data/raw}"
BASE_URL="https://github.com/nflverse/nflverse-data/releases/download/pbp"
START_YEAR="${START_YEAR:-2014}"
END_YEAR="${END_YEAR:-2025}"

mkdir -p "$RAW_DIR"

echo "Downloading NFL PBP data ${START_YEAR}–${END_YEAR} into ${RAW_DIR}"
echo "Total size estimate: ~3 GB"
echo ""

for year in $(seq "$START_YEAR" "$END_YEAR"); do
    fname="play_by_play_${year}.csv.gz"
    dest="${RAW_DIR}/${fname}"

    if [ -f "$dest" ]; then
        echo "[SKIP] ${fname} already exists"
        continue
    fi

    echo "[DL]   ${fname} ..."
    wget -q --show-progress \
        -O "${dest}.tmp" \
        "${BASE_URL}/${fname}" && \
        mv "${dest}.tmp" "$dest" || {
        echo "[FAIL] ${fname} — removing partial file"
        rm -f "${dest}.tmp"
    }
done

echo ""
echo "Done. Files in ${RAW_DIR}:"
ls -lh "$RAW_DIR"
