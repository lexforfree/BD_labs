#!/bin/bash
# Download Airbnb listings from InsideAirbnb (no auth required).
# Combines US city exports with descriptions and lat/lon.
# URL list maintained May 2026; each city has its own update date.
set -e

RAW_DIR="${1:-$(dirname "$0")/../data/raw}"
mkdir -p "$RAW_DIR"

# Format: "country/state-or-region/city/YYYY-MM-DD"
CITIES=(
  "united-states/tx/austin/2025-09-16"
  "united-states/ma/boston/2025-12-27"
  "united-states/il/chicago/2025-09-22"
  "united-states/tx/dallas/2026-01-20"
  "united-states/co/denver/2025-09-29"
  "united-states/hi/hawaii/2025-09-16"
  "united-states/nj/jersey-city/2025-09-25"
  "united-states/ca/los-angeles/2025-12-04"
  "united-states/tn/nashville/2025-09-23"
  "united-states/la/new-orleans/2025-09-11"
  "united-states/nc/asheville/2025-09-22"
  "united-states/oh/columbus/2025-09-26"
  "united-states/nv/clark-county-nv/2025-09-23"
  "united-states/tx/fort-worth/2025-09-16"
  "united-states/fl/broward-county/2025-09-26"
  "united-states/ma/cambridge/2025-09-28"
  "united-states/mt/bozeman/2025-11-12"
  "united-states/ny/albany/2026-02-15"
)

BASE="https://data.insideairbnb.com"

echo "Downloading ${#CITIES[@]} cities into ${RAW_DIR}"
echo ""

for city_path in "${CITIES[@]}"; do
    city_name=$(echo "$city_path" | awk -F'/' '{print $3}')
    out="${RAW_DIR}/${city_name}.csv.gz"

    if [ -f "$out" ]; then
        echo "[SKIP] ${city_name}"
        continue
    fi

    url="${BASE}/${city_path}/data/listings.csv.gz"
    echo "[DL]  ${city_name} ← ${url}"
    wget -q --show-progress -O "${out}.tmp" "$url" && mv "${out}.tmp" "$out" || {
        echo "[FAIL] ${city_name} — removing partial"
        rm -f "${out}.tmp"
    }
done

echo ""
echo "Done. Files in ${RAW_DIR}:"
ls -lh "$RAW_DIR"
echo ""
echo "Total size: $(du -sh "$RAW_DIR" | cut -f1)"
