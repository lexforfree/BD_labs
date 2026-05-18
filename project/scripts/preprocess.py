"""
Merge and clean Airbnb listings from all cities.

Input:  data/raw/*.csv.gz  (one file per city from InsideAirbnb)
Output: data/processed/listings.csv  (flat file; size depends on raw exports)

Columns kept:
  id, name, description, city, price, bedrooms, bathrooms, latitude, longitude, text

`text` = the field we embed: combines name + description + location context.
"""
import csv
import gzip
import os
import re
import sys

RAW_DIR = sys.argv[1] if len(sys.argv) > 1 else "/data/raw"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "/data/processed/listings.csv"

OUTPUT_COLS = ["id", "name", "description", "city",
               "price", "bedrooms", "bathrooms",
               "latitude", "longitude", "text"]

PRICE_RE = re.compile(r"[\$,]")

# Backward compatibility for raw files produced by an earlier downloader version
# that accidentally used the InsideAirbnb export date as the output filename.
DATE_FILENAME_TO_CITY = {
    "2025-09-11": "new-orleans",
    "2025-09-16": "austin",
    "2025-09-22": "chicago",
    "2025-09-23": "nashville",
    "2025-09-25": "jersey-city",
    "2025-09-26": "columbus",
    "2025-09-28": "cambridge",
    "2025-09-29": "denver",
    "2025-11-12": "bozeman",
    "2025-12-04": "los-angeles",
    "2025-12-27": "boston",
    "2026-01-20": "dallas",
    "2026-02-15": "albany",
}


def clean_price(raw: str) -> str:
    try:
        return str(float(PRICE_RE.sub("", raw.strip())))
    except ValueError:
        return ""


def clean_text(s: str) -> str:
    return " ".join(s.replace("\n", " ").replace("\r", "").split())


def make_text(name: str, desc: str, city: str, beds: str, baths: str, price: str) -> str:
    parts = []
    if name:
        parts.append(name)
    if desc:
        parts.append(desc[:500])  # cap description length
    if city:
        parts.append(f"Located in {city}.")
    if beds:
        parts.append(f"{beds} bedrooms.")
    if baths:
        parts.append(f"{baths} bathrooms.")
    if price:
        try:
            parts.append(f"${float(price):.0f} per night.")
        except ValueError:
            pass
    return clean_text(" ".join(parts))


def city_from_filename(fname: str) -> str:
    stem = fname.replace(".csv.gz", "").replace(".csv", "")
    return DATE_FILENAME_TO_CITY.get(stem, stem)


def process_city(path: str, city_name: str, writer: csv.DictWriter) -> tuple[int, int]:
    written = skipped = 0
    open_fn = gzip.open if path.endswith(".gz") else open

    with open_fn(path, "rt", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                listing_id = row.get("id", "").strip()
                name = clean_text(row.get("name", ""))
                desc = clean_text(row.get("description", ""))
                price = clean_price(row.get("price", ""))
                beds = row.get("bedrooms", "").strip()
                baths = row.get("bathrooms", "").strip()
                lat = row.get("latitude", "").strip()
                lon = row.get("longitude", "").strip()

                if not listing_id or not lat or not lon:
                    skipped += 1
                    continue
                try:
                    float(lat)
                    float(lon)
                except ValueError:
                    skipped += 1
                    continue

                text = make_text(name, desc, city_name, beds, baths, price)
                if len(text) < 20:
                    skipped += 1
                    continue

                writer.writerow({
                    "id": listing_id,
                    "name": name,
                    "description": desc[:1000],
                    "city": city_name,
                    "price": price,
                    "bedrooms": beds,
                    "bathrooms": baths,
                    "latitude": lat,
                    "longitude": lon,
                    "text": text,
                })
                written += 1
            except Exception:
                skipped += 1
    return written, skipped


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    files = sorted(f for f in os.listdir(RAW_DIR)
                   if f.endswith(".csv.gz") or f.endswith(".csv"))
    if not files:
        print(f"No files in {RAW_DIR}", file=sys.stderr)
        sys.exit(1)

    total_written = total_skipped = 0

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLS)
        writer.writeheader()

        for fname in files:
            city = city_from_filename(fname)
            path = os.path.join(RAW_DIR, fname)
            print(f"  {city}...", end=" ", flush=True)
            w, s = process_city(path, city, writer)
            total_written += w
            total_skipped += s
            print(f"{w:,} ok  {s:,} skipped")

    size_mb = os.path.getsize(OUT_PATH) / 1024 / 1024
    print(f"\nOutput: {OUT_PATH} ({size_mb:.1f} MB)")
    print(f"Total:  {total_written:,} rows written,  {total_skipped:,} skipped")


if __name__ == "__main__":
    main()
