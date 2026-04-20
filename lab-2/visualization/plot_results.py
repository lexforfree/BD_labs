"""
Generate static PNG charts from pipeline results.
Called by the Airflow generate_report task after each run.

Produces:
  /results/timing_comparison.png   — horizontal bar chart (MapReduce vs Hive vs Spark)
  /results/wp_heatmap.png          — win probability heatmap (score_diff × qtr, down=3)
"""
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

matplotlib.use("Agg")  # headless rendering

RESULTS_DIR = os.environ.get("RESULTS_DIR", "/results")

SCORE_BUCKETS = ["le-14", "-13to-7", "-6to-1", "0", "1to6", "7to13", "ge14"]
QTRS = [1, 2, 3, 4]


def load_json(fname: str):
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def plot_timing() -> None:
    data = load_json("timing_comparison.json")
    if not data:
        print("No timing data, skipping chart.")
        return

    labels = []
    values = []
    colors = ["#e07b39", "#5b9bd5", "#70ad47"]

    for key, label, color in [
        ("mapreduce_ms", "MapReduce", colors[0]),
        ("hive_ms", "Hive", colors[1]),
        ("spark_ms", "Spark", colors[2]),
    ]:
        if key in data:
            labels.append(label)
            values.append(data[key] / 1000)  # convert to seconds

    if not values:
        print("No timing values found.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, values, color=colors[: len(labels)], height=0.5)
    ax.bar_label(bars, fmt="{:.1f}s", padding=4)
    ax.set_xlabel("Wall-clock time (seconds)")
    ax.set_title("Win Probability Query — Execution Time Comparison")
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.2)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "timing_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_wp_heatmap(source: str = "spark") -> None:
    """Heatmap: rows=qtr, cols=score_diff_bucket, cell=win_probability for down=3, field=26-50."""
    fname_map = {"mr": "mr_result.json", "hive": "hive_result.json", "spark": "spark_result.json"}
    data = load_json(fname_map.get(source, "spark_result.json"))
    if not data:
        print(f"No {source} results, skipping heatmap.")
        return

    # Index results by (qtr, down, score_diff_bucket, field_bucket)
    index = {}
    for row in data:
        key = (row["qtr"], row["down"], row["score_diff_bucket"], row["field_bucket"])
        index[key] = row["win_probability"]

    # Build matrix: qtr × score_diff_bucket  (fixed down=3, field=26-50)
    DOWN = 3
    FIELD = "26-50"
    matrix = np.full((len(QTRS), len(SCORE_BUCKETS)), np.nan)
    for i, qtr in enumerate(QTRS):
        for j, sb in enumerate(SCORE_BUCKETS):
            wp = index.get((qtr, DOWN, sb, FIELD))
            if wp is not None:
                matrix[i, j] = wp

    fig, ax = plt.subplots(figsize=(10, 4))
    cmap = mcolors.LinearSegmentedColormap.from_list("wp", ["#c0392b", "#f5f5f5", "#2980b9"])
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(SCORE_BUCKETS)))
    ax.set_xticklabels(SCORE_BUCKETS)
    ax.set_yticks(range(len(QTRS)))
    ax.set_yticklabels([f"Q{q}" for q in QTRS])
    ax.set_xlabel("Score differential bucket (possession team)")
    ax.set_ylabel("Quarter")
    ax.set_title(f"Win Probability Heatmap — Down {DOWN}, Field {FIELD} yds  (source: {source})")

    # Annotate cells with the probability value
    for i in range(len(QTRS)):
        for j in range(len(SCORE_BUCKETS)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9,
                        color="black" if 0.3 < val < 0.7 else "white")

    plt.colorbar(im, ax=ax, label="Win Probability")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "wp_heatmap.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_timing()
    # Prefer Spark results for the heatmap (fastest/most recent)
    for src in ["spark", "hive", "mr"]:
        if load_json({"spark": "spark_result.json", "hive": "hive_result.json", "mr": "mr_result.json"}[src]):
            plot_wp_heatmap(src)
            break
    print("Done.")
