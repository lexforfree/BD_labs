"""
NFL Win Probability Dashboard — Flask app.

Endpoints:
  GET /                      — HTML dashboard
  GET /api/timing            — timing comparison JSON
  GET /api/results/<tool>    — win probability results (mr | hive | spark)
"""
import json
import os

from flask import Flask, jsonify, render_template, send_file

app = Flask(__name__)
RESULTS_DIR = os.environ.get("RESULTS_DIR", "/results")


def load_json(filename: str):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@app.route("/")
def index():
    timing = load_json("timing_comparison.json") or {}
    available = {
        "mr": os.path.exists(os.path.join(RESULTS_DIR, "mr_result.json")),
        "hive": os.path.exists(os.path.join(RESULTS_DIR, "hive_result.json")),
        "spark": os.path.exists(os.path.join(RESULTS_DIR, "spark_result.json")),
    }
    return render_template("index.html", timing=timing, available=available)


@app.route("/api/timing")
def timing():
    data = load_json("timing_comparison.json") or {}
    return jsonify(data)


@app.route("/chart/<name>")
def chart(name: str):
    allowed = {"heatmap": "wp_heatmap.png", "timing": "timing_comparison.png"}
    fname = allowed.get(name)
    if not fname:
        return "Not found", 404
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        return "Chart not generated yet", 404
    return send_file(path, mimetype="image/png")


@app.route("/api/results/<tool>")
def results(tool: str):
    name_map = {"mr": "mr_result.json", "hive": "hive_result.json", "spark": "spark_result.json"}
    fname = name_map.get(tool)
    if not fname:
        return jsonify({"error": "unknown tool"}), 404
    data = load_json(fname)
    if data is None:
        return jsonify({"error": "results not available yet — run the pipeline first"}), 404
    return jsonify(data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
