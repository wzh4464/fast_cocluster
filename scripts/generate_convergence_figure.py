#!/usr/bin/env python3
"""Generate convergence figure for the hierarchical merge.

Reads merge_round_metrics from a JSON file (produced by a dedicated JSON
output mode of DiMergeCo or by post-processing captured per-round log lines
into JSON) and plots the objective J (avg merge score) and cluster count
vs merge round.

Usage:
    python scripts/generate_convergence_figure.py <metrics_json>

The input JSON should have the shape:
    {
      "merge_round_metrics": [
        {"round": 0, "num_clusters": 64, "avg_merge_score": 0.0, "num_merges": 0},
        {"round": 1, "num_clusters": 48, "avg_merge_score": 0.82, "num_merges": 16},
        ...
      ]
    }
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if "merge_round_metrics" in data:
        return data["merge_round_metrics"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Cannot find merge_round_metrics in {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot hierarchical merge convergence")
    parser.add_argument("metrics_json", help="Path to JSON with merge_round_metrics")
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory for figures (default: same dir as input)",
    )
    args = parser.parse_args()

    metrics = load_metrics(args.metrics_json)
    if not metrics:
        print("No merge round metrics found.", file=sys.stderr)
        sys.exit(1)

    # Ensure metrics are ordered by round so the x-axis is monotonically increasing
    metrics = sorted(metrics, key=lambda m: m["round"])

    rounds = [m["round"] for m in metrics]
    num_clusters = [m["num_clusters"] for m in metrics]
    avg_scores = [m["avg_merge_score"] for m in metrics]

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.metrics_json).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- figure ---
    fig, ax1 = plt.subplots(figsize=(5.5, 3.5))

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
    })
    for spine in ax1.spines.values():
        spine.set_linewidth(0.8)

    color_j = "#2166ac"
    color_k = "#b2182b"

    # Left axis: objective J (avg merge score)
    ax1.plot(rounds, avg_scores, "s-", color=color_j, markersize=5, linewidth=1.5, label="Avg merge score (J)")
    ax1.set_xlabel("Merge round (tree level)", fontsize=10)
    ax1.set_ylabel("Avg merge score (J)", fontsize=10, color=color_j)
    ax1.tick_params(axis="y", labelcolor=color_j, labelsize=9)
    ax1.set_xticks(rounds)
    ax1.grid(False)
    ax1.spines["top"].set_visible(False)

    # Right axis: cluster count
    ax2 = ax1.twinx()
    ax2.plot(rounds, num_clusters, "o--", color=color_k, markersize=5, linewidth=1.5, label="Cluster count")
    ax2.set_ylabel("Number of clusters", fontsize=10, color=color_k)
    ax2.tick_params(axis="y", labelcolor=color_k, labelsize=9)
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2,
        loc="upper right", frameon=True, framealpha=0.9,
        edgecolor="#cccccc", fontsize=9,
    )

    fig.tight_layout()

    stem = Path(args.metrics_json).stem
    pdf_path = out_dir / f"{stem}_convergence.pdf"
    png_path = out_dir / f"{stem}_convergence.png"
    fig.savefig(str(pdf_path), bbox_inches="tight", dpi=300)
    fig.savefig(str(png_path), bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")


if __name__ == "__main__":
    main()
