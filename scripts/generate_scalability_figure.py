#!/usr/bin/env python3
"""Generate scalability comparison figure for the paper.

Reads classic4_scalability.json and produces a dual-axis plot showing
wall-clock time and NMI across block configurations.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_scalability_data(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def aggregate_by_method(results: list[dict]) -> dict:
    """Group results by method and compute mean/std for time and NMI."""
    groups = defaultdict(list)
    for r in results:
        groups[r["method"]].append(r)

    aggregated = {}
    for method, entries in groups.items():
        times = [e["time_s"] for e in entries]
        nmis = [e["nmi"] for e in entries]
        m = entries[0]["m_blocks"]
        n = entries[0]["n_blocks"]
        aggregated[method] = {
            "m_blocks": m,
            "n_blocks": n,
            "n_subproblems": entries[0]["n_subproblems"],
            "time_mean": np.mean(times),
            "time_std": np.std(times),
            "nmi_mean": np.mean(nmis),
            "nmi_std": np.std(nmis),
        }
    return aggregated


def make_label(m: int, n: int) -> str:
    if m == 0 and n == 0:
        return r"1$\times$1"
    return rf"{m}$\times${n}"


def main():
    # --- paths ---
    base = Path(__file__).resolve().parent.parent
    data_path = base / "baselines" / "results" / "classic4_scalability.json"
    # Fallback to main repo if running from a worktree
    if not data_path.exists():
        data_path = Path("/home/jie/fast_cocluster/baselines/results/classic4_scalability.json")
    out_dir = Path("/home/jie/big-cocluster-paper/src/images")
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = out_dir / "scalability_comparison.pdf"
    png_path = out_dir / "scalability_comparison.png"

    # --- load & aggregate ---
    data = load_scalability_data(str(data_path))
    agg = aggregate_by_method(data["results"])

    # Sort by number of subproblems (1, 4, 9, 16, 25, 36, 64)
    methods_sorted = sorted(agg.keys(), key=lambda m: agg[m]["n_subproblems"])

    labels = []
    time_means = []
    time_stds = []
    nmi_means = []
    nmi_stds = []

    for method in methods_sorted:
        v = agg[method]
        labels.append(make_label(v["m_blocks"], v["n_blocks"]))
        time_means.append(v["time_mean"])
        time_stds.append(v["time_std"])
        nmi_means.append(v["nmi_mean"])
        nmi_stds.append(v["nmi_std"])

    x = np.arange(len(labels))

    # --- figure ---
    fig, ax1 = plt.subplots(figsize=(5.5, 3.5))

    # Style
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
    })
    for spine in ax1.spines.values():
        spine.set_linewidth(0.8)

    color_time = "#2166ac"
    color_nmi = "#b2182b"

    # Left axis: wall-clock time
    bar_width = 0.38
    bars = ax1.bar(
        x, time_means, bar_width,
        yerr=time_stds, capsize=3,
        color=color_time, alpha=0.75, edgecolor="white", linewidth=0.5,
        label="Wall-clock time",
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
    )
    ax1.set_xlabel("Block configuration", fontsize=10)
    ax1.set_ylabel("Wall-clock time (s)", fontsize=10, color=color_time)
    ax1.tick_params(axis="y", labelcolor=color_time, labelsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylim(bottom=0)

    # Remove grid clutter
    ax1.grid(False)
    ax1.spines["top"].set_visible(False)

    # Right axis: NMI
    ax2 = ax1.twinx()
    ax2.plot(
        x, nmi_means, "o-",
        color=color_nmi, markersize=5, linewidth=1.5,
        label="NMI", zorder=5,
    )
    ax2.fill_between(
        x,
        np.array(nmi_means) - np.array(nmi_stds),
        np.array(nmi_means) + np.array(nmi_stds),
        color=color_nmi, alpha=0.12,
    )
    ax2.set_ylabel("NMI", fontsize=10, color=color_nmi)
    ax2.tick_params(axis="y", labelcolor=color_nmi, labelsize=9)
    ax2.set_ylim(-0.05, 1.0)
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2,
        loc="upper center", frameon=True, framealpha=0.9,
        edgecolor="#cccccc", fontsize=9, ncol=2,
    )

    fig.tight_layout()

    # --- save ---
    fig.savefig(str(pdf_path), bbox_inches="tight", dpi=300)
    fig.savefig(str(png_path), bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")


if __name__ == "__main__":
    main()
