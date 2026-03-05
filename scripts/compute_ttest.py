#!/usr/bin/env python3
"""Paired t-test verification: DiMergeCo-SCC vs each baseline.

Loads NMI scores per seed from JSON result files for Classic4 and BCW datasets,
runs scipy.stats.ttest_rel (paired) between DiMergeCo-SCC and each baseline,
prints a summary table, and saves results to baselines/results/ttest_results.json.
"""

import json
import os
from pathlib import Path
from scipy import stats
import numpy as np

# Resolve paths relative to this script's parent (repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "baselines" / "results"


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_nmi_by_method(data: dict) -> dict[str, list[float]]:
    """Return {method_name: [nmi_seed0, nmi_seed1, ...]} sorted by seed."""
    buckets: dict[str, list[tuple[int, float]]] = {}
    for r in data["results"]:
        method = r["method"]
        buckets.setdefault(method, []).append((r["seed"], r["nmi"]))
    # Sort by seed and return just the NMI values
    return {m: [nmi for _, nmi in sorted(pairs)] for m, pairs in buckets.items()}


def find_dimerge_scc(nmi_map: dict[str, list[float]]) -> tuple[str, list[float]]:
    """Find the DiMergeCo+SCC method (contains 'DiMerge' and 'SCC')."""
    for name, scores in nmi_map.items():
        if "DiMerge" in name and "SCC" in name:
            return name, scores
    raise KeyError("No DiMergeCo+SCC method found")


def run_paired_ttests(dataset_name: str,
                      baseline_data: dict,
                      dimerge_data: dict) -> list[dict]:
    """Run paired t-tests between DiMergeCo-SCC and each baseline."""
    baseline_nmi = extract_nmi_by_method(baseline_data)
    dimerge_nmi = extract_nmi_by_method(dimerge_data)

    dm_name, dm_scores = find_dimerge_scc(dimerge_nmi)
    dm_arr = np.array(dm_scores)

    results = []
    for bl_name, bl_scores in baseline_nmi.items():
        bl_arr = np.array(bl_scores)
        # Ensure same number of seeds
        n = min(len(dm_arr), len(bl_arr))
        t_stat, p_value = stats.ttest_rel(dm_arr[:n], bl_arr[:n])
        mean_diff = float(np.mean(dm_arr[:n]) - np.mean(bl_arr[:n]))
        results.append({
            "dataset": dataset_name,
            "dimerge_method": dm_name,
            "baseline_method": bl_name,
            "dimerge_mean_nmi": float(np.mean(dm_arr[:n])),
            "baseline_mean_nmi": float(np.mean(bl_arr[:n])),
            "mean_diff": mean_diff,
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant_0.01": bool(p_value < 0.01),
            "dimerge_wins": bool(mean_diff > 0),
            "n_pairs": n,
        })
    return results


def main():
    # Load all JSON files
    classic4_baselines = load_json(RESULTS_DIR / "classic4_baselines.json")
    classic4_dimerge = load_json(RESULTS_DIR / "classic4_dimerge_co_variants.json")
    bcw_baselines = load_json(RESULTS_DIR / "bcw_baselines.json")
    bcw_dimerge = load_json(RESULTS_DIR / "bcw_dimerge_co_variants.json")

    all_results = []
    all_results.extend(run_paired_ttests("Classic4", classic4_baselines, classic4_dimerge))
    all_results.extend(run_paired_ttests("BCW", bcw_baselines, bcw_dimerge))

    # Print summary table
    header = f"{'Dataset':<10} {'Baseline':<20} {'DiMergeCo NMI':>14} {'Baseline NMI':>13} {'Diff':>8} {'t-stat':>8} {'p-value':>12} {'p<0.01':>7} {'Winner':>12}"
    print("=" * len(header))
    print("Paired t-test: DiMergeCo-SCC vs. each baseline (NMI)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in all_results:
        if not r["significant_0.01"]:
            winner = "non-significant"
        elif r["dimerge_wins"]:
            winner = "DiMergeCo"
        else:
            winner = r["baseline_method"]
        sig = "YES" if r["significant_0.01"] else "no"
        print(
            f"{r['dataset']:<10} "
            f"{r['baseline_method']:<20} "
            f"{r['dimerge_mean_nmi']:>14.4f} "
            f"{r['baseline_mean_nmi']:>13.4f} "
            f"{r['mean_diff']:>+8.4f} "
            f"{r['t_statistic']:>8.3f} "
            f"{r['p_value']:>12.2e} "
            f"{sig:>7} "
            f"{winner:>12}"
        )

    print("-" * len(header))
    print()

    # Summary interpretation
    print("INTERPRETATION:")
    for dataset in ["Classic4", "BCW"]:
        ds_results = [r for r in all_results if r["dataset"] == dataset]
        wins = [r for r in ds_results if r["dimerge_wins"] and r["significant_0.01"]]
        losses = [r for r in ds_results if not r["dimerge_wins"] and r["significant_0.01"]]
        print(f"  {dataset}:")
        if wins:
            print(f"    DiMergeCo-SCC significantly beats: {', '.join(r['baseline_method'] for r in wins)}")
        if losses:
            print(f"    DiMergeCo-SCC significantly loses to: {', '.join(r['baseline_method'] for r in losses)}")
        no_sig = [r for r in ds_results if not r["significant_0.01"]]
        if no_sig:
            print(f"    No significant difference with: {', '.join(r['baseline_method'] for r in no_sig)}")

    # Save results
    output_path = RESULTS_DIR / "ttest_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
