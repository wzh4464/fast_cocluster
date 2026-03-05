#!/usr/bin/env python3
"""
Scalability comparison: DiMergeCo-SCC vs standalone SCC.

Measures wall-clock time as a function of subproblem parallelism
(simulated by varying m_blocks x n_blocks) on Classic4 and RCV1-train.

For the paper's scalability figure (R1-RC2): demonstrates O(log n)
communication advantage of DiMergeCo over centralized methods.

Usage:
    python run_scalability.py [--dataset classic4] [--seeds 0,1,2]
    python run_scalability.py --dataset rcv1
"""

import sys
import os
import time
import json
import argparse
import datetime
import numpy as np
from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

BASELINES_DIR = Path(__file__).resolve().parent.parent / "baselines"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = BASELINES_DIR / "results"

# Add baselines to path for imports
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))


def load_classic4():
    X = np.load(DATA_DIR / "classic4_paper.npy")
    labels = np.load(DATA_DIR / "classic4_paper_labels.npy")
    print(f"Loaded Classic4: {X.shape}")
    return X, labels


def load_rcv1():
    from sklearn.datasets import fetch_rcv1
    rcv1 = fetch_rcv1(subset="train")
    X_full = rcv1.data
    targets = rcv1.target
    top_cats = ["CCAT", "ECAT", "GCAT", "MCAT"]
    cat_indices = [list(rcv1.target_names).index(c) for c in top_cats]
    mask = np.asarray(targets[:, cat_indices].sum(axis=1)).ravel() > 0
    X = X_full[mask]
    labels = np.asarray(targets[mask][:, cat_indices].argmax(axis=1)).ravel()
    print(f"Loaded RCV1-train: {X.shape}")
    return X, labels


def run_scc_baseline(X, n_clusters, seed):
    """Standalone SCC (Dhillon 2001)."""
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import svds
    from sklearn.preprocessing import normalize
    from sklearn.cluster import KMeans

    X_sp = csr_matrix(X) if not hasattr(X, 'toarray') else X
    eps = 1e-12
    row_sums = np.asarray(X_sp.sum(axis=1)).ravel()
    col_sums = np.asarray(X_sp.sum(axis=0)).ravel()
    du_inv = np.where(np.abs(row_sums) < eps, 0.0, row_sums ** -0.5)
    dv_inv = np.where(np.abs(col_sums) < eps, 0.0, col_sums ** -0.5)
    An = diags(du_inv) @ X_sp @ diags(dv_inv)

    t0 = time.time()
    U, s, Vt = svds(An, k=n_clusters)
    Z = np.vstack([U, Vt.T])
    Z = normalize(Z, norm='l2', axis=1)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    all_labels = km.fit_predict(Z)
    elapsed = time.time() - t0
    return all_labels[:X.shape[0]], elapsed


def run_dimerge_scc(X, n_clusters, m_blocks, n_blocks, t_p, seed):
    """DiMergeCo with SCC atom."""
    # Import DiMergeCo from the baselines module
    from run_dimerge_co_variants import DiMergeCo, atom_scc_dhillon

    dmc = DiMergeCo(
        atom_fn=atom_scc_dhillon,
        k=n_clusters,
        m_blocks=m_blocks,
        n_blocks=n_blocks,
        t_p=t_p,
    )
    pred_labels, elapsed = dmc.run(X, seed=seed)
    return pred_labels, elapsed


def main():
    parser = argparse.ArgumentParser(description="Scalability comparison")
    parser.add_argument("--dataset", type=str, default="classic4",
                        choices=["classic4", "rcv1"])
    parser.add_argument("--seeds", type=str, default="0,1,2",
                        help="Comma-separated random seeds")
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--t-p", type=int, default=10)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    if args.dataset == "classic4":
        X, labels = load_classic4()
    else:
        X, labels = load_rcv1()

    # Different parallelism levels (number of subproblems = m * n)
    configs = [
        (1, 1),   # 1 subproblem (= centralized)
        (2, 2),   # 4 subproblems
        (3, 3),   # 9 subproblems
        (4, 4),   # 16 subproblems
        (5, 5),   # 25 subproblems
        (6, 6),   # 36 subproblems
        (8, 8),   # 64 subproblems
    ]

    results = []

    # Baseline (standalone SCC, no partitioning)
    print("\n--- Baseline SCC (no partitioning) ---")
    for seed in seeds:
        pred, elapsed = run_scc_baseline(X, args.n_clusters, seed)
        nmi, ari = (normalized_mutual_info_score(labels, pred),
                    adjusted_rand_score(labels, pred))
        results.append({
            "method": "SCC-baseline",
            "m_blocks": 0, "n_blocks": 0, "n_subproblems": 1,
            "seed": seed, "nmi": round(float(nmi), 4),
            "ari": round(float(ari), 4), "time_s": round(elapsed, 4),
        })
        print(f"  seed={seed}: NMI={nmi:.4f} ARI={ari:.4f} Time={elapsed:.3f}s")

    # DiMergeCo at various parallelism levels
    for m, n in configs:
        if m == 1 and n == 1:
            continue  # Skip 1x1, same as baseline
        n_sub = m * n
        print(f"\n--- DiMergeCo-SCC ({m}x{n} = {n_sub} subproblems) ---")
        for seed in seeds:
            try:
                pred, elapsed = run_dimerge_scc(
                    X, args.n_clusters, m, n, args.t_p, seed)
                nmi, ari = (normalized_mutual_info_score(labels, pred),
                            adjusted_rand_score(labels, pred))
                results.append({
                    "method": f"DiMergeCo-SCC-{m}x{n}",
                    "m_blocks": m, "n_blocks": n, "n_subproblems": n_sub,
                    "seed": seed, "nmi": round(float(nmi), 4),
                    "ari": round(float(ari), 4), "time_s": round(elapsed, 4),
                })
                print(f"  seed={seed}: NMI={nmi:.4f} ARI={ari:.4f} Time={elapsed:.3f}s")
            except Exception as e:
                print(f"  seed={seed}: FAILED - {e}")
                results.append({
                    "method": f"DiMergeCo-SCC-{m}x{n}",
                    "m_blocks": m, "n_blocks": n, "n_subproblems": n_sub,
                    "seed": seed, "nmi": None, "ari": None, "time_s": None,
                    "error": str(e),
                })

    # Summary
    print("\n" + "=" * 80)
    print("SCALABILITY SUMMARY")
    print("=" * 80)
    print(f"{'Config':<20} {'Subproblems':>12} {'NMI':>16} {'Time (s)':>16}")
    print("-" * 68)

    for key in ["SCC-baseline"] + [f"DiMergeCo-SCC-{m}x{n}" for m, n in configs if not (m == 1 and n == 1)]:
        ok = [r for r in results if r["method"] == key and r.get("error") is None]
        if not ok:
            continue
        nmis = [r["nmi"] for r in ok]
        times = [r["time_s"] for r in ok]
        n_sub = ok[0]["n_subproblems"]
        print(f"{key:<20} {n_sub:>12} "
              f"{np.mean(nmis):.3f} +/- {np.std(nmis):.3f}  "
              f"{np.mean(times):>7.3f} +/- {np.std(times):.3f}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.dataset}_scalability.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.datetime.now().isoformat(),
            "dataset": args.dataset,
            "n_clusters": args.n_clusters,
            "t_p": args.t_p,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
