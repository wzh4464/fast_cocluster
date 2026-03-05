#!/usr/bin/env python3
"""
Ablation study: Compare 5 merging strategies for DiMergeCo.

Strategies:
  1. Hierarchical   — binary tree merge with consensus (default DiMergeCo)
  2. Centralized    — collect all local co-clusters, build membership matrix, k-means
  3. Union          — take union of all local co-clusters, deduplicate, k-means
  4. Random-pair    — randomly pair blocks and merge, repeat until one cluster set
  5. Greedy-overlap — greedily merge most overlapping cluster pairs

Usage:
    python run_merge_ablation.py [--dataset classic4] [--seeds 0,1,2,...,9]
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
from sklearn.cluster import KMeans

BASELINES_DIR = Path(__file__).resolve().parent.parent / "baselines"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = BASELINES_DIR / "results"

if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))


def load_dataset(name):
    if name == "classic4":
        X = np.load(DATA_DIR / "classic4_paper.npy")
        labels = np.load(DATA_DIR / "classic4_paper_labels.npy")
    elif name == "bcw":
        bcw_path = DATA_DIR / "bcw.npy"
        if not bcw_path.exists():
            from sklearn.datasets import load_breast_cancer
            data = load_breast_cancer()
            np.save(bcw_path, data.data.astype(np.float64))
            np.save(DATA_DIR / "bcw_labels.npy", data.target.astype(np.int64))
        X = np.load(bcw_path)
        labels = np.load(DATA_DIR / "bcw_labels.npy")
    else:
        raise ValueError(f"Unknown dataset: {name}")
    print(f"Loaded {name}: {X.shape}, {len(np.unique(labels))} classes")
    return X, labels


# ── Atom co-clustering (SCC-Dhillon) ─────────────────────────────────────────

def atom_scc(X_sub, k, seed):
    """SCC on a submatrix, returns row labels."""
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import svds
    from sklearn.preprocessing import normalize

    X_sp = csr_matrix(X_sub)
    eps = 1e-12
    rs = np.asarray(X_sp.sum(axis=1)).ravel()
    cs = np.asarray(X_sp.sum(axis=0)).ravel()
    du = np.where(np.abs(rs) < eps, 0.0, rs ** -0.5)
    dv = np.where(np.abs(cs) < eps, 0.0, cs ** -0.5)
    An = diags(du) @ X_sp @ diags(dv)

    n_svs = min(k, min(An.shape) - 1)
    if n_svs < 1:
        return np.zeros(X_sub.shape[0], dtype=int)

    U, s, Vt = svds(An, k=n_svs)
    Z = np.vstack([U, Vt.T])
    Z = normalize(Z, norm='l2', axis=1)
    km = KMeans(n_clusters=k, n_init=5, random_state=seed)
    all_labels = km.fit_predict(Z)
    return all_labels[:X_sub.shape[0]]


# ── Partition phase (shared by all strategies) ────────────────────────────────

def random_partition_and_cluster(X, k, m_blocks, n_blocks, t_p, seed):
    """Run T_p random partitions, each with m*n sub-problems.
    Returns list of (global_row_indices, local_labels) tuples.
    """
    rng = np.random.RandomState(seed)
    n_rows, n_cols = X.shape
    all_local_results = []  # list of (row_indices, labels_array)

    for t in range(t_p):
        row_perm = rng.permutation(n_rows)
        col_perm = rng.permutation(n_cols)
        row_groups = np.array_split(row_perm, m_blocks)
        col_groups = np.array_split(col_perm, n_blocks)

        for rg in row_groups:
            rg_sorted = np.sort(rg)
            # Use all columns (just vary row partition for simplicity in Python)
            for cg in col_groups:
                cg_sorted = np.sort(cg)
                X_sub = X[np.ix_(rg_sorted, cg_sorted)]
                if X_sub.shape[0] < k or X_sub.shape[1] < k:
                    continue
                try:
                    local_labels = atom_scc(X_sub, k, seed + t)
                    all_local_results.append((rg_sorted, local_labels))
                except Exception:
                    pass

    return all_local_results


# ── Merging strategies ────────────────────────────────────────────────────────

def merge_centralized(local_results, n_rows, k, seed):
    """Strategy 1: Centralized — build full membership matrix, k-means."""
    n_coclusters = sum(len(np.unique(lab)) for _, lab in local_results)
    membership = np.zeros((n_rows, n_coclusters), dtype=np.float32)

    col_offset = 0
    for row_idx, labels in local_results:
        unique_labels = np.unique(labels)
        for lab in unique_labels:
            mask = labels == lab
            membership[row_idx[mask], col_offset] = 1.0
            col_offset += 1

    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    return km.fit_predict(membership)


def merge_hierarchical(local_results, n_rows, k, seed):
    """Strategy 2: Hierarchical — binary tree pairwise merge."""
    if not local_results:
        return np.zeros(n_rows, dtype=int)

    # Build membership matrices for each local result
    memberships = []
    for row_idx, labels in local_results:
        n_local_k = len(np.unique(labels))
        M = np.zeros((n_rows, n_local_k), dtype=np.float32)
        for i, lab in enumerate(np.unique(labels)):
            mask = labels == lab
            M[row_idx[mask], i] = 1.0
        memberships.append(M)

    # Binary tree merge: pairwise concatenate and reduce with k-means
    rng = np.random.RandomState(seed)
    level = 0
    while len(memberships) > 1:
        next_level = []
        indices = list(range(len(memberships)))
        rng.shuffle(indices)
        for i in range(0, len(indices) - 1, 2):
            M_merged = np.hstack([memberships[indices[i]], memberships[indices[i+1]]])
            # Reduce dimensionality: k-means on merged membership
            km = KMeans(n_clusters=k, n_init=3, random_state=seed + level)
            cluster_labels = km.fit_predict(M_merged)
            # Convert back to membership
            M_reduced = np.zeros((n_rows, k), dtype=np.float32)
            for c in range(k):
                M_reduced[cluster_labels == c, c] = 1.0
            next_level.append(M_reduced)
        if len(indices) % 2 == 1:
            next_level.append(memberships[indices[-1]])
        memberships = next_level
        level += 1

    # Final k-means on the last membership
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    return km.fit_predict(memberships[0])


def merge_union(local_results, n_rows, k, seed):
    """Strategy 3: Union — concatenate all local memberships, k-means."""
    all_memberships = []
    for row_idx, labels in local_results:
        unique_labels = np.unique(labels)
        M = np.zeros((n_rows, len(unique_labels)), dtype=np.float32)
        for i, lab in enumerate(unique_labels):
            mask = labels == lab
            M[row_idx[mask], i] = 1.0
        all_memberships.append(M)

    if not all_memberships:
        return np.zeros(n_rows, dtype=int)

    M_full = np.hstack(all_memberships)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    return km.fit_predict(M_full)


def merge_random_pair(local_results, n_rows, k, seed):
    """Strategy 4: Random pair — randomly pair and merge until one remains."""
    if not local_results:
        return np.zeros(n_rows, dtype=int)

    rng = np.random.RandomState(seed)
    # Start with simple vote-counting per local result
    vote_matrices = []
    for row_idx, labels in local_results:
        votes = np.zeros((n_rows, k), dtype=np.float32)
        for lab in range(k):
            mask = labels == lab
            if mask.any():
                votes[row_idx[mask], lab] += 1.0
        vote_matrices.append(votes)

    # Randomly pair and average
    while len(vote_matrices) > 1:
        indices = list(range(len(vote_matrices)))
        rng.shuffle(indices)
        next_level = []
        for i in range(0, len(indices) - 1, 2):
            merged = vote_matrices[indices[i]] + vote_matrices[indices[i+1]]
            next_level.append(merged)
        if len(indices) % 2 == 1:
            next_level.append(vote_matrices[indices[-1]])
        vote_matrices = next_level

    return np.argmax(vote_matrices[0], axis=1)


def merge_greedy_overlap(local_results, n_rows, k, seed):
    """Strategy 5: Greedy overlap — merge most overlapping clusters greedily."""
    if not local_results:
        return np.zeros(n_rows, dtype=int)

    # Build per-row cluster assignment sets
    all_memberships = []
    for row_idx, labels in local_results:
        unique_labels = np.unique(labels)
        M = np.zeros((n_rows, len(unique_labels)), dtype=np.float32)
        for i, lab in enumerate(unique_labels):
            mask = labels == lab
            M[row_idx[mask], i] = 1.0
        all_memberships.append(M)

    M_full = np.hstack(all_memberships)
    n_total_clusters = M_full.shape[1]

    # Greedy merge: iteratively merge the two most overlapping columns
    active = list(range(n_total_clusters))

    while len(active) > k:
        # Find most overlapping pair
        best_overlap = -1
        best_pair = (0, 1)
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                overlap = np.minimum(M_full[:, active[i]], M_full[:, active[j]]).sum()
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_pair = (i, j)

        # Merge: combine columns
        i, j = best_pair
        M_full[:, active[i]] = np.maximum(M_full[:, active[i]], M_full[:, active[j]])
        active.pop(j)

    # Assign rows to nearest remaining cluster
    M_final = M_full[:, active]
    row_sums = M_final.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    M_final = M_final / row_sums
    return np.argmax(M_final, axis=1)


MERGE_STRATEGIES = {
    "Hierarchical":  merge_hierarchical,
    "Centralized":   merge_centralized,
    "Union":         merge_union,
    "Random-pair":   merge_random_pair,
    "Greedy-overlap": merge_greedy_overlap,
}


def main():
    parser = argparse.ArgumentParser(description="Merge strategy ablation")
    parser.add_argument("--dataset", type=str, default="classic4",
                        choices=["classic4", "bcw"])
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--n-clusters", type=int, default=None,
                        help="Auto: 4 for classic4, 2 for bcw")
    parser.add_argument("--m-blocks", type=int, default=2)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--t-p", type=int, default=10)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    if args.n_clusters is None:
        args.n_clusters = 2 if args.dataset == "bcw" else 4

    X, labels = load_dataset(args.dataset)

    results = []
    for strategy_name, merge_fn in MERGE_STRATEGIES.items():
        print(f"\n--- Strategy: {strategy_name} ---")
        for seed in seeds:
            print(f"  seed={seed} ...", end=" ", flush=True)
            try:
                t0 = time.time()
                # Phase 1: partition and cluster (shared)
                local_results = random_partition_and_cluster(
                    X, args.n_clusters, args.m_blocks, args.n_blocks,
                    args.t_p, seed)
                t_partition = time.time() - t0

                # Phase 2: merge (strategy-specific)
                t1 = time.time()
                pred_labels = merge_fn(
                    local_results, X.shape[0], args.n_clusters, seed)
                t_merge = time.time() - t1

                total_time = time.time() - t0
                nmi = normalized_mutual_info_score(labels, pred_labels)
                ari = adjusted_rand_score(labels, pred_labels)

                r = {
                    "strategy": strategy_name,
                    "seed": seed,
                    "nmi": round(float(nmi), 4),
                    "ari": round(float(ari), 4),
                    "time_partition_s": round(t_partition, 4),
                    "time_merge_s": round(t_merge, 4),
                    "time_total_s": round(total_time, 4),
                    "error": None,
                }
                print(f"NMI={nmi:.4f} ARI={ari:.4f} "
                      f"T_part={t_partition:.2f}s T_merge={t_merge:.2f}s")
            except Exception as e:
                import traceback; traceback.print_exc()
                r = {
                    "strategy": strategy_name,
                    "seed": seed,
                    "nmi": None, "ari": None,
                    "time_partition_s": None, "time_merge_s": None,
                    "time_total_s": None, "error": str(e),
                }
                print(f"FAILED: {e}")
            results.append(r)

    # Summary
    print("\n" + "=" * 90)
    print(f"ABLATION SUMMARY — {args.dataset} "
          f"(blocks={args.m_blocks}x{args.n_blocks}, T_p={args.t_p})")
    print("=" * 90)
    print(f"{'Strategy':<18} {'NMI':>16} {'ARI':>16} "
          f"{'Merge Time':>16} {'Total Time':>16}")
    print("-" * 86)

    for strategy_name in MERGE_STRATEGIES:
        ok = [r for r in results
              if r["strategy"] == strategy_name and r["error"] is None]
        if not ok:
            print(f"{strategy_name:<18} {'ALL FAILED':>16}")
            continue
        nmis = [r["nmi"] for r in ok]
        aris = [r["ari"] for r in ok]
        merge_times = [r["time_merge_s"] for r in ok]
        total_times = [r["time_total_s"] for r in ok]
        print(f"{strategy_name:<18} "
              f"{np.mean(nmis):.3f} +/- {np.std(nmis):.3f}  "
              f"{np.mean(aris):.3f} +/- {np.std(aris):.3f}  "
              f"{np.mean(merge_times):>7.3f} +/- {np.std(merge_times):.3f}  "
              f"{np.mean(total_times):>7.3f} +/- {np.std(total_times):.3f}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.dataset}_merge_ablation.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.datetime.now().isoformat(),
            "dataset": args.dataset,
            "n_clusters": args.n_clusters,
            "m_blocks": args.m_blocks,
            "n_blocks": args.n_blocks,
            "t_p": args.t_p,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
