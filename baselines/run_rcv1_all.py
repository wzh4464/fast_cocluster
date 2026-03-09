#!/usr/bin/env python3
"""Run baselines + DiMergeCo on full RCV1-all (804K x 47K, sparse).

Demonstrates DiMergeCo's scalability advantage on a large dataset.
Only methods supporting sparse input are tested:
  - SCC (SpectralCoclustering) — standalone and DiMergeCo
  - FNMF (sklearn NMF) — standalone and DiMergeCo

NMF baselines (PNMTF, ONMTF, NBVD, ONM3F) require dense matrices and
are infeasible at this scale (804K x 47K dense = 304 GB).

Usage:
    python baselines/run_rcv1_all.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.cluster import SpectralCoclustering
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).parent))
from run_dimerge_co_variants import DiMergeCo, random_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "rcv1"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_rcv1_all():
    X = sparse.load_npz(str(DATA_DIR / "rcv1_all_sparse.npz"))
    labels = np.load(str(DATA_DIR / "rcv1_all_sparse_labels.npy"))
    return X, labels


def fnmf_atom(X_sub, k, seed=0):
    """FNMF atom for DiMergeCo."""
    nmf = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=100)
    W = nmf.fit_transform(X_sub)
    return np.argmax(W, axis=1)


def scc_atom(X_sub, k, seed=0):
    """SCC atom for DiMergeCo."""
    model = SpectralCoclustering(
        n_clusters=k, random_state=seed, svd_method="randomized"
    )
    model.fit(X_sub)
    return model.row_labels_


def run_fnmf(X, k, seed=0):
    nmf = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=100)
    t0 = time.time()
    W = nmf.fit_transform(X)
    elapsed = time.time() - t0
    pred = np.argmax(W, axis=1)
    return pred, elapsed


def run_scc(X, k, seed=0):
    model = SpectralCoclustering(
        n_clusters=k, random_state=seed, svd_method="randomized"
    )
    t0 = time.time()
    model.fit(X)
    elapsed = time.time() - t0
    return model.row_labels_, elapsed


def main():
    print("Loading RCV1-all sparse dataset...")
    X, labels = load_rcv1_all()
    k = 4
    print(f"Shape: {X.shape}, NNZ: {X.nnz:,}, k={k}")

    seeds = list(range(3))  # 3 seeds for large dataset
    results = []

    # --- Standalone FNMF ---
    print("\n=== FNMF (standalone) ===")
    for seed in seeds:
        print(f"  seed={seed}...", end=" ", flush=True)
        try:
            pred, elapsed = run_fnmf(X, k, seed)
            nmi = normalized_mutual_info_score(labels, pred)
            ari = adjusted_rand_score(labels, pred)
            print(f"NMI={nmi:.4f}, ARI={ari:.4f}, Time={elapsed:.1f}s")
            results.append(dict(
                method="FNMF", seed=seed, nmi=nmi, ari=ari, time_s=elapsed, error=None
            ))
        except Exception as e:
            print(f"FAILED: {e}")
            results.append(dict(method="FNMF", seed=seed, nmi=None, ari=None, time_s=None, error=str(e)))

    # --- Standalone SCC ---
    print("\n=== SCC (standalone) ===")
    for seed in seeds:
        print(f"  seed={seed}...", end=" ", flush=True)
        try:
            pred, elapsed = run_scc(X, k, seed)
            nmi = normalized_mutual_info_score(labels, pred)
            ari = adjusted_rand_score(labels, pred)
            print(f"NMI={nmi:.4f}, ARI={ari:.4f}, Time={elapsed:.1f}s")
            results.append(dict(
                method="SCC", seed=seed, nmi=nmi, ari=ari, time_s=elapsed, error=None
            ))
        except Exception as e:
            print(f"FAILED: {e}")
            results.append(dict(method="SCC", seed=seed, nmi=None, ari=None, time_s=None, error=str(e)))

    # --- DiMergeCo-FNMF (row-only partitioning) ---
    # Row-only (n_blocks=1) to preserve TF-IDF column structure
    for m_blocks in [4, 8]:
        method_name = f"DiMergeCo-FNMF-{m_blocks}x1"
        print(f"\n=== {method_name} ===")
        for seed in seeds:
            print(f"  seed={seed}...", end=" ", flush=True)
            try:
                dm = DiMergeCo(fnmf_atom, k, m_blocks=m_blocks, n_blocks=1, t_p=10)
                pred, elapsed = dm.run(X, seed=seed)
                nmi = normalized_mutual_info_score(labels, pred)
                ari = adjusted_rand_score(labels, pred)
                print(f"NMI={nmi:.4f}, ARI={ari:.4f}, Time={elapsed:.1f}s")
                results.append(dict(
                    method=method_name, seed=seed, nmi=nmi, ari=ari, time_s=elapsed, error=None,
                    m_blocks=m_blocks, n_blocks=1, t_p=10,
                ))
            except Exception as e:
                print(f"FAILED: {e}")
                results.append(dict(method=method_name, seed=seed, nmi=None, ari=None, time_s=None, error=str(e)))

    # --- DiMergeCo-SCC (row-only partitioning) ---
    for m_blocks in [4, 8]:
        method_name = f"DiMergeCo-SCC-{m_blocks}x1"
        print(f"\n=== {method_name} ===")
        for seed in seeds:
            print(f"  seed={seed}...", end=" ", flush=True)
            try:
                dm = DiMergeCo(scc_atom, k, m_blocks=m_blocks, n_blocks=1, t_p=10)
                pred, elapsed = dm.run(X, seed=seed)
                nmi = normalized_mutual_info_score(labels, pred)
                ari = adjusted_rand_score(labels, pred)
                print(f"NMI={nmi:.4f}, ARI={ari:.4f}, Time={elapsed:.1f}s")
                results.append(dict(
                    method=method_name, seed=seed, nmi=nmi, ari=ari, time_s=elapsed, error=None,
                    m_blocks=m_blocks, n_blocks=1, t_p=10,
                ))
            except Exception as e:
                print(f"FAILED: {e}")
                results.append(dict(method=method_name, seed=seed, nmi=None, ari=None, time_s=None, error=str(e)))

    # --- Save ---
    output = {
        "experiment": "RCV1-all baselines (804K x 47K sparse)",
        "dataset": "rcv1_all",
        "shape": list(X.shape),
        "nnz": X.nnz,
        "k": k,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }
    out_path = RESULTS_DIR / "rcv1_all_sparse_baselines.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 60)
    methods = sorted(set(r["method"] for r in results))
    for method in methods:
        ok = [r for r in results if r["method"] == method and r["nmi"] is not None]
        if ok:
            nmis = [r["nmi"] for r in ok]
            aris = [r["ari"] for r in ok]
            times = [r["time_s"] for r in ok]
            print(f"{method}: NMI={np.mean(nmis):.3f}+/-{np.std(nmis):.3f}  "
                  f"ARI={np.mean(aris):.3f}+/-{np.std(aris):.3f}  "
                  f"Time={np.mean(times):.1f}s  n={len(ok)}")
        else:
            print(f"{method}: ALL FAILED")


if __name__ == "__main__":
    main()
