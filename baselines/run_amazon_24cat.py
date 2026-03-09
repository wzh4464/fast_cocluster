#!/usr/bin/env python3
"""Run baselines + DiMergeCo on Amazon 24-category dataset (sparse).

Only methods that support sparse input are tested:
  - SCC (sklearn SpectralCoclustering)
  - SpectralCC (sklearn SpectralCoclustering, same impl)
  - FNMF (sklearn NMF)
  - DiMergeCo-SCC

NMF baselines (PNMTF, ONMTF, NBVD, ONM3F) require dense matrices and
are infeasible at this scale (1.1M x 731K).

Usage:
    python baselines/run_amazon_24cat.py
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

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_amazon_24cat():
    X = sparse.load_npz(str(DATA_DIR / "amazon_24cat_sparse.npz"))
    row_labels = np.load(str(DATA_DIR / "amazon_24cat_row_labels.npy"))
    col_labels = np.load(str(DATA_DIR / "amazon_24cat_col_labels.npy"))
    return X, row_labels, col_labels


def scc_atom(X_sub, k, seed=0):
    """SCC atom for DiMergeCo (uses sklearn SpectralCoclustering)."""
    model = SpectralCoclustering(
        n_clusters=k, random_state=seed, svd_method="randomized"
    )
    model.fit(X_sub)
    return model.row_labels_


def run_scc(X, k, seed=0):
    model = SpectralCoclustering(
        n_clusters=k, random_state=seed, svd_method="randomized"
    )
    t0 = time.time()
    model.fit(X)
    elapsed = time.time() - t0
    return model.row_labels_, elapsed


def run_fnmf(X, k, seed=0):
    nmf = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=100)
    t0 = time.time()
    W = nmf.fit_transform(X)
    elapsed = time.time() - t0
    pred = np.argmax(W, axis=1)
    return pred, elapsed


def main():
    print("Loading Amazon 24-category dataset...")
    X, row_labels, col_labels = load_amazon_24cat()
    k = 24
    print(f"Shape: {X.shape}, NNZ: {X.nnz:,}, k={k}")

    seeds = list(range(3))  # 3 seeds for large dataset (slow)
    results = []

    # --- SCC ---
    print("\n=== SCC (SpectralCoclustering) ===")
    for seed in seeds:
        print(f"  seed={seed}...", end=" ", flush=True)
        try:
            pred, elapsed = run_scc(X, k, seed)
            nmi = normalized_mutual_info_score(row_labels, pred)
            ari = adjusted_rand_score(row_labels, pred)
            print(f"NMI={nmi:.4f}, ARI={ari:.4f}, Time={elapsed:.1f}s")
            results.append(dict(
                method="SCC", seed=seed, nmi=nmi, ari=ari, time_s=elapsed, error=None
            ))
        except Exception as e:
            print(f"FAILED: {e}")
            results.append(dict(method="SCC", seed=seed, nmi=None, ari=None, time_s=None, error=str(e)))

    # --- FNMF ---
    print("\n=== FNMF (sklearn NMF) ===")
    for seed in seeds:
        print(f"  seed={seed}...", end=" ", flush=True)
        try:
            pred, elapsed = run_fnmf(X, k, seed)
            nmi = normalized_mutual_info_score(row_labels, pred)
            ari = adjusted_rand_score(row_labels, pred)
            print(f"NMI={nmi:.4f}, ARI={ari:.4f}, Time={elapsed:.1f}s")
            results.append(dict(
                method="FNMF", seed=seed, nmi=nmi, ari=ari, time_s=elapsed, error=None
            ))
        except Exception as e:
            print(f"FAILED: {e}")
            results.append(dict(method="FNMF", seed=seed, nmi=None, ari=None, time_s=None, error=str(e)))

    # --- DiMergeCo-SCC ---
    print("\n=== DiMergeCo-SCC ===")
    for seed in seeds:
        print(f"  seed={seed}...", end=" ", flush=True)
        try:
            dm = DiMergeCo(scc_atom, k, m_blocks=4, n_blocks=4, t_p=10)
            pred, elapsed = dm.run(X, seed=seed)
            nmi = normalized_mutual_info_score(row_labels, pred)
            ari = adjusted_rand_score(row_labels, pred)
            print(f"NMI={nmi:.4f}, ARI={ari:.4f}, Time={elapsed:.1f}s")
            results.append(dict(
                method="DiMergeCo-SCC", seed=seed, nmi=nmi, ari=ari, time_s=elapsed, error=None,
                m_blocks=4, n_blocks=4, t_p=10,
            ))
        except Exception as e:
            print(f"FAILED: {e}")
            results.append(dict(method="DiMergeCo-SCC", seed=seed, nmi=None, ari=None, time_s=None, error=str(e)))

    # --- Save ---
    output = {
        "experiment": "Amazon 24-category baselines",
        "dataset": "amazon_24cat",
        "shape": list(X.shape),
        "nnz": X.nnz,
        "k": k,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }
    out_path = RESULTS_DIR / "amazon_24cat_baselines.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 60)
    for method in ["SCC", "FNMF", "DiMergeCo-SCC"]:
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
