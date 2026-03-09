#!/usr/bin/env python3
"""Run DiMergeCo-FNMF on Amazon 24-cat dataset (1.1M x 731K sparse).

Standalone FNMF already done (NMI=0.427). SCC is unstable on this sparse data.
DiMergeCo-SCC failed due to SCC atom instability on subblocks.
This script runs DiMergeCo-FNMF with row-only partitioning.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).parent))
from run_dimerge_co_variants import DiMergeCo

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_amazon_24cat():
    X = sparse.load_npz(str(DATA_DIR / "amazon_24cat_sparse.npz"))
    row_labels = np.load(str(DATA_DIR / "amazon_24cat_row_labels.npy"))
    return X, row_labels


def fnmf_atom(X_sub, k, seed=0):
    nmf = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=100)
    W = nmf.fit_transform(X_sub)
    return np.argmax(W, axis=1)


def main():
    print("Loading Amazon 24-category dataset...")
    X, row_labels = load_amazon_24cat()
    k = 24
    print(f"Shape: {X.shape}, NNZ: {X.nnz:,}, k={k}")

    seeds = list(range(3))  # 3 seeds (slow due to k=24)
    results = []

    # --- DiMergeCo-FNMF with row-only partitioning ---
    configs = [
        (4, 1, 10, "DiMergeCo-FNMF-4x1"),
        (8, 1, 10, "DiMergeCo-FNMF-8x1"),
    ]

    for m_blocks, n_blocks, t_p, method_name in configs:
        print(f"\n=== {method_name} (t_p={t_p}, {len(seeds)} seeds) ===")
        for seed in seeds:
            print(f"  seed={seed}...", end=" ", flush=True)
            try:
                dm = DiMergeCo(fnmf_atom, k, m_blocks=m_blocks, n_blocks=n_blocks, t_p=t_p)
                pred, elapsed = dm.run(X, seed=seed)
                nmi = normalized_mutual_info_score(row_labels, pred)
                ari = adjusted_rand_score(row_labels, pred)
                print(f"NMI={nmi:.4f}, ARI={ari:.4f}, Time={elapsed:.1f}s")
                results.append(dict(
                    method=method_name, seed=seed, nmi=nmi, ari=ari, time_s=elapsed, error=None,
                    m_blocks=m_blocks, n_blocks=n_blocks, t_p=t_p,
                ))
            except Exception as e:
                print(f"FAILED: {e}")
                results.append(dict(method=method_name, seed=seed, nmi=None, ari=None, time_s=None, error=str(e)))

        # Save checkpoint after each config
        output = {
            "experiment": "Amazon 24-cat DiMergeCo-FNMF",
            "dataset": "amazon_24cat",
            "shape": list(X.shape),
            "nnz": X.nnz,
            "k": k,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
        }
        out_path = RESULTS_DIR / "amazon_24cat_dimerge_fnmf.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  [saved checkpoint to {out_path}]")

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
