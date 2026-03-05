#!/usr/bin/env python3
"""
Memory profiling: Measure peak RSS for each method on each dataset.

Uses resource.getrusage() to measure peak memory within the Python process.
For R1-RC3: peak memory per method per dataset.

Usage:
    python run_memory_profiling.py [--datasets classic4,bcw]
"""

import sys
import os
import time
import json
import resource
import argparse
import datetime
import numpy as np
from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

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
    elif name == "rcv1":
        from sklearn.datasets import fetch_rcv1
        rcv1 = fetch_rcv1(subset="train")
        X_full = rcv1.data
        targets = rcv1.target
        top_cats = ["CCAT", "ECAT", "GCAT", "MCAT"]
        cat_idx = [list(rcv1.target_names).index(c) for c in top_cats]
        mask = np.asarray(targets[:, cat_idx].sum(axis=1)).ravel() > 0
        X = X_full[mask]
        labels = np.asarray(targets[mask][:, cat_idx].argmax(axis=1)).ravel()
    else:
        raise ValueError(f"Unknown dataset: {name}")
    print(f"Loaded {name}: {X.shape}")
    return X, labels


def get_peak_rss_mb():
    """Get peak RSS in MB. Linux: ru_maxrss is in KB; macOS/BSD: in bytes."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == 'darwin':
        return usage.ru_maxrss / (1024.0 * 1024.0)
    return usage.ru_maxrss / 1024.0


def measure_method(method_fn, X, n_clusters, seed):
    """Run a method and measure absolute peak RSS (since process start) and time."""
    import gc
    gc.collect()

    t0 = time.time()
    pred_labels = method_fn(X, n_clusters, seed)
    elapsed = time.time() - t0

    rss_after = get_peak_rss_mb()
    return pred_labels, elapsed, rss_after


def run_spectralcc(X, n_clusters, seed):
    from sklearn.cluster import SpectralCoclustering
    from scipy.sparse import csr_matrix
    X_sp = csr_matrix(X) if not hasattr(X, 'toarray') else X
    model = SpectralCoclustering(n_clusters=n_clusters, random_state=seed,
                                 svd_method='arpack')
    model.fit(X_sp)
    return model.row_labels_


def run_scc(X, n_clusters, seed):
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import svds
    from sklearn.preprocessing import normalize
    from sklearn.cluster import KMeans
    X_sp = csr_matrix(X) if not hasattr(X, 'toarray') else X
    eps = 1e-12
    rs = np.asarray(X_sp.sum(axis=1)).ravel()
    cs = np.asarray(X_sp.sum(axis=0)).ravel()
    du = np.where(np.abs(rs) < eps, 0.0, rs ** -0.5)
    dv = np.where(np.abs(cs) < eps, 0.0, cs ** -0.5)
    An = diags(du) @ X_sp @ diags(dv)
    min_dim = min(An.shape)
    eff_k = min(n_clusters, min_dim - 1)
    if eff_k < 1:
        return np.zeros(X.shape[0], dtype=int)
    U, s, Vt = svds(An, k=eff_k)
    Z = np.vstack([U, Vt.T])
    Z = normalize(Z, norm='l2', axis=1)
    km = KMeans(n_clusters=eff_k, n_init=10, random_state=seed)
    labels = km.fit_predict(Z)
    return labels[:X.shape[0]]


def run_dimerge_scc(X, n_clusters, seed):
    from run_dimerge_co_variants import DiMergeCo, atom_scc_dhillon
    dmc = DiMergeCo(atom_fn=atom_scc_dhillon, k=n_clusters,
                    m_blocks=2, n_blocks=2, t_p=10)
    pred, _ = dmc.run(X, seed=seed)
    return pred


METHODS = {
    "SCC":            run_scc,
    "SpectralCC":     run_spectralcc,
    "DiMergeCo-SCC":  run_dimerge_scc,
}


def main():
    parser = argparse.ArgumentParser(description="Memory profiling")
    parser.add_argument("--datasets", type=str, default="classic4,bcw",
                        help="Comma-separated datasets")
    parser.add_argument("--seeds", type=str, default="0")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    results = []
    for ds_name in datasets:
        n_clusters = 2 if ds_name == "bcw" else 4
        X, labels = load_dataset(ds_name)

        for method_name, method_fn in METHODS.items():
            for seed in seeds:
                print(f"{ds_name} / {method_name} / seed={seed} ...", end=" ", flush=True)
                try:
                    pred, elapsed, peak_rss = measure_method(
                        method_fn, X, n_clusters, seed)
                    nmi = normalized_mutual_info_score(labels, pred)
                    ari = adjusted_rand_score(labels, pred)
                    r = {
                        "dataset": ds_name, "method": method_name,
                        "seed": seed, "nmi": round(float(nmi), 4),
                        "ari": round(float(ari), 4),
                        "time_s": round(elapsed, 4),
                        "peak_rss_mb": round(peak_rss, 1),
                        "error": None,
                    }
                    print(f"NMI={nmi:.4f} RSS={peak_rss:.0f}MB Time={elapsed:.2f}s")
                except Exception as e:
                    r = {
                        "dataset": ds_name, "method": method_name,
                        "seed": seed, "nmi": None, "ari": None,
                        "time_s": None, "peak_rss_mb": None,
                        "error": str(e),
                    }
                    print(f"FAILED: {e}")
                results.append(r)

    # Summary
    print("\n" + "=" * 80)
    print("MEMORY PROFILING SUMMARY")
    print("=" * 80)
    print(f"{'Dataset':<12} {'Method':<16} {'Peak RSS (MB)':>14} {'Time (s)':>12}")
    print("-" * 58)
    for r in results:
        if r["error"] is None:
            print(f"{r['dataset']:<12} {r['method']:<16} "
                  f"{r['peak_rss_mb']:>14.1f} {r['time_s']:>12.3f}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "memory_profiling.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.datetime.now().isoformat(),
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
