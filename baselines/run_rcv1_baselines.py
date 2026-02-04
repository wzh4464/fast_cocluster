#!/usr/bin/env python3
"""
Run all baseline co-clustering methods on RCV1.
Downloads from sklearn, converts multi-label to single-label using
top-level categories (CCAT, ECAT, GCAT, MCAT).

Methods: SpectralCC, SCC-Dhillon, NBVD, ONM3F, ONMTF, PNMTF, FNMF

Usage:
    python run_rcv1_baselines.py [--subset train|test|all] [--methods SpectralCC,SCC-Dhillon,...]
"""

import sys
import os
import time
import json
import types
import argparse
import datetime
import numpy as np
from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

BASELINES_DIR = Path(__file__).parent
RESULTS_DIR = BASELINES_DIR / "results"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ── NMTFcoclust import machinery (same as run_classic4_baselines.py) ─────────

def _patch_sklearn_compat():
    """Patch sklearn.utils.check_array for NMTFcoclust compatibility."""
    import sklearn.utils.validation as val
    _orig_check_array = val.check_array

    def _compat_check_array(*args, **kwargs):
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return _orig_check_array(*args, **kwargs)

    val.check_array = _compat_check_array


def _setup_nmtfcoclust_imports():
    """Create fake coclust-like package structure for NMTFcoclust models."""
    _patch_sklearn_compat()
    import coclust_shim

    nmtfcoclust_dir = BASELINES_DIR / "NMTFcoclust"
    models_dir = nmtfcoclust_dir / "Models"

    parent_pkg = types.ModuleType("_nmtf_parent")
    parent_pkg.__path__ = [str(nmtfcoclust_dir)]
    parent_pkg.__package__ = "_nmtf_parent"

    models_pkg = types.ModuleType("_nmtf_parent.Models")
    models_pkg.__path__ = [str(models_dir)]
    models_pkg.__package__ = "_nmtf_parent.Models"

    init_mod = types.ModuleType("_nmtf_parent.initialization")
    init_mod.random_init = coclust_shim.random_init

    io_pkg = types.ModuleType("_nmtf_parent.io")
    io_pkg.__path__ = []
    io_pkg.__package__ = "_nmtf_parent.io"
    input_checking_mod = types.ModuleType("_nmtf_parent.io.input_checking")
    input_checking_mod.check_positive = coclust_shim.check_positive
    io_pkg.input_checking = input_checking_mod

    sys.modules["_nmtf_parent"] = parent_pkg
    sys.modules["_nmtf_parent.Models"] = models_pkg
    sys.modules["_nmtf_parent.initialization"] = init_mod
    sys.modules["_nmtf_parent.io"] = io_pkg
    sys.modules["_nmtf_parent.io.input_checking"] = input_checking_mod

    if str(BASELINES_DIR) not in sys.path:
        sys.path.insert(0, str(BASELINES_DIR))

    return models_dir


def _load_nmtfcoclust_model(model_filename, class_name):
    """Load a NMTFcoclust model class by patching its module's package."""
    import importlib.util

    models_dir = _setup_nmtfcoclust_imports()
    filepath = models_dir / model_filename

    source = filepath.read_text()
    source = source.replace("force_all_finite", "ensure_all_finite")
    source = source.replace("(enum/denom)", "(enum/(denom + 1e-16))")
    source = source.replace("(enum / denom)", "(enum / (denom + 1e-16))")
    source = source.replace(
        'raise ValueError("matrix may contain negative or unexpected NaN values")',
        'break  # NaN encountered, stop this run early'
    )

    spec = importlib.util.spec_from_file_location(
        f"_nmtf_parent.Models.{model_filename[:-3]}",
        filepath,
        submodule_search_locations=[]
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "_nmtf_parent.Models"
    sys.modules[spec.name] = mod
    code = compile(source, str(filepath), "exec")
    exec(code, mod.__dict__)
    return getattr(mod, class_name)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_rcv1(subset="all"):
    """Load RCV1 and convert multi-label to single top-level category."""
    from sklearn.datasets import fetch_rcv1
    print(f"Fetching RCV1 (subset={subset})...")
    rcv1 = fetch_rcv1(subset=subset, shuffle=False)
    X = rcv1.data  # sparse CSR, shape (n_docs, n_terms)
    Y = rcv1.target  # sparse, shape (n_docs, 103 categories)

    top_cats = ['CCAT', 'ECAT', 'GCAT', 'MCAT']
    cat_indices = []
    for cat in top_cats:
        idx = list(rcv1.target_names).index(cat)
        cat_indices.append(idx)
        print(f"  {cat} -> index {idx}")

    Y_top = Y[:, cat_indices].toarray()
    labels = np.full(X.shape[0], -1, dtype=int)
    for i in range(X.shape[0]):
        memberships = Y_top[i]
        if memberships.sum() > 0:
            labels[i] = np.argmax(memberships)

    mask = labels >= 0
    X_filtered = X[mask]
    labels_filtered = labels[mask]

    print(f"\nRCV1 loaded:")
    print(f"  Original: {X.shape[0]} docs, {X.shape[1]} terms")
    print(f"  After filtering (has top-level cat): {X_filtered.shape[0]} docs")
    print(f"  Sparsity: {1 - X_filtered.nnz / (X_filtered.shape[0] * X_filtered.shape[1]):.4%}")
    for i, cat in enumerate(top_cats):
        print(f"  {cat}: {(labels_filtered == i).sum()} docs")

    return X_filtered, labels_filtered, top_cats


# ── Baseline methods ─────────────────────────────────────────────────────────

def run_spectralcc(X, labels, n_clusters, seed):
    """sklearn SpectralCoclustering (uses ARPACK sparse SVD internally)."""
    from sklearn.cluster import SpectralCoclustering
    model = SpectralCoclustering(n_clusters=n_clusters, random_state=seed,
                                 svd_method='arpack')
    t0 = time.time()
    model.fit(X)
    elapsed = time.time() - t0
    return model.row_labels_, elapsed


def run_scc_dhillon(X, labels, n_clusters, seed):
    """SCC (Dhillon 2001) — sparse version using scipy.sparse.linalg.svds."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import svds
    from sklearn.preprocessing import normalize
    from sklearn.cluster import KMeans

    eps = 1e-12

    row_sums = np.asarray(X.sum(axis=1)).ravel()
    col_sums = np.asarray(X.sum(axis=0)).ravel()
    du_inv = np.where(np.abs(row_sums) < eps, 0.0, row_sums ** -0.5)
    dv_inv = np.where(np.abs(col_sums) < eps, 0.0, col_sums ** -0.5)
    An = diags(du_inv) @ X @ diags(dv_inv)

    t0 = time.time()
    U, s, Vt = svds(An, k=n_clusters)
    V = Vt.T

    Z = np.vstack([U, V])
    Z = normalize(Z, norm='l2', axis=1)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    all_labels = km.fit_predict(Z)
    elapsed = time.time() - t0

    row_labels = all_labels[:X.shape[0]]
    return row_labels, elapsed


def run_nbvd(X, labels, n_clusters, seed):
    """NBVD = NMTF (Long 2005)."""
    NBVD = _load_nmtfcoclust_model("NMTFcoclust_NBVD.py", "NBVD")
    t0 = time.time()
    # Convert sparse to dense for NMTFcoclust
    X_dense = X.toarray() if hasattr(X, 'toarray') else X
    model = NBVD(n_row_clusters=n_clusters, n_col_clusters=n_clusters,
                 max_iter=100, n_init=1, tol=1e-9, random_state=seed)
    model.fit(X_dense)
    elapsed = time.time() - t0
    row_labels = np.array(model.row_labels_) - 1
    return row_labels, elapsed


def run_onm3f(X, labels, n_clusters, seed):
    """ONM3F = ONMTF with Lagrange multipliers (Ding 2006/2008)."""
    ONM3F = _load_nmtfcoclust_model("NMTFcoclust_ONM3F.py", "ONM3F")
    t0 = time.time()
    X_dense = X.toarray() if hasattr(X, 'toarray') else X
    model = ONM3F(n_row_clusters=n_clusters, n_col_clusters=n_clusters,
                  max_iter=100, n_init=1, tol=1e-9, random_state=seed)
    model.fit(X_dense)
    elapsed = time.time() - t0
    row_labels = np.array(model.row_labels_) - 1
    return row_labels, elapsed


def run_onmtf(X, labels, n_clusters, seed):
    """ONMTF (Yoo 2010)."""
    ONMTF = _load_nmtfcoclust_model("NMTFcoclust_ONMTF.py", "ONMTF")
    t0 = time.time()
    X_dense = X.toarray() if hasattr(X, 'toarray') else X
    model = ONMTF(n_row_clusters=n_clusters, n_col_clusters=n_clusters,
                  max_iter=100, n_init=1, tol=1e-9, random_state=seed)
    model.fit(X_dense)
    elapsed = time.time() - t0
    row_labels = np.array(model.row_labels_) - 1
    return row_labels, elapsed


def run_pnmtf(X, labels, n_clusters, seed):
    """Penalized NMTF (Wang 2017). Proxy for Chen 2023 PNMTF."""
    PNMTF = _load_nmtfcoclust_model("NMTFcoclust_PNMTF.py", "PNMTF")
    t0 = time.time()
    X_dense = X.toarray() if hasattr(X, 'toarray') else X
    model = PNMTF(n_row_clusters=n_clusters, n_col_clusters=n_clusters,
                  tau=0.5, eta=0.5, gamma=0.1,
                  max_iter=100, n_init=1, tol=1e-9, random_state=seed)
    model.fit(X_dense)
    elapsed = time.time() - t0
    row_labels = np.array(model.row_labels_) - 1
    return row_labels, elapsed


def run_fnmf(X, labels, n_clusters, seed):
    """Fast NMF (Kim & Park 2011) using ANLS-BPP. Row labels via argmax(W)."""
    nonnegfac_dir = BASELINES_DIR / "nonnegfac-python"
    if str(nonnegfac_dir) not in sys.path:
        sys.path.insert(0, str(nonnegfac_dir))
    from nonnegfac.nmf import NMF_ANLS_BLOCKPIVOT

    t0 = time.time()
    X_dense = X.toarray() if hasattr(X, 'toarray') else X
    np.random.seed(seed)
    solver = NMF_ANLS_BLOCKPIVOT(default_max_iter=100)
    W, H, info = solver.run(X_dense, n_clusters, verbose=-1)
    elapsed = time.time() - t0
    row_labels = np.argmax(W, axis=1)
    return row_labels, elapsed


# ── Registry ─────────────────────────────────────────────────────────────────

METHODS = {
    "SpectralCC": {
        "fn": run_spectralcc,
        "paper_name": "SpectralCC (sklearn, ARPACK sparse SVD)",
    },
    "SCC-Dhillon": {
        "fn": run_scc_dhillon,
        "paper_name": "SCC (Dhillon 2001, k SVs, sparse ARPACK)",
    },
    "NBVD": {
        "fn": run_nbvd,
        "paper_name": "NMTF (Long 2005, via NMTFcoclust NBVD)",
    },
    "ONM3F": {
        "fn": run_onm3f,
        "paper_name": "ONMTF (Ding 2006, via NMTFcoclust ONM3F)",
    },
    "ONMTF": {
        "fn": run_onmtf,
        "paper_name": "ONMTF (Yoo 2010, via NMTFcoclust ONMTF)",
    },
    "PNMTF": {
        "fn": run_pnmtf,
        "paper_name": "PNMTF (Wang 2017, proxy for Chen 2023)",
    },
    "FNMF": {
        "fn": run_fnmf,
        "paper_name": "FNMTF (Kim & Park 2011, fast NMF + argmax)",
    },
}


def evaluate(true_labels, pred_labels):
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    return nmi, ari


def save_for_rust(X, labels, subset_name):
    """Save sparse matrix data for potential Rust consumption."""
    from scipy.sparse import save_npz
    out_dir = DATA_DIR / "rcv1"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_npz(out_dir / f"rcv1_{subset_name}.npz", X)
    np.save(out_dir / f"rcv1_{subset_name}_labels.npy", labels)
    print(f"\nSaved RCV1 data to {out_dir}/rcv1_{subset_name}.{{npz,_labels.npy}}")
    print(f"  Shape: {X.shape}, nnz: {X.nnz}")


def main():
    parser = argparse.ArgumentParser(description="Run RCV1 baselines")
    parser.add_argument("--subset", type=str, default="train",
                        choices=["train", "test", "all"],
                        help="RCV1 subset (default: train)")
    parser.add_argument("--methods", type=str, default=",".join(METHODS.keys()),
                        help="Comma-separated methods")
    parser.add_argument("--seeds", type=str, default="0",
                        help="Comma-separated seeds (default: 0)")
    parser.add_argument("--n-clusters", type=int, default=4,
                        help="Number of co-clusters (default: 4)")
    parser.add_argument("--save-data", action="store_true",
                        help="Save data files for Rust")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    print("=" * 70)
    print(f"RCV1 Baseline Experiments — {datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(f"Subset: {args.subset}, Methods: {methods}, Seeds: {seeds}")
    print("=" * 70)

    X, labels, top_cats = load_rcv1(args.subset)

    if args.save_data:
        save_for_rust(X, labels, args.subset)

    results = []
    for method_name in methods:
        if method_name not in METHODS:
            print(f"Unknown method: {method_name}")
            continue
        info = METHODS[method_name]
        print(f"\n--- {method_name} ({info['paper_name']}) ---")
        for seed in seeds:
            print(f"  seed={seed} ...", end=" ", flush=True)
            try:
                t0 = time.time()
                pred, elapsed = info["fn"](X, labels, args.n_clusters, seed)
                nmi, ari = evaluate(labels, pred)
                r = {"method": method_name, "seed": seed,
                     "nmi": round(float(nmi), 4), "ari": round(float(ari), 4),
                     "time_s": round(elapsed, 2), "error": None}
                print(f"NMI={nmi:.4f}  ARI={ari:.4f}  Time={elapsed:.1f}s")
            except Exception as e:
                import traceback
                traceback.print_exc()
                r = {"method": method_name, "seed": seed,
                     "nmi": None, "ari": None, "time_s": None, "error": str(e)}
                print(f"FAILED: {e}")
            results.append(r)

    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY — RCV1 {args.subset} ({X.shape[0]} x {X.shape[1]})")
    print("=" * 70)
    print(f"{'Method':<14} {'NMI':>16} {'ARI':>16} {'Time (s)':>16}")
    print("-" * 64)

    for method_name in methods:
        ok = [r for r in results if r["method"] == method_name and r["error"] is None]
        if not ok:
            print(f"{method_name:<14} {'FAILED':>16}")
            continue
        nmis = [r["nmi"] for r in ok]
        aris = [r["ari"] for r in ok]
        times = [r["time_s"] for r in ok]
        print(f"{method_name:<14} "
              f"{np.mean(nmis):.4f} +/- {np.std(nmis):.4f}  "
              f"{np.mean(aris):.4f} +/- {np.std(aris):.4f}  "
              f"{np.mean(times):>8.1f} +/- {np.std(times):.1f}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"rcv1_{args.subset}_baselines.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.datetime.now().isoformat(),
            "dataset": f"rcv1_{args.subset}",
            "shape": list(X.shape),
            "n_clusters": args.n_clusters,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
