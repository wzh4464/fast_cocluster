#!/usr/bin/env python3
"""
Run all baseline co-clustering methods on BCW (Breast Cancer Wisconsin, 569x30).
Measures NMI, ARI, and wall-clock time for each method.

Usage:
    python run_bcw_baselines.py [--methods METHOD1,METHOD2,...] [--seeds 0,1,2]

Methods: SpectralCC, SCC-Dhillon, NBVD, ONM3F, ONMTF, PNMTF, FNMF
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

# ── Paths ─────────────────────────────────────────────────────────────────────

BASELINES_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = BASELINES_DIR / "results"

# ── Patch NMTFcoclust imports ─────────────────────────────────────────────────

def _patch_sklearn_compat():
    import sklearn.utils.validation as val
    _orig_check_array = val.check_array
    def _compat_check_array(*args, **kwargs):
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return _orig_check_array(*args, **kwargs)
    val.check_array = _compat_check_array


def _setup_nmtfcoclust_imports():
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
        f"_nmtf_parent.Models.{model_filename[:-3]}", filepath,
        submodule_search_locations=[])
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "_nmtf_parent.Models"
    sys.modules[spec.name] = mod
    code = compile(source, str(filepath), "exec")
    exec(code, mod.__dict__)
    return getattr(mod, class_name)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_bcw():
    """Load BCW dataset (569 x 30) with labels."""
    bcw_path = DATA_DIR / "bcw.npy"
    labels_path = DATA_DIR / "bcw_labels.npy"

    if not bcw_path.exists():
        print("BCW data not found. Downloading...")
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        np.save(bcw_path, data.data.astype(np.float64))
        np.save(labels_path, data.target.astype(np.int64))

    X = np.load(bcw_path)
    labels = np.load(labels_path)
    print(f"Loaded BCW: {X.shape}, {len(np.unique(labels))} classes")
    return X, labels


# ── Baseline methods ──────────────────────────────────────────────────────────

def run_spectralcc(X, labels, n_clusters, seed):
    from sklearn.cluster import SpectralCoclustering
    from scipy.sparse import csr_matrix
    X_sparse = csr_matrix(X) if not hasattr(X, 'toarray') else X
    model = SpectralCoclustering(n_clusters=n_clusters, random_state=seed,
                                 svd_method='arpack')
    t0 = time.time()
    model.fit(X_sparse)
    elapsed = time.time() - t0
    return model.row_labels_, elapsed


def run_scc_dhillon(X, labels, n_clusters, seed):
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
    V = Vt.T
    Z = np.vstack([U, V])
    Z = normalize(Z, norm='l2', axis=1)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    all_labels = km.fit_predict(Z)
    elapsed = time.time() - t0
    return all_labels[:X.shape[0]], elapsed


def run_nbvd(X, labels, n_clusters, seed):
    NBVD = _load_nmtfcoclust_model("NMTFcoclust_NBVD.py", "NBVD")
    model = NBVD(n_row_clusters=n_clusters, n_col_clusters=n_clusters,
                 max_iter=100, n_init=1, tol=1e-9, random_state=seed)
    t0 = time.time()
    model.fit(X)
    elapsed = time.time() - t0
    return np.array(model.row_labels_) - 1, elapsed


def run_onm3f(X, labels, n_clusters, seed):
    ONM3F = _load_nmtfcoclust_model("NMTFcoclust_ONM3F.py", "ONM3F")
    model = ONM3F(n_row_clusters=n_clusters, n_col_clusters=n_clusters,
                  max_iter=100, n_init=1, tol=1e-9, random_state=seed)
    t0 = time.time()
    model.fit(X)
    elapsed = time.time() - t0
    return np.array(model.row_labels_) - 1, elapsed


def run_onmtf(X, labels, n_clusters, seed):
    ONMTF = _load_nmtfcoclust_model("NMTFcoclust_ONMTF.py", "ONMTF")
    model = ONMTF(n_row_clusters=n_clusters, n_col_clusters=n_clusters,
                  max_iter=100, n_init=1, tol=1e-9, random_state=seed)
    t0 = time.time()
    model.fit(X)
    elapsed = time.time() - t0
    return np.array(model.row_labels_) - 1, elapsed


def run_pnmtf(X, labels, n_clusters, seed):
    PNMTF = _load_nmtfcoclust_model("NMTFcoclust_PNMTF.py", "PNMTF")
    model = PNMTF(n_row_clusters=n_clusters, n_col_clusters=n_clusters,
                  tau=0.5, eta=0.5, gamma=0.1,
                  max_iter=100, n_init=1, tol=1e-9, random_state=seed)
    t0 = time.time()
    model.fit(X)
    elapsed = time.time() - t0
    return np.array(model.row_labels_) - 1, elapsed


def run_fnmf(X, labels, n_clusters, seed):
    nonnegfac_dir = BASELINES_DIR / "nonnegfac-python"
    if str(nonnegfac_dir) not in sys.path:
        sys.path.insert(0, str(nonnegfac_dir))
    from nonnegfac.nmf import NMF_ANLS_BLOCKPIVOT
    np.random.seed(seed)
    solver = NMF_ANLS_BLOCKPIVOT(default_max_iter=100)
    t0 = time.time()
    W, H, info = solver.run(X, n_clusters, verbose=-1)
    elapsed = time.time() - t0
    return np.argmax(W, axis=1), elapsed


# ── Registry ──────────────────────────────────────────────────────────────────

METHODS = {
    "SpectralCC":  {"fn": run_spectralcc,  "paper_name": "SpectralCC (sklearn)"},
    "SCC-Dhillon": {"fn": run_scc_dhillon, "paper_name": "SCC (Dhillon 2001, k SVs)"},
    "NBVD":        {"fn": run_nbvd,        "paper_name": "NMTF (Long 2005)"},
    "ONM3F":       {"fn": run_onm3f,       "paper_name": "ONMTF (Ding 2006)"},
    "ONMTF":       {"fn": run_onmtf,       "paper_name": "ONMTF (Yoo 2010)"},
    "PNMTF":       {"fn": run_pnmtf,       "paper_name": "PNMTF (Wang 2017)"},
    "FNMF":        {"fn": run_fnmf,         "paper_name": "FNMTF (Kim & Park 2011)"},
}


def evaluate(true_labels, pred_labels):
    return (normalized_mutual_info_score(true_labels, pred_labels),
            adjusted_rand_score(true_labels, pred_labels))


def run_single(method_name, X, labels, n_clusters, seed):
    info = METHODS[method_name]
    try:
        pred_labels, elapsed = info["fn"](X, labels, n_clusters, seed)
        nmi, ari = evaluate(labels, pred_labels)
        return {"method": method_name, "paper_name": info["paper_name"],
                "seed": seed, "nmi": round(float(nmi), 4),
                "ari": round(float(ari), 4), "time_s": round(elapsed, 4),
                "error": None}
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"method": method_name, "paper_name": info["paper_name"],
                "seed": seed, "nmi": None, "ari": None, "time_s": None,
                "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Run BCW baselines")
    parser.add_argument("--methods", type=str, default=",".join(METHODS.keys()),
                        help=f"Comma-separated methods: {list(METHODS.keys())}")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9",
                        help="Comma-separated random seeds")
    parser.add_argument("--n-clusters", type=int, default=2,
                        help="Number of co-clusters (default: 2 for BCW)")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    for m in methods:
        if m not in METHODS:
            print(f"Unknown method: {m}. Available: {list(METHODS.keys())}")
            sys.exit(1)

    print("=" * 70)
    print(f"BCW Baseline Experiments — {datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(f"Methods: {methods}, Seeds: {seeds}, k={args.n_clusters}")
    print("=" * 70)

    X, labels = load_bcw()

    results = []
    for method_name in methods:
        print(f"\n--- {method_name} ({METHODS[method_name]['paper_name']}) ---")
        for seed in seeds:
            print(f"  seed={seed} ...", end=" ", flush=True)
            r = run_single(method_name, X, labels, args.n_clusters, seed)
            results.append(r)
            if r["error"]:
                print(f"FAILED: {r['error']}")
            else:
                print(f"NMI={r['nmi']:.4f}  ARI={r['ari']:.4f}  Time={r['time_s']:.4f}s")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (mean +/- std)")
    print("=" * 70)
    print(f"{'Method':<12} {'NMI':>16} {'ARI':>16} {'Time (s)':>16}")
    print("-" * 64)
    for method_name in methods:
        ok = [r for r in results if r["method"] == method_name and r["error"] is None]
        if not ok:
            print(f"{method_name:<12} {'ALL FAILED':>16}")
            continue
        nmis = [r["nmi"] for r in ok]
        aris = [r["ari"] for r in ok]
        times = [r["time_s"] for r in ok]
        print(f"{method_name:<12} "
              f"{np.mean(nmis):.3f} +/- {np.std(nmis):.3f}  "
              f"{np.mean(aris):.3f} +/- {np.std(aris):.3f}  "
              f"{np.mean(times):>7.4f} +/- {np.std(times):.4f}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "bcw_baselines.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.datetime.now().isoformat(),
            "dataset": "bcw",
            "shape": list(X.shape),
            "n_clusters": args.n_clusters,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
