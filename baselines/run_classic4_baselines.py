#!/usr/bin/env python3
"""
Run all baseline co-clustering methods on Classic4 (paper version: 6460x4667).
Measures NMI, ARI, and wall-clock time for each method.

Usage:
    python run_classic4_baselines.py [--methods METHOD1,METHOD2,...] [--seeds 0,1,2]

Methods: SpectralCC, NBVD, ONM3F, ONMTF, PNMTF, FNMF
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
# NMTFcoclust models use relative imports from `coclust` package:
#   from ..initialization import random_init
#   from ..io.input_checking import check_positive
# We create a fake package hierarchy so these imports resolve to our shim.

def _patch_sklearn_compat():
    """Patch sklearn.utils.check_array for NMTFcoclust compatibility.
    sklearn 1.8 renamed force_all_finite -> ensure_all_finite."""
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

    # The Models directory imports from .. (parent package)
    # So we need: parent.initialization.random_init, parent.io.input_checking.check_positive
    nmtfcoclust_dir = BASELINES_DIR / "NMTFcoclust"
    models_dir = nmtfcoclust_dir / "Models"

    # Create the parent package (NMTFcoclust) in sys.modules
    parent_pkg = types.ModuleType("_nmtf_parent")
    parent_pkg.__path__ = [str(nmtfcoclust_dir)]
    parent_pkg.__package__ = "_nmtf_parent"

    # Create Models sub-package
    models_pkg = types.ModuleType("_nmtf_parent.Models")
    models_pkg.__path__ = [str(models_dir)]
    models_pkg.__package__ = "_nmtf_parent.Models"

    # Create initialization module
    init_mod = types.ModuleType("_nmtf_parent.initialization")
    init_mod.random_init = coclust_shim.random_init

    # Create io.input_checking module
    io_pkg = types.ModuleType("_nmtf_parent.io")
    io_pkg.__path__ = []
    io_pkg.__package__ = "_nmtf_parent.io"
    input_checking_mod = types.ModuleType("_nmtf_parent.io.input_checking")
    input_checking_mod.check_positive = coclust_shim.check_positive
    io_pkg.input_checking = input_checking_mod

    # Register in sys.modules
    sys.modules["_nmtf_parent"] = parent_pkg
    sys.modules["_nmtf_parent.Models"] = models_pkg
    sys.modules["_nmtf_parent.initialization"] = init_mod
    sys.modules["_nmtf_parent.io"] = io_pkg
    sys.modules["_nmtf_parent.io.input_checking"] = input_checking_mod

    # Add baselines dir to path for coclust_shim
    if str(BASELINES_DIR) not in sys.path:
        sys.path.insert(0, str(BASELINES_DIR))

    return models_dir


def _load_nmtfcoclust_model(model_filename, class_name):
    """Load a NMTFcoclust model class by patching its module's package."""
    import importlib.util

    models_dir = _setup_nmtfcoclust_imports()
    filepath = models_dir / model_filename

    # Patch source for compatibility:
    # 1. force_all_finite -> ensure_all_finite (sklearn 1.8 rename)
    # 2. Add epsilon to divisions to prevent NaN in multiplicative updates
    source = filepath.read_text()
    source = source.replace("force_all_finite", "ensure_all_finite")
    # Prevent division-by-zero in multiplicative update rules
    source = source.replace("(enum/denom)", "(enum/(denom + 1e-16))")
    source = source.replace("(enum / denom)", "(enum / (denom + 1e-16))")
    # Prevent NaN propagation: replace the NaN check that raises with a reset
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
    # Execute patched source instead of original file
    code = compile(source, str(filepath), "exec")
    exec(code, mod.__dict__)
    return getattr(mod, class_name)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_classic4():
    """Load Classic4 paper version (6460 x 4667) with labels."""
    X = np.load(DATA_DIR / "classic4_paper.npy")
    labels = np.load(DATA_DIR / "classic4_paper_labels.npy")
    n_nz = np.count_nonzero(X)
    print(f"Loaded Classic4: {X.shape}, {len(np.unique(labels))} classes, "
          f"nnz={n_nz}, sparsity={1 - n_nz/X.size:.1%}")
    return X, labels


# ── Baseline methods ──────────────────────────────────────────────────────────

def run_spectralcc(X, labels, n_clusters, seed):
    """sklearn SpectralCoclustering (= Dhillon's SCC)."""
    from sklearn.cluster import SpectralCoclustering
    from scipy.sparse import csr_matrix

    X_sparse = csr_matrix(X) if not hasattr(X, 'toarray') else X
    model = SpectralCoclustering(n_clusters=n_clusters, random_state=seed,
                                 svd_method='arpack')
    t0 = time.time()
    model.fit(X_sparse)
    elapsed = time.time() - t0
    return model.row_labels_, elapsed


def run_nbvd(X, labels, n_clusters, seed):
    """NBVD = NMTF (Long 2005)."""
    NBVD = _load_nmtfcoclust_model("NMTFcoclust_NBVD.py", "NBVD")
    model = NBVD(n_row_clusters=n_clusters, n_col_clusters=n_clusters,
                 max_iter=100, n_init=1, tol=1e-9, random_state=seed)
    t0 = time.time()
    model.fit(X)
    elapsed = time.time() - t0
    row_labels = np.array(model.row_labels_) - 1  # 1-indexed -> 0-indexed
    return row_labels, elapsed


def run_onm3f(X, labels, n_clusters, seed):
    """ONM3F = ONMTF with Lagrange multipliers (Ding 2006/2008)."""
    ONM3F = _load_nmtfcoclust_model("NMTFcoclust_ONM3F.py", "ONM3F")
    model = ONM3F(n_row_clusters=n_clusters, n_col_clusters=n_clusters,
                  max_iter=100, n_init=1, tol=1e-9, random_state=seed)
    t0 = time.time()
    model.fit(X)
    elapsed = time.time() - t0
    row_labels = np.array(model.row_labels_) - 1
    return row_labels, elapsed


def run_onmtf(X, labels, n_clusters, seed):
    """ONMTF (Yoo 2010)."""
    ONMTF = _load_nmtfcoclust_model("NMTFcoclust_ONMTF.py", "ONMTF")
    model = ONMTF(n_row_clusters=n_clusters, n_col_clusters=n_clusters,
                  max_iter=100, n_init=1, tol=1e-9, random_state=seed)
    t0 = time.time()
    model.fit(X)
    elapsed = time.time() - t0
    row_labels = np.array(model.row_labels_) - 1
    return row_labels, elapsed


def run_pnmtf(X, labels, n_clusters, seed):
    """Penalized NMTF (Wang 2017). Proxy for Chen 2023 PNMTF."""
    PNMTF = _load_nmtfcoclust_model("NMTFcoclust_PNMTF.py", "PNMTF")
    model = PNMTF(n_row_clusters=n_clusters, n_col_clusters=n_clusters,
                  tau=0.5, eta=0.5, gamma=0.1,
                  max_iter=100, n_init=1, tol=1e-9, random_state=seed)
    t0 = time.time()
    model.fit(X)
    elapsed = time.time() - t0
    row_labels = np.array(model.row_labels_) - 1
    return row_labels, elapsed


def run_scc_dhillon(X, labels, n_clusters, seed):
    """SCC (Dhillon 2001) — faithful reproduction using k singular vectors.

    This matches our Rust implementation: D1^{-1/2} A D2^{-1/2}, take k SVs,
    form Z=[U;V], L2-normalize rows, k-means. sklearn uses ceil(log2(k)) SVs
    instead of k, which gives lower quality.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds
    from sklearn.preprocessing import normalize
    from sklearn.cluster import KMeans

    X_sp = csr_matrix(X) if not hasattr(X, 'toarray') else X
    eps = 1e-12

    # Bi-normalize: D1^{-1/2} A D2^{-1/2}
    row_sums = np.asarray(X_sp.sum(axis=1)).ravel()
    col_sums = np.asarray(X_sp.sum(axis=0)).ravel()
    du_inv = np.where(np.abs(row_sums) < eps, 0.0, row_sums ** -0.5)
    dv_inv = np.where(np.abs(col_sums) < eps, 0.0, col_sums ** -0.5)

    # Scale rows and columns
    from scipy.sparse import diags
    An = diags(du_inv) @ X_sp @ diags(dv_inv)

    # SVD with k singular vectors (not ceil(log2(k)))
    t0 = time.time()
    U, s, Vt = svds(An, k=n_clusters)
    V = Vt.T

    # Form embedding Z = [U; V], L2-normalize rows
    Z = np.vstack([U, V])
    Z = normalize(Z, norm='l2', axis=1)

    # k-means on the embedding
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    all_labels = km.fit_predict(Z)
    elapsed = time.time() - t0

    row_labels = all_labels[:X.shape[0]]
    return row_labels, elapsed


def run_fnmf(X, labels, n_clusters, seed):
    """Fast NMF (Kim & Park 2011) using ANLS-BPP. Row labels via argmax(W)."""
    nonnegfac_dir = BASELINES_DIR / "nonnegfac-python"
    if str(nonnegfac_dir) not in sys.path:
        sys.path.insert(0, str(nonnegfac_dir))
    from nonnegfac.nmf import NMF_ANLS_BLOCKPIVOT

    np.random.seed(seed)
    solver = NMF_ANLS_BLOCKPIVOT(default_max_iter=100)
    t0 = time.time()
    W, H, info = solver.run(X, n_clusters, verbose=-1)
    elapsed = time.time() - t0
    row_labels = np.argmax(W, axis=1)
    return row_labels, elapsed


# ── Registry ──────────────────────────────────────────────────────────────────

METHODS = {
    "SpectralCC": {
        "fn": run_spectralcc,
        "paper_name": "SpectralCC (sklearn, ceil(log2(k)) SVs)",
        "needs_norm": False,
    },
    "SCC-Dhillon": {
        "fn": run_scc_dhillon,
        "paper_name": "SCC (Dhillon 2001, k SVs, = our Rust impl)",
        "needs_norm": False,
    },
    "NBVD": {
        "fn": run_nbvd,
        "paper_name": "NMTF (Long 2005, via NMTFcoclust NBVD)",
        "needs_norm": False,
    },
    "ONM3F": {
        "fn": run_onm3f,
        "paper_name": "ONMTF (Ding 2006, via NMTFcoclust ONM3F)",
        "needs_norm": False,
    },
    "ONMTF": {
        "fn": run_onmtf,
        "paper_name": "ONMTF (Yoo 2010, via NMTFcoclust ONMTF)",
        "needs_norm": False,
    },
    "PNMTF": {
        "fn": run_pnmtf,
        "paper_name": "PNMTF (Wang 2017, proxy for Chen 2023)",
        "needs_norm": False,
    },
    "FNMF": {
        "fn": run_fnmf,
        "paper_name": "FNMTF (Kim & Park 2011, fast NMF + argmax)",
        "needs_norm": False,
    },
}


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(true_labels, pred_labels):
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    return nmi, ari


def run_single(method_name, X_raw, X_norm, labels, n_clusters, seed):
    info = METHODS[method_name]
    X = X_norm if info["needs_norm"] else X_raw

    try:
        pred_labels, elapsed = info["fn"](X, labels, n_clusters, seed)
        nmi, ari = evaluate(labels, pred_labels)
        return {
            "method": method_name,
            "paper_name": info["paper_name"],
            "seed": seed,
            "nmi": round(float(nmi), 4),
            "ari": round(float(ari), 4),
            "time_s": round(elapsed, 2),
            "error": None,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "method": method_name,
            "paper_name": info["paper_name"],
            "seed": seed,
            "nmi": None,
            "ari": None,
            "time_s": None,
            "error": str(e),
        }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run Classic4 baselines")
    parser.add_argument("--methods", type=str, default=",".join(METHODS.keys()),
                        help=f"Comma-separated methods: {list(METHODS.keys())}")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9",
                        help="Comma-separated random seeds (default: 0-9)")
    parser.add_argument("--n-clusters", type=int, default=4,
                        help="Number of co-clusters (default: 4)")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    for m in methods:
        if m not in METHODS:
            print(f"Unknown method: {m}. Available: {list(METHODS.keys())}")
            sys.exit(1)

    # Load data
    print("=" * 70)
    print(f"Classic4 Baseline Experiments — {datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(f"Methods: {methods}")
    print(f"Seeds: {seeds}")
    print("=" * 70)

    X_raw, labels = load_classic4()
    X_norm = X_raw / (X_raw.sum() + 1e-16)

    results = []
    for method_name in methods:
        print(f"\n--- {method_name} ({METHODS[method_name]['paper_name']}) ---")
        for seed in seeds:
            print(f"  seed={seed} ...", end=" ", flush=True)
            r = run_single(method_name, X_raw, X_norm, labels, args.n_clusters, seed)
            results.append(r)
            if r["error"]:
                print(f"FAILED: {r['error']}")
            else:
                print(f"NMI={r['nmi']:.4f}  ARI={r['ari']:.4f}  Time={r['time_s']:.1f}s")

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
              f"{np.mean(times):>7.1f} +/- {np.std(times):.1f}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "classic4_baselines.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.datetime.now().isoformat(),
            "dataset": "classic4_paper",
            "shape": [6460, 4667],
            "n_clusters": args.n_clusters,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
