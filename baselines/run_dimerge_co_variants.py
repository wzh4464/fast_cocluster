#!/usr/bin/env python3
"""
DiMergeCo framework in Python with pluggable atom co-clustering methods.

Implements the DiMergeCo algorithm (random block partitioning + local co-clustering
+ consensus via membership matrix + k-means) and evaluates with multiple atom methods:
  SCC-Dhillon, SpectralCC, NBVD, ONM3F, ONMTF, PNMTF, FNMF

Usage:
    python run_dimerge_co_variants.py --dataset classic4 --methods SCC-Dhillon,FNMF
    python run_dimerge_co_variants.py --dataset rcv1 --methods SCC-Dhillon,FNMF,NBVD
    python run_dimerge_co_variants.py --dataset classic4 --methods all
"""

import sys
import os
import time
import json
import types
import argparse
import datetime
import numpy as np
from itertools import product
from pathlib import Path
from scipy import sparse
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans

# ── Paths ─────────────────────────────────────────────────────────────────────

BASELINES_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = BASELINES_DIR / "results"

# ── NMTFcoclust import machinery (shared with other baseline scripts) ─────────

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


# ── DiMergeCo framework ──────────────────────────────────────────────────────

def random_split(indices, n_groups, rng):
    """Shuffle indices and split into n_groups roughly equal groups.

    Returns list of np.ndarray, each containing sorted global indices.
    Matches Rust implementation: split_into_groups + sort within each group.
    """
    perm = indices.copy()
    rng.shuffle(perm)
    groups = np.array_split(perm, n_groups)
    # Sort within each group (matches Rust's split_into_groups behavior)
    return [np.sort(g) for g in groups]


class DiMergeCo:
    """DiMergeCo: Divide-and-Merge Co-clustering framework.

    For each of T_p iterations:
      1. Randomly partition rows into m_blocks groups and cols into n_blocks groups
      2. For each (row_group, col_group) sub-matrix, run the atom co-clusterer
      3. Collect all local co-clusters with their global row indices

    Then build a membership matrix (n_rows x total_coclusters) and run k-means
    on it to produce final row labels.
    """

    def __init__(self, atom_fn, k, m_blocks=2, n_blocks=2, t_p=10):
        """
        Parameters
        ----------
        atom_fn : callable
            atom_fn(X_sub, k, seed) -> row_labels (np.ndarray of length X_sub.shape[0])
            Must return integer labels in [0, k).
        k : int
            Number of clusters.
        m_blocks : int
            Number of row blocks per random partition.
        n_blocks : int
            Number of column blocks per random partition.
        t_p : int
            Number of random partitioning iterations.
        """
        self.atom_fn = atom_fn
        self.k = k
        self.m_blocks = m_blocks
        self.n_blocks = n_blocks
        self.t_p = t_p

    def run(self, X, seed=0):
        """Run DiMergeCo on matrix X.

        Parameters
        ----------
        X : np.ndarray or scipy.sparse matrix
            Data matrix (n_rows x n_cols).
        seed : int
            Base random seed.

        Returns
        -------
        row_labels : np.ndarray of shape (n_rows,)
            Cluster assignments for each row.
        elapsed : float
            Wall-clock time in seconds.
        """
        t0 = time.time()

        n_rows, n_cols = X.shape
        row_indices = np.arange(n_rows)
        col_indices = np.arange(n_cols)

        all_coclusters = []  # list of np.ndarray of global row indices

        for t in range(self.t_p):
            # Deterministic RNG per iteration (matches Rust: seed_from_u64)
            rng_rows = np.random.RandomState(seed * 1000 + t)
            rng_cols = np.random.RandomState(seed * 1000 + t + 500)

            row_groups = random_split(row_indices, self.m_blocks, rng_rows)
            col_groups = random_split(col_indices, self.n_blocks, rng_cols)

            for rg, cg in product(row_groups, col_groups):
                if len(rg) == 0 or len(cg) == 0:
                    continue

                # Extract sub-matrix
                if sparse.issparse(X):
                    X_sub = X[np.ix_(rg, cg)]
                else:
                    X_sub = X[np.ix_(rg, cg)]

                # Run atom co-clusterer
                try:
                    local_labels = self.atom_fn(X_sub, self.k, seed=t)
                except Exception as e:
                    print(f"    [warn] atom failed on block "
                          f"({len(rg)}x{len(cg)}), iter {t}: {e}")
                    # Fall back: assign all to cluster 0
                    local_labels = np.zeros(len(rg), dtype=int)

                # Collect membership: which global rows belong to each local cluster
                for label in range(self.k):
                    mask = local_labels == label
                    if mask.any():
                        all_coclusters.append(rg[mask])

        n_coclusters = len(all_coclusters)
        print(f"    DiMergeCo: {n_coclusters} local co-clusters from "
              f"{self.t_p} iters x {self.m_blocks}x{self.n_blocks} blocks x {self.k} clusters")

        if n_coclusters == 0:
            print("    [warn] no co-clusters produced, returning all-zeros labels")
            elapsed = time.time() - t0
            return np.zeros(n_rows, dtype=int), elapsed

        # Build membership matrix: n_rows x n_coclusters (sparse for memory)
        rows_list, cols_list = [], []
        for c, indices in enumerate(all_coclusters):
            rows_list.append(indices)
            cols_list.append(np.full(len(indices), c))

        rows_flat = np.concatenate(rows_list)
        cols_flat = np.concatenate(cols_list)
        data = np.ones(len(rows_flat))
        M = sparse.csr_matrix((data, (rows_flat, cols_flat)),
                              shape=(n_rows, n_coclusters))

        print(f"    Membership matrix: {M.shape}, nnz={M.nnz}")

        # K-means on membership matrix -> final labels
        # Convert to dense for k-means (membership matrices are typically small)
        M_dense = M.toarray()
        km = KMeans(n_clusters=self.k, n_init=10, random_state=seed)
        row_labels = km.fit_predict(M_dense)

        elapsed = time.time() - t0
        return row_labels, elapsed


# ── Atom co-clustering functions ──────────────────────────────────────────────
# Each: atom_fn(X_sub, k, seed) -> row_labels (np.ndarray of length X_sub.shape[0])

def atom_scc_dhillon(X_sub, k, seed=0):
    """SCC (Dhillon 2001) — sparse SVD, bi-normalize, k singular vectors."""
    from scipy.sparse import diags, csr_matrix
    from scipy.sparse.linalg import svds
    from sklearn.preprocessing import normalize

    X_sp = csr_matrix(X_sub) if not sparse.issparse(X_sub) else X_sub
    eps = 1e-12

    row_sums = np.asarray(X_sp.sum(axis=1)).ravel()
    col_sums = np.asarray(X_sp.sum(axis=0)).ravel()

    # Skip if matrix is all zeros
    if row_sums.max() < eps:
        return np.zeros(X_sp.shape[0], dtype=int)

    with np.errstate(divide='ignore', invalid='ignore'):
        du_inv = np.where(np.abs(row_sums) < eps, 0.0, row_sums ** -0.5)
        dv_inv = np.where(np.abs(col_sums) < eps, 0.0, col_sums ** -0.5)
    An = diags(du_inv) @ X_sp @ diags(dv_inv)

    # k singular vectors (cap at min dimension - 1 for ARPACK)
    n_svs = min(k, min(An.shape) - 1)
    if n_svs < 1:
        return np.zeros(X_sp.shape[0], dtype=int)

    U, s, Vt = svds(An, k=n_svs)
    V = Vt.T

    Z = np.vstack([U, V])
    Z = normalize(Z, norm='l2', axis=1)

    km = KMeans(n_clusters=k, n_init=5, random_state=seed)
    all_labels = km.fit_predict(Z)
    return all_labels[:X_sp.shape[0]]


def atom_spectralcc(X_sub, k, seed=0):
    """sklearn SpectralCoclustering — uses ceil(log2(k)) SVs."""
    from sklearn.cluster import SpectralCoclustering
    from scipy.sparse import csr_matrix

    X_sp = csr_matrix(X_sub) if not sparse.issparse(X_sub) else csr_matrix(X_sub)

    model = SpectralCoclustering(n_clusters=k, random_state=seed,
                                 svd_method='arpack')
    try:
        model.fit(X_sp)
        return model.row_labels_
    except Exception:
        # Fallback to SCC-Dhillon if sklearn fails (e.g. on degenerate blocks)
        return atom_scc_dhillon(X_sub, k, seed)


def atom_nbvd(X_sub, k, seed=0):
    """NBVD = NMTF (Long 2005)."""
    NBVD = _load_nmtfcoclust_model("NMTFcoclust_NBVD.py", "NBVD")
    X_dense = X_sub.toarray() if sparse.issparse(X_sub) else np.asarray(X_sub)
    model = NBVD(n_row_clusters=k, n_col_clusters=k,
                 max_iter=100, n_init=1, tol=1e-9, random_state=seed)
    model.fit(X_dense)
    return np.array(model.row_labels_) - 1  # 1-indexed -> 0-indexed


def atom_onm3f(X_sub, k, seed=0):
    """ONM3F = ONMTF with Lagrange multipliers (Ding 2006)."""
    ONM3F = _load_nmtfcoclust_model("NMTFcoclust_ONM3F.py", "ONM3F")
    X_dense = X_sub.toarray() if sparse.issparse(X_sub) else np.asarray(X_sub)
    model = ONM3F(n_row_clusters=k, n_col_clusters=k,
                  max_iter=100, n_init=1, tol=1e-9, random_state=seed)
    model.fit(X_dense)
    return np.array(model.row_labels_) - 1


def atom_onmtf(X_sub, k, seed=0):
    """ONMTF (Yoo 2010)."""
    ONMTF = _load_nmtfcoclust_model("NMTFcoclust_ONMTF.py", "ONMTF")
    X_dense = X_sub.toarray() if sparse.issparse(X_sub) else np.asarray(X_sub)
    model = ONMTF(n_row_clusters=k, n_col_clusters=k,
                  max_iter=100, n_init=1, tol=1e-9, random_state=seed)
    model.fit(X_dense)
    return np.array(model.row_labels_) - 1


def atom_pnmtf(X_sub, k, seed=0):
    """Penalized NMTF (Wang 2017)."""
    PNMTF = _load_nmtfcoclust_model("NMTFcoclust_PNMTF.py", "PNMTF")
    X_dense = X_sub.toarray() if sparse.issparse(X_sub) else np.asarray(X_sub)
    model = PNMTF(n_row_clusters=k, n_col_clusters=k,
                  tau=0.5, eta=0.5, gamma=0.1,
                  max_iter=100, n_init=1, tol=1e-9, random_state=seed)
    model.fit(X_dense)
    return np.array(model.row_labels_) - 1


def atom_fnmf(X_sub, k, seed=0):
    """Fast NMF (Kim & Park 2011) via ANLS-BPP. Row labels via argmax(W)."""
    nonnegfac_dir = BASELINES_DIR / "nonnegfac-python"
    if str(nonnegfac_dir) not in sys.path:
        sys.path.insert(0, str(nonnegfac_dir))
    from nonnegfac.nmf import NMF_ANLS_BLOCKPIVOT

    X_dense = X_sub.toarray() if sparse.issparse(X_sub) else np.asarray(X_sub)
    np.random.seed(seed)
    solver = NMF_ANLS_BLOCKPIVOT(default_max_iter=100)
    W, H, info = solver.run(X_dense, k, verbose=-1)
    return np.argmax(W, axis=1)


# ── Method registry ──────────────────────────────────────────────────────────

ATOM_METHODS = {
    "SCC-Dhillon": {
        "fn": atom_scc_dhillon,
        "paper_name": "DiMergeCo + SCC (Dhillon 2001)",
        "sparse": True,
    },
    "SpectralCC": {
        "fn": atom_spectralcc,
        "paper_name": "DiMergeCo + SpectralCC (sklearn)",
        "sparse": True,
    },
    "NBVD": {
        "fn": atom_nbvd,
        "paper_name": "DiMergeCo + NMTF (Long 2005)",
        "sparse": False,
    },
    "ONM3F": {
        "fn": atom_onm3f,
        "paper_name": "DiMergeCo + ONMTF (Ding 2006)",
        "sparse": False,
    },
    "ONMTF": {
        "fn": atom_onmtf,
        "paper_name": "DiMergeCo + ONMTF (Yoo 2010)",
        "sparse": False,
    },
    "PNMTF": {
        "fn": atom_pnmtf,
        "paper_name": "DiMergeCo + PNMTF (Wang 2017)",
        "sparse": False,
    },
    "FNMF": {
        "fn": atom_fnmf,
        "paper_name": "DiMergeCo + FNMF (Kim & Park 2011)",
        "sparse": False,
    },
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_classic4():
    """Load Classic4 paper version (6460 x 4667) with labels."""
    X = np.load(DATA_DIR / "classic4_paper.npy")
    labels = np.load(DATA_DIR / "classic4_paper_labels.npy")
    n_nz = np.count_nonzero(X)
    print(f"Loaded Classic4: {X.shape}, {len(np.unique(labels))} classes, "
          f"nnz={n_nz}, sparsity={1 - n_nz/X.size:.1%}")
    return X, labels


def load_rcv1(subset="train"):
    """Load RCV1 and convert multi-label to single top-level category."""
    from sklearn.datasets import fetch_rcv1
    print(f"Fetching RCV1 (subset={subset})...")
    rcv1 = fetch_rcv1(subset=subset, shuffle=False)
    X = rcv1.data
    Y = rcv1.target

    top_cats = ['CCAT', 'ECAT', 'GCAT', 'MCAT']
    cat_indices = [list(rcv1.target_names).index(cat) for cat in top_cats]

    Y_top = Y[:, cat_indices].toarray()
    labels = np.full(X.shape[0], -1, dtype=int)
    for i in range(X.shape[0]):
        if Y_top[i].sum() > 0:
            labels[i] = np.argmax(Y_top[i])

    mask = labels >= 0
    X_filtered = X[mask]
    labels_filtered = labels[mask]

    print(f"RCV1 loaded: {X_filtered.shape[0]} docs x {X_filtered.shape[1]} terms, "
          f"sparsity={1 - X_filtered.nnz / (X_filtered.shape[0] * X_filtered.shape[1]):.4%}")
    for i, cat in enumerate(top_cats):
        print(f"  {cat}: {(labels_filtered == i).sum()} docs")

    return X_filtered, labels_filtered


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(true_labels, pred_labels):
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    return nmi, ari


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DiMergeCo with multiple atom co-clustering methods")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["classic4", "rcv1"],
                        help="Dataset to evaluate on")
    parser.add_argument("--methods", type=str, default="all",
                        help="Comma-separated atom methods, or 'all'")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9",
                        help="Comma-separated random seeds (default: 0-9)")
    parser.add_argument("--n-clusters", type=int, default=4,
                        help="Number of co-clusters (default: 4)")
    parser.add_argument("--m-blocks", type=int, default=2,
                        help="Number of row blocks (default: 2)")
    parser.add_argument("--n-blocks", type=int, default=2,
                        help="Number of column blocks (default: 2)")
    parser.add_argument("--t-p", type=int, default=10,
                        help="Number of partitioning iterations (default: 10)")
    parser.add_argument("--rcv1-subset", type=str, default="train",
                        choices=["train", "test", "all"])
    args = parser.parse_args()

    if args.methods == "all":
        methods = list(ATOM_METHODS.keys())
    else:
        methods = [m.strip() for m in args.methods.split(",")]

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    for m in methods:
        if m not in ATOM_METHODS:
            print(f"Unknown method: {m}. Available: {list(ATOM_METHODS.keys())}")
            sys.exit(1)

    # Load data
    print("=" * 70)
    print(f"DiMergeCo Variants — {args.dataset} — {datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(f"Methods: {methods}")
    print(f"Seeds: {seeds}")
    print(f"Blocks: {args.m_blocks}x{args.n_blocks}, T_p={args.t_p}, k={args.n_clusters}")
    print("=" * 70)

    if args.dataset == "classic4":
        X, labels = load_classic4()
    else:
        X, labels = load_rcv1(args.rcv1_subset)

    results = []
    for method_name in methods:
        info = ATOM_METHODS[method_name]
        print(f"\n--- DiMergeCo + {method_name} ({info['paper_name']}) ---")

        for seed in seeds:
            print(f"  seed={seed} ...", flush=True)
            try:
                dmc = DiMergeCo(
                    atom_fn=info["fn"],
                    k=args.n_clusters,
                    m_blocks=args.m_blocks,
                    n_blocks=args.n_blocks,
                    t_p=args.t_p,
                )
                pred_labels, elapsed = dmc.run(X, seed=seed)
                nmi, ari = evaluate(labels, pred_labels)
                r = {
                    "method": f"DiMergeCo+{method_name}",
                    "atom": method_name,
                    "paper_name": info["paper_name"],
                    "seed": seed,
                    "nmi": round(float(nmi), 4),
                    "ari": round(float(ari), 4),
                    "time_s": round(elapsed, 2),
                    "m_blocks": args.m_blocks,
                    "n_blocks": args.n_blocks,
                    "t_p": args.t_p,
                    "error": None,
                }
                print(f"    NMI={nmi:.4f}  ARI={ari:.4f}  Time={elapsed:.1f}s")
            except Exception as e:
                import traceback
                traceback.print_exc()
                r = {
                    "method": f"DiMergeCo+{method_name}",
                    "atom": method_name,
                    "paper_name": info["paper_name"],
                    "seed": seed,
                    "nmi": None,
                    "ari": None,
                    "time_s": None,
                    "m_blocks": args.m_blocks,
                    "n_blocks": args.n_blocks,
                    "t_p": args.t_p,
                    "error": str(e),
                }
                print(f"    FAILED: {e}")
            results.append(r)

    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY — DiMergeCo on {args.dataset} "
          f"(blocks={args.m_blocks}x{args.n_blocks}, T_p={args.t_p})")
    print("=" * 70)
    print(f"{'Method':<28} {'NMI':>16} {'ARI':>16} {'Time (s)':>16}")
    print("-" * 80)

    for method_name in methods:
        full_name = f"DiMergeCo+{method_name}"
        ok = [r for r in results if r["method"] == full_name and r["error"] is None]
        if not ok:
            print(f"{full_name:<28} {'ALL FAILED':>16}")
            continue
        nmis = [r["nmi"] for r in ok]
        aris = [r["ari"] for r in ok]
        times = [r["time_s"] for r in ok]
        print(f"{full_name:<28} "
              f"{np.mean(nmis):.3f} +/- {np.std(nmis):.3f}  "
              f"{np.mean(aris):.3f} +/- {np.std(aris):.3f}  "
              f"{np.mean(times):>7.1f} +/- {np.std(times):.1f}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_tag = args.dataset if args.dataset == "classic4" else f"rcv1_{args.rcv1_subset}"
    out_path = RESULTS_DIR / f"{dataset_tag}_dimerge_co_variants.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.datetime.now().isoformat(),
            "dataset": dataset_tag,
            "n_clusters": args.n_clusters,
            "m_blocks": args.m_blocks,
            "n_blocks": args.n_blocks,
            "t_p": args.t_p,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
