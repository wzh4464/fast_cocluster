#!/usr/bin/env python3
"""
Generate NMI/ARI baselines for atom co-clustering methods on Classic4.

Runs each Python NMF method (NBVD, ONM3F, ONMTF, PNMTF, FNMF) with seeds 0-9,
computes NMI and ARI, and saves results as JSON for Rust consistency checks.

Usage:
    uv run python scripts/generate_atom_baselines.py

Output:
    data/atom_baselines/{method}_classic4.json
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

# Resolve repo root from this script's location
_REPO_ROOT = Path(__file__).resolve().parent.parent
_BASELINES_DIR = _REPO_ROOT / "baselines"

# --- Stub modules for NMTFcoclust relative imports ---
import types

# Create stub NMTFcoclust package structure to satisfy relative imports
nmtf_pkg = types.ModuleType("NMTFcoclust")
nmtf_pkg.__path__ = []
sys.modules["NMTFcoclust"] = nmtf_pkg

# Stub initialization module
init_mod = types.ModuleType("NMTFcoclust.initialization")


def random_init(n_clusters, X, random_state):
    """Stub - not actually called since F_init/G_init/S_init are None"""
    pass


init_mod.random_init = random_init
sys.modules["NMTFcoclust.initialization"] = init_mod

# Stub io.input_checking module
io_pkg = types.ModuleType("NMTFcoclust.io")
io_pkg.__path__ = []
sys.modules["NMTFcoclust.io"] = io_pkg

io_check = types.ModuleType("NMTFcoclust.io.input_checking")


def check_positive(X):
    pass


io_check.check_positive = check_positive
sys.modules["NMTFcoclust.io.input_checking"] = io_check

# Models subpackage — use local baselines/ copy
models_pkg = types.ModuleType("NMTFcoclust.Models")
models_pkg.__path__ = [str(_BASELINES_DIR / "NMTFcoclust" / "Models")]
sys.modules["NMTFcoclust.Models"] = models_pkg

# Now we can import the models
from NMTFcoclust.Models.NMTFcoclust_NBVD import NBVD
from NMTFcoclust.Models.NMTFcoclust_ONM3F import ONM3F
from NMTFcoclust.Models.NMTFcoclust_ONMTF import ONMTF
from NMTFcoclust.Models.NMTFcoclust_PNMTF import PNMTF

# FNMF — use local baselines/ copy
sys.path.insert(0, str(_BASELINES_DIR))
from nonnegfac.nmf import NMF as FNMF


def load_classic4():
    """Load Classic4 small benchmark dataset."""
    data_path = "data/classic4_benchmark_small.npy"
    labels_path = "data/classic4_benchmark_small_labels.npy"
    X = np.load(data_path)
    labels = np.load(labels_path).astype(int)
    print(f"Loaded Classic4: {X.shape[0]} x {X.shape[1]}, {len(labels)} labels")
    return X, labels


def run_tri_factor_method(cls, X, true_labels, n_seeds=10, n_row_clusters=4,
                          n_col_clusters=4, max_iter=100, **kwargs):
    """Run a tri-factor NMF method with multiple seeds and compute metrics."""
    results = {"seeds": [], "nmis": [], "aris": []}

    for seed in range(n_seeds):
        model = cls(
            n_row_clusters=n_row_clusters,
            n_col_clusters=n_col_clusters,
            max_iter=max_iter,
            n_init=1,
            tol=1e-9,
            random_state=seed,
            **kwargs
        )
        model.fit(X)

        # row_labels_ is 1-indexed in Python baselines, convert to 0-indexed
        pred = np.array(model.row_labels_) - 1
        nmi = normalized_mutual_info_score(true_labels, pred)
        ari = adjusted_rand_score(true_labels, pred)

        results["seeds"].append(seed)
        results["nmis"].append(float(nmi))
        results["aris"].append(float(ari))
        print(f"  seed {seed}: NMI={nmi:.4f}, ARI={ari:.4f}")

    results["mean_nmi"] = float(np.mean(results["nmis"]))
    results["mean_ari"] = float(np.mean(results["aris"]))
    results["std_nmi"] = float(np.std(results["nmis"]))
    results["std_ari"] = float(np.std(results["aris"]))
    return results


def run_fnmf(X, true_labels, n_seeds=10, k=4, max_iter=50):
    """Run FNMF (ANLS-BPP) with multiple seeds and compute metrics."""
    results = {"seeds": [], "nmis": [], "aris": []}
    alg = FNMF(default_max_iter=max_iter)

    for seed in range(n_seeds):
        np.random.seed(seed)
        W, H, rec = alg.run(X, k, verbose=-1)
        pred = np.argmax(W, axis=1)
        nmi = normalized_mutual_info_score(true_labels, pred)
        ari = adjusted_rand_score(true_labels, pred)

        results["seeds"].append(seed)
        results["nmis"].append(float(nmi))
        results["aris"].append(float(ari))
        print(f"  seed {seed}: NMI={nmi:.4f}, ARI={ari:.4f}")

    results["mean_nmi"] = float(np.mean(results["nmis"]))
    results["mean_ari"] = float(np.mean(results["aris"]))
    results["std_nmi"] = float(np.std(results["nmis"]))
    results["std_ari"] = float(np.std(results["aris"]))
    return results


def main():
    os.chdir(str(_REPO_ROOT))
    X, true_labels = load_classic4()

    out_dir = Path("data/atom_baselines")
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = [
        ("nbvd", NBVD, {}),
        ("onm3f", ONM3F, {}),
        ("onmtf", ONMTF, {}),
        ("pnmtf", PNMTF, {"tau": 0.1, "eta": 0.1, "gamma": 0.1}),
    ]

    for name, cls, kwargs in methods:
        print(f"\n{'='*50}")
        print(f"Running {name.upper()}...")
        print(f"{'='*50}")
        results = run_tri_factor_method(cls, X, true_labels, **kwargs)
        out_path = out_dir / f"{name}_classic4.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n{name.upper()}: NMI={results['mean_nmi']:.4f}+-{results['std_nmi']:.4f}, "
              f"ARI={results['mean_ari']:.4f}+-{results['std_ari']:.4f}")
        print(f"Saved to {out_path}")

    # FNMF
    print(f"\n{'='*50}")
    print("Running FNMF...")
    print(f"{'='*50}")
    results = run_fnmf(X, true_labels)
    out_path = out_dir / "fnmf_classic4.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFNMF: NMI={results['mean_nmi']:.4f}+-{results['std_nmi']:.4f}, "
          f"ARI={results['mean_ari']:.4f}+-{results['std_ari']:.4f}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
