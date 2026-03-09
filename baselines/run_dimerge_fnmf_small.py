#!/usr/bin/env python3
"""Run DiMergeCo-FNMF on Classic4 and BCW (small datasets, 10 seeds)."""

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

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def fnmf_atom(X_sub, k, seed=0):
    if sparse.issparse(X_sub):
        X_sub_dense = X_sub.toarray()
    else:
        X_sub_dense = X_sub
    X_sub_dense = np.maximum(X_sub_dense, 0)  # ensure non-negative
    nmf = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=200)
    W = nmf.fit_transform(X_sub_dense)
    return np.argmax(W, axis=1)


def load_classic4():
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    data_dir = Path(__file__).resolve().parent.parent / "data"
    X = sparse.load_npz(str(data_dir / "classic4" / "classic4.npz"))
    labels = np.load(str(data_dir / "classic4" / "classic4_labels.npy"))
    return X, labels, 4


def load_bcw():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = data.data
    labels = data.target
    return X, labels, 2


def run_dimerge_fnmf(X, labels, k, name, seeds=range(10)):
    results = []
    print(f"\n=== DiMergeCo-FNMF on {name} (2x2, t_p=10, {len(list(seeds))} seeds) ===")
    for seed in seeds:
        print(f"  seed={seed}...", end=" ", flush=True)
        try:
            dm = DiMergeCo(fnmf_atom, k, m_blocks=2, n_blocks=2, t_p=10)
            pred, elapsed = dm.run(X, seed=seed)
            nmi = normalized_mutual_info_score(labels, pred)
            ari = adjusted_rand_score(labels, pred)
            print(f"NMI={nmi:.4f}, ARI={ari:.4f}, Time={elapsed:.1f}s")
            results.append(dict(
                method="DiMergeCo-FNMF", dataset=name, seed=seed,
                nmi=nmi, ari=ari, time_s=elapsed, error=None,
                m_blocks=2, n_blocks=2, t_p=10,
            ))
        except Exception as e:
            print(f"FAILED: {e}")
            results.append(dict(
                method="DiMergeCo-FNMF", dataset=name, seed=seed,
                nmi=None, ari=None, time_s=None, error=str(e),
            ))
    return results


def main():
    all_results = []

    # Classic4
    try:
        X_c4, labels_c4, k_c4 = load_classic4()
        print(f"Classic4: {X_c4.shape}, k={k_c4}")
        all_results.extend(run_dimerge_fnmf(X_c4, labels_c4, k_c4, "classic4"))
    except Exception as e:
        print(f"Classic4 load failed: {e}")

    # BCW
    try:
        X_bcw, labels_bcw, k_bcw = load_bcw()
        print(f"\nBCW: {X_bcw.shape if hasattr(X_bcw, 'shape') else 'unknown'}, k={k_bcw}")
        all_results.extend(run_dimerge_fnmf(X_bcw, labels_bcw, k_bcw, "bcw"))
    except Exception as e:
        print(f"BCW load failed: {e}")

    # Save
    output = {
        "experiment": "DiMergeCo-FNMF on small datasets",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": all_results,
    }
    out_path = RESULTS_DIR / "dimerge_fnmf_small.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 60)
    for dataset in ["classic4", "bcw"]:
        ok = [r for r in all_results if r["dataset"] == dataset and r.get("nmi") is not None]
        if ok:
            nmis = [r["nmi"] for r in ok]
            aris = [r["ari"] for r in ok]
            print(f"DiMergeCo-FNMF {dataset}: NMI={np.mean(nmis):.3f}+/-{np.std(nmis):.3f}  "
                  f"ARI={np.mean(aris):.3f}+/-{np.std(aris):.3f}  n={len(ok)}")


if __name__ == "__main__":
    main()
