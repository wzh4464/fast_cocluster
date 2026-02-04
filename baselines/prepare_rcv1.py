#!/usr/bin/env python3
"""Download RCV1, convert to single-label, save dense .npy for Rust + sparse .npz for Python."""
import numpy as np
from pathlib import Path
from scipy.sparse import save_npz

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "rcv1"
DATA_DIR.mkdir(parents=True, exist_ok=True)

from sklearn.datasets import fetch_rcv1

for subset in ["train"]:
    print(f"=== Fetching RCV1 subset={subset} ===")
    rcv1 = fetch_rcv1(subset=subset, shuffle=False)
    X = rcv1.data
    Y = rcv1.target

    # Top-level categories
    top_cats = ['CCAT', 'ECAT', 'GCAT', 'MCAT']
    cat_indices = [list(rcv1.target_names).index(c) for c in top_cats]
    Y_top = Y[:, cat_indices].toarray()

    labels = np.full(X.shape[0], -1, dtype=int)
    for i in range(X.shape[0]):
        if Y_top[i].sum() > 0:
            labels[i] = np.argmax(Y_top[i])

    mask = labels >= 0
    X_f = X[mask]
    labels_f = labels[mask]

    print(f"  Shape: {X_f.shape}, nnz: {X_f.nnz}")
    print(f"  Sparsity: {1 - X_f.nnz / (X_f.shape[0] * X_f.shape[1]):.4%}")
    for i, c in enumerate(top_cats):
        print(f"  {c}: {(labels_f == i).sum()}")

    # Save sparse
    save_npz(DATA_DIR / f"rcv1_{subset}.npz", X_f)
    np.save(DATA_DIR / f"rcv1_{subset}_labels.npy", labels_f)

    # Save dense .npy for Rust (only if feasible)
    dense_bytes = X_f.shape[0] * X_f.shape[1] * 8
    print(f"  Dense size: {dense_bytes / 1e9:.1f} GB")
    if dense_bytes < 16e9:  # < 16 GB
        print(f"  Saving dense .npy ...")
        np.save(DATA_DIR / f"rcv1_{subset}.npy", X_f.toarray())
        print(f"  Done.")
    else:
        print(f"  Too large for dense, skipping .npy")

print("\nAll done. Files in", DATA_DIR)
