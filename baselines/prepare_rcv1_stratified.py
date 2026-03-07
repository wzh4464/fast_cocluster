#!/usr/bin/env python3
"""Generate stratified 50K subsample of RCV1-all for Rust evaluation.

The full RCV1-all has 804K rows (too large for dense .npy at ~304 GB).
This script creates a 50K stratified subsample preserving the label
distribution of the full dataset.

Usage:
    python baselines/prepare_rcv1_stratified.py
"""
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_rcv1

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "rcv1"
DATA_DIR.mkdir(parents=True, exist_ok=True)

N_TARGET = 50000
SEED = 0

print("Fetching RCV1 (subset=all)...")
rcv1 = fetch_rcv1(subset="all", shuffle=False)
X = rcv1.data
Y = rcv1.target

top_cats = ["CCAT", "ECAT", "GCAT", "MCAT"]
cat_indices = [list(rcv1.target_names).index(c) for c in top_cats]
Y_top = Y[:, cat_indices].toarray()

labels = np.full(X.shape[0], -1, dtype=int)
for i in range(X.shape[0]):
    if Y_top[i].sum() > 0:
        labels[i] = np.argmax(Y_top[i])

mask = labels >= 0
X_full = X[mask]
labels_full = labels[mask]
print(f"Full RCV1-all: {X_full.shape}, label dist: {np.bincount(labels_full)}")

# Stratified subsample
np.random.seed(SEED)
indices = []
for c in range(4):
    class_idx = np.where(labels_full == c)[0]
    n_class = int(N_TARGET * len(class_idx) / len(labels_full))
    chosen = np.random.choice(class_idx, size=n_class, replace=False)
    indices.extend(chosen)

remaining = N_TARGET - len(indices)
all_idx = set(range(len(labels_full)))
used = set(indices)
extra = np.random.choice(list(all_idx - used), size=remaining, replace=False)
indices.extend(extra)
indices = np.sort(np.array(indices))

X_sub = X_full[indices]
labels_sub = labels_full[indices]
print(f"Subsample: {X_sub.shape}, label dist: {np.bincount(labels_sub)}")

# Save
X_dense = X_sub.toarray()
np.save(DATA_DIR / "rcv1_all.npy", X_dense)
np.save(DATA_DIR / "rcv1_all_labels.npy", labels_sub)
print(f"Saved to {DATA_DIR}/rcv1_all.npy ({X_dense.nbytes / 1e9:.1f} GB)")
