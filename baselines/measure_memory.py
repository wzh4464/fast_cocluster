#!/usr/bin/env python3
"""Measure peak RSS for FNMF on RCV1-all and Amazon."""
import os
import sys
import numpy as np
from scipy import sparse
from sklearn.decomposition import NMF

def get_peak_rss_mb():
    """Get peak RSS from /proc/self/status VmHWM."""
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024  # kB to MB
    return -1

def measure_fnmf(X, k, name):
    print(f"\n=== {name} FNMF (k={k}) ===")
    print(f"Shape: {X.shape}, NNZ: {X.nnz:,}")
    rss_before = get_peak_rss_mb()
    print(f"Peak RSS before: {rss_before:.0f} MB")

    nmf = NMF(n_components=k, init="nndsvda", random_state=0, max_iter=100)
    W = nmf.fit_transform(X)
    pred = np.argmax(W, axis=1)

    rss_after = get_peak_rss_mb()
    print(f"Peak RSS after:  {rss_after:.0f} MB")
    print(f"Result: VmHWM = {rss_after:.0f} MB")

if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "rcv1"

    if dataset == "rcv1":
        X = sparse.load_npz("data/rcv1/rcv1_all_sparse.npz")
        measure_fnmf(X, 4, "RCV1-all")
    elif dataset == "amazon":
        X = sparse.load_npz("data/amazon_24cat_sparse.npz")
        measure_fnmf(X, 24, "Amazon 24-cat")
