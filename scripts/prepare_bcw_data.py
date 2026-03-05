#!/usr/bin/env python3
"""Download Breast Cancer Wisconsin (BCW) dataset and save as .npy files."""

import numpy as np
from pathlib import Path
from sklearn.datasets import load_breast_cancer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    data = load_breast_cancer()
    X = data.data.astype(np.float64)
    labels = data.target.astype(np.int64)

    np.save(DATA_DIR / "bcw.npy", X)
    np.save(DATA_DIR / "bcw_labels.npy", labels)

    print(f"BCW saved: shape={X.shape}, classes={len(set(labels))}")
    print(f"  Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    print(f"  Files: {DATA_DIR / 'bcw.npy'}, {DATA_DIR / 'bcw_labels.npy'}")


if __name__ == "__main__":
    main()
