#!/usr/bin/env python3
"""
Create full Classic4 benchmark datasets for reproducing paper experiments.

Creates two versions:
- full: All 7094 docs, top-5000 features by variance
- paper: Pruned to ~6461 docs x ~4667 features (matching paper's dimensions)
"""

import numpy as np
from pathlib import Path
from scipy.io import mmread
from scipy.sparse import csr_matrix
import json


def create_full_benchmarks():
    data_dir = Path(__file__).parent.parent / 'data'

    # Load full sparse matrix (TF-IDF)
    print("Loading full Classic4 sparse matrix...")
    full_sparse: csr_matrix = mmread(data_dir / 'classic4.mtx').tocsr()
    print(f"Full matrix shape: {full_sparse.shape}")

    with open(data_dir / 'classic4_metadata.json') as f:
        metadata = json.load(f)

    collections = metadata['collections']
    class_names = ['cacm', 'cisi', 'cran', 'med']
    class_sizes = [collections[name] for name in class_names]

    # Build ground truth labels
    full_labels = []
    for class_id, size in enumerate(class_sizes):
        full_labels.extend([class_id] * size)
    full_labels = np.array(full_labels)
    print(f"Labels: {len(full_labels)} ({dict(zip(class_names, class_sizes))})")

    # === Version 1: Full dataset with feature selection ===
    print(f"\n{'='*70}")
    print("Creating FULL benchmark (all docs, top-5000 features)")
    print(f"{'='*70}")

    n_features_full = 5000
    dense_full = full_sparse.toarray()

    # Feature selection by variance
    feat_var = np.var(dense_full, axis=0)
    top_feat = np.argsort(feat_var)[-n_features_full:]
    top_feat.sort()
    reduced_full = dense_full[:, top_feat]

    print(f"  Shape: {reduced_full.shape}")
    print(f"  Sparsity: {np.mean(reduced_full == 0):.2%}")
    print(f"  Non-negative: {np.all(reduced_full >= 0)}")
    print(f"  Value range: [{reduced_full.min():.4f}, {reduced_full.max():.4f}]")
    for cid, name in enumerate(class_names):
        print(f"  Class {cid} ({name}): {np.sum(full_labels == cid)} docs")

    np.save(data_dir / 'classic4_full.npy', reduced_full)
    np.save(data_dir / 'classic4_full_labels.npy', full_labels)
    print(f"  Saved: classic4_full.npy, classic4_full_labels.npy")

    # === Version 2: Paper-matched (prune short docs + feature selection) ===
    print(f"\n{'='*70}")
    print("Creating PAPER-MATCHED benchmark (~6461 docs x ~4667 features)")
    print(f"{'='*70}")

    # Remove documents with very few non-zero features (likely noise)
    nnz_per_doc = np.diff(full_sparse.indptr)  # number of non-zero entries per row
    print(f"  Non-zero features per doc: min={nnz_per_doc.min()}, "
          f"median={np.median(nnz_per_doc):.0f}, max={nnz_per_doc.max()}")

    # Find threshold that gives ~6461 docs
    target_docs = 6461
    for thresh in range(1, 50):
        mask = nnz_per_doc >= thresh
        if mask.sum() <= target_docs:
            break
    # Use previous threshold
    thresh = max(thresh - 1, 1)
    doc_mask = nnz_per_doc >= thresh
    n_kept = doc_mask.sum()
    print(f"  Threshold nnz >= {thresh}: keeps {n_kept} docs (target {target_docs})")

    # If still too many, randomly subsample to target
    kept_indices = np.where(doc_mask)[0]
    if n_kept > target_docs:
        rng = np.random.RandomState(42)
        # Subsample proportionally from each class
        final_indices = []
        for cid in range(4):
            class_mask = full_labels[kept_indices] == cid
            class_kept = kept_indices[class_mask]
            n_take = int(len(class_kept) * target_docs / n_kept)
            chosen = rng.choice(class_kept, size=min(n_take, len(class_kept)), replace=False)
            chosen.sort()
            final_indices.extend(chosen.tolist())
        kept_indices = np.array(sorted(final_indices))
    print(f"  Final doc count: {len(kept_indices)}")

    paper_labels = full_labels[kept_indices]
    paper_sparse = full_sparse[kept_indices]

    # Feature selection to ~4667
    n_features_paper = 4667
    dense_paper = paper_sparse.toarray()
    feat_var_p = np.var(dense_paper, axis=0)
    # Also remove features that are zero across all selected docs
    nonzero_feat = np.where(feat_var_p > 0)[0]
    print(f"  Non-zero features: {len(nonzero_feat)}")
    if len(nonzero_feat) > n_features_paper:
        top_feat_p = nonzero_feat[np.argsort(feat_var_p[nonzero_feat])[-n_features_paper:]]
    else:
        top_feat_p = nonzero_feat
    top_feat_p.sort()
    reduced_paper = dense_paper[:, top_feat_p]

    print(f"  Shape: {reduced_paper.shape}")
    print(f"  Sparsity: {np.mean(reduced_paper == 0):.2%}")
    print(f"  Non-negative: {np.all(reduced_paper >= 0)}")
    for cid, name in enumerate(class_names):
        print(f"  Class {cid} ({name}): {np.sum(paper_labels == cid)} docs")

    np.save(data_dir / 'classic4_paper.npy', reduced_paper)
    np.save(data_dir / 'classic4_paper_labels.npy', paper_labels)
    print(f"  Saved: classic4_paper.npy, classic4_paper_labels.npy")

    # Save metadata
    meta = {
        'full': {
            'docs': reduced_full.shape[0],
            'features': reduced_full.shape[1],
            'data_file': 'classic4_full.npy',
            'labels_file': 'classic4_full_labels.npy',
            'class_names': class_names,
            'class_sizes': {name: int(np.sum(full_labels == cid))
                           for cid, name in enumerate(class_names)},
        },
        'paper': {
            'docs': reduced_paper.shape[0],
            'features': reduced_paper.shape[1],
            'data_file': 'classic4_paper.npy',
            'labels_file': 'classic4_paper_labels.npy',
            'class_names': class_names,
            'class_sizes': {name: int(np.sum(paper_labels == cid))
                           for cid, name in enumerate(class_names)},
        },
    }
    with open(data_dir / 'classic4_full_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}")


if __name__ == '__main__':
    create_full_benchmarks()
