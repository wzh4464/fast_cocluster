#!/usr/bin/env python3
"""
Create smaller benchmark datasets from Classic4 for faster testing.

IMPORTANT: Samples evenly from all 4 classes (CACM, CISI, CRAN, MED)
and saves ground truth labels alongside the data.

Uses feature selection (top-N by variance) instead of SVD to preserve
the non-negative TF-IDF structure needed by spectral co-clustering.

Creates multiple sizes:
- tiny: 200 docs x 500 features
- small: 500 docs x 1000 features
- medium: 1000 docs x 2000 features
"""

import numpy as np
from pathlib import Path
from scipy.io import mmread
import json


def create_benchmark_datasets():
    data_dir = Path(__file__).parent.parent / 'data'

    # Load full sparse matrix (TF-IDF)
    print("Loading full Classic4 sparse matrix...")
    full_sparse = mmread(data_dir / 'classic4.mtx').tocsr()
    print(f"Full matrix shape: {full_sparse.shape}")

    # Load metadata for class boundaries
    with open(data_dir / 'classic4_metadata.json') as f:
        metadata = json.load(f)

    collections = metadata['collections']
    print(f"Collections: {collections}")

    # Build class labels from document ordering
    # Documents are loaded in order: cacm, cisi, cran, med
    class_names = ['cacm', 'cisi', 'cran', 'med']
    class_sizes = [collections[name] for name in class_names]
    full_labels = []
    for class_id, size in enumerate(class_sizes):
        full_labels.extend([class_id] * size)
    full_labels = np.array(full_labels)
    print(f"Total docs with labels: {len(full_labels)}")

    # Create different sizes
    configs = [
        ('tiny', 200, 500, "Fast testing"),
        ('small', 500, 1000, "Quick evaluation"),
        ('medium', 1000, 2000, "Standard testing"),
    ]

    for name, n_docs, n_features, description in configs:
        print(f"\n{'=' * 70}")
        print(f"Creating {name} dataset: {n_docs} docs x {n_features} features")
        print(f"Purpose: {description}")
        print(f"{'=' * 70}")

        n_classes = 4
        docs_per_class = n_docs // n_classes

        # Sample evenly from each class
        selected_indices = []
        selected_labels = []
        rng = np.random.RandomState(42)

        for class_id in range(n_classes):
            class_indices = np.where(full_labels == class_id)[0]
            chosen = rng.choice(
                class_indices,
                size=min(docs_per_class, len(class_indices)),
                replace=False,
            )
            chosen.sort()
            selected_indices.extend(chosen.tolist())
            selected_labels.extend([class_id] * len(chosen))

        selected_indices = np.array(selected_indices)
        selected_labels = np.array(selected_labels)

        print(f"  Selected {len(selected_indices)} docs ({docs_per_class} per class)")
        for class_id, cname in enumerate(class_names):
            count = np.sum(selected_labels == class_id)
            print(f"    Class {class_id} ({cname}): {count} docs")

        # Extract subset (still sparse)
        subset_sparse = full_sparse[selected_indices]

        # Feature selection: top N features by variance across the subset
        # This preserves the non-negative TF-IDF structure
        dense_subset = subset_sparse.toarray()
        feature_var = np.var(dense_subset, axis=0)
        top_features = np.argsort(feature_var)[-n_features:]
        top_features.sort()

        reduced = dense_subset[:, top_features]

        sparsity = np.sum(reduced == 0) / reduced.size
        print(f"  Final shape: {reduced.shape}")
        print(f"  Sparsity: {sparsity:.2%}")
        print(f"  Non-negative: {np.all(reduced >= 0)}")
        print(f"  Value range: [{reduced.min():.4f}, {reduced.max():.4f}]")

        # Save data
        data_file = data_dir / f'classic4_benchmark_{name}.npy'
        np.save(data_file, reduced)
        print(f"  Saved data to {data_file}")

        # Save labels
        labels_file = data_dir / f'classic4_benchmark_{name}_labels.npy'
        np.save(labels_file, selected_labels)
        print(f"  Saved labels to {labels_file}")

    # Save metadata
    meta = {
        'datasets': {
            name: {
                'docs': n_docs,
                'features': n_features,
                'description': desc,
                'data_file': f'classic4_benchmark_{name}.npy',
                'labels_file': f'classic4_benchmark_{name}_labels.npy',
                'docs_per_class': n_docs // 4,
                'class_names': class_names,
            }
            for name, n_docs, n_features, desc in configs
        },
    }

    meta_file = data_dir / 'benchmark_datasets.json'
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved metadata to {meta_file}")

    print(f"\n{'=' * 70}")
    print("All benchmark datasets created with proper multi-class sampling!")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    create_benchmark_datasets()
