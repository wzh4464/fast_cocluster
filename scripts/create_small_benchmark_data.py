#!/usr/bin/env python3
"""
Create smaller benchmark datasets from Classic4 for faster testing.

Creates multiple sizes:
- tiny: 200 docs × 500 features (~800KB)
- small: 500 docs × 1000 features (~4MB)
- medium: 1000 docs × 2000 features (~16MB)
"""

import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import json

def create_benchmark_datasets():
    data_dir = Path(__file__).parent.parent / 'data'

    # Load full dataset
    print("Loading full Classic4 dataset...")
    full_data = np.load(data_dir / 'classic4_subset_1000.npy')
    print(f"Original shape: {full_data.shape}")
    print(f"Original size: {full_data.nbytes / 1024 / 1024:.1f}MB")

    # Create different sizes
    configs = [
        ('tiny', 200, 500, "Fast testing (~1 second per benchmark)"),
        ('small', 500, 1000, "Quick evaluation (~5 seconds per benchmark)"),
        ('medium', 1000, 2000, "Standard testing (~20 seconds per benchmark)"),
    ]

    for name, n_docs, n_features, description in configs:
        print(f"\n{'='*70}")
        print(f"Creating {name} dataset: {n_docs} docs × {n_features} features")
        print(f"Purpose: {description}")
        print(f"{'='*70}")

        # Take subset of documents
        subset_docs = full_data[:n_docs, :]

        # Feature selection via SVD (dimensionality reduction)
        # This also makes the data more suitable for clustering
        print(f"Reducing features from {subset_docs.shape[1]} to {n_features}...")

        # Use TruncatedSVD for dimensionality reduction
        svd = TruncatedSVD(n_components=min(n_features, n_docs - 1, subset_docs.shape[1] - 1))
        reduced_data = svd.fit_transform(subset_docs)

        # Ensure we have exactly n_features
        if reduced_data.shape[1] < n_features:
            # Pad with zeros if needed
            padding = np.zeros((reduced_data.shape[0], n_features - reduced_data.shape[1]))
            reduced_data = np.hstack([reduced_data, padding])
        elif reduced_data.shape[1] > n_features:
            # Truncate if needed
            reduced_data = reduced_data[:, :n_features]

        print(f"Final shape: {reduced_data.shape}")
        print(f"Final size: {reduced_data.nbytes / 1024 / 1024:.1f}MB")
        print(f"Variance explained: {svd.explained_variance_ratio_[:min(50, len(svd.explained_variance_ratio_))].sum():.2%}")

        # Save
        output_file = data_dir / f'classic4_benchmark_{name}.npy'
        np.save(output_file, reduced_data)
        print(f"✓ Saved to {output_file}")

    # Save metadata
    metadata = {
        'datasets': {
            name: {
                'docs': n_docs,
                'features': n_features,
                'description': desc,
                'file': f'classic4_benchmark_{name}.npy'
            }
            for name, n_docs, n_features, desc in configs
        },
        'recommendation': 'Use tiny for development, small for CI, medium for thorough testing'
    }

    metadata_file = data_dir / 'benchmark_datasets.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Saved metadata to {metadata_file}")

    print(f"\n{'='*70}")
    print("All benchmark datasets created!")
    print(f"{'='*70}")
    print("\nUsage in benchmark:")
    print('  let data_path = "data/classic4_benchmark_tiny.npy";  // Fast')
    print('  let data_path = "data/classic4_benchmark_small.npy"; // Default')
    print('  let data_path = "data/classic4_benchmark_medium.npy"; // Thorough')
    print()

if __name__ == '__main__':
    create_benchmark_datasets()
