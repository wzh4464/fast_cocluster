#!/usr/bin/env python3
"""
Download and prepare Classic4 dataset for benchmarking.

Classic4 is a standard document clustering benchmark with 4 categories:
- CACM: Computer Science (3,204 docs)
- CISI: Information Retrieval (1,460 docs)
- CRAN: Aeronautics (1,400 docs)
- MED: Medical (1,033 docs)
Total: 7,097 documents

Dataset source: University of Glasgow IR Test Collections
https://ir.dcs.gla.ac.uk/resources/test_collections/
"""

import os
import sys
import urllib.request
import tarfile
from pathlib import Path
from scipy.sparse import lil_matrix
import numpy as np

# Dataset URLs from University of Glasgow
DATASETS = {
    'cacm': 'https://ir.dcs.gla.ac.uk/resources/test_collections/cacm/cacm.tar.gz',
    'cisi': 'https://ir.dcs.gla.ac.uk/resources/test_collections/cisi/cisi.tar.gz',
    'cran': 'https://ir.dcs.gla.ac.uk/resources/test_collections/cran/cran.tar.gz',
    'med': 'https://ir.dcs.gla.ac.uk/resources/test_collections/medl/med.tar.gz',
}

def download_file(url: str, dest: Path) -> bool:
    """Download a file with progress indication."""
    try:
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, dest)
        print(f"✓ Downloaded to {dest}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {url}: {e}")
        return False

def extract_tar_gz(tar_path: Path, extract_to: Path) -> bool:
    """Extract a tar.gz file."""
    try:
        print(f"Extracting {tar_path.name}...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(extract_to)
        print(f"✓ Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"✗ Failed to extract {tar_path}: {e}")
        return False

def parse_document(doc_text: str) -> dict:
    """Parse a document in the standard format used by these collections."""
    doc = {'id': None, 'title': '', 'authors': '', 'text': ''}

    lines = doc_text.strip().split('\n')
    current_field = None

    for line in lines:
        line = line.strip()
        if line.startswith('.I'):
            doc['id'] = line.split()[1] if len(line.split()) > 1 else None
        elif line.startswith('.T'):
            current_field = 'title'
        elif line.startswith('.A'):
            current_field = 'authors'
        elif line.startswith('.W') or line.startswith('.B'):
            current_field = 'text'
        elif line.startswith('.'):
            current_field = None
        elif current_field:
            doc[current_field] += ' ' + line

    return doc

def load_collection(collection_path: Path) -> list:
    """Load documents from a collection file."""
    documents = []

    # Find the main document file (case-insensitive search)
    all_files = [f for f in collection_path.glob('*') if f.is_file() and f.suffix.lower() == '.all']
    if not all_files:
        # Fallback: find any file that's not query or relevance
        all_files = list(collection_path.glob('*'))
        all_files = [f for f in all_files if f.is_file() and
                    not f.suffix.lower() in ('.qry', '.rel') and
                    not f.name.endswith('.OLD')]

    if not all_files:
        print(f"Warning: No document files found in {collection_path}")
        return documents

    doc_file = all_files[0]
    print(f"  Reading {doc_file.name}...")

    try:
        with open(doc_file, 'r', encoding='latin-1') as f:
            content = f.read()

        # Split by document markers
        doc_texts = content.split('.I ')
        for doc_text in doc_texts[1:]:  # Skip first empty split
            doc = parse_document('.I ' + doc_text)
            if doc['id']:
                # Combine title and text as document content
                full_text = (doc['title'] + ' ' + doc['text']).strip()
                if full_text:
                    documents.append({
                        'id': doc['id'],
                        'collection': collection_path.name,
                        'text': full_text
                    })
    except Exception as e:
        print(f"Error reading {doc_file}: {e}")

    return documents

def build_vocabulary(documents: list) -> tuple:
    """Build vocabulary and term-document matrix."""
    from collections import Counter

    # Simple tokenization
    all_terms = []
    doc_terms = []

    for doc in documents:
        # Basic preprocessing: lowercase, split, filter short terms
        terms = [t.lower() for t in doc['text'].split() if len(t) > 2 and t.isalnum()]
        doc_terms.append(terms)
        all_terms.extend(terms)

    # Count term frequencies
    term_freq = Counter(all_terms)

    # Keep terms that appear in at least 2 docs and at most 80% of docs
    min_df = 2
    max_df = int(0.8 * len(documents))

    vocab = {}
    term_doc_count = Counter()
    for terms in doc_terms:
        for term in set(terms):
            term_doc_count[term] += 1

    vocab_terms = [term for term, df in term_doc_count.items()
                   if min_df <= df <= max_df]
    vocab = {term: idx for idx, term in enumerate(sorted(vocab_terms))}

    print(f"  Vocabulary size: {len(vocab)} terms")

    # Build term-document matrix
    n_docs = len(documents)
    n_terms = len(vocab)
    matrix = lil_matrix((n_docs, n_terms), dtype=np.float64)

    for doc_idx, terms in enumerate(doc_terms):
        term_counts = Counter(terms)
        for term, count in term_counts.items():
            if term in vocab:
                # Use TF-IDF weighting
                tf = 1 + np.log(count)
                matrix[doc_idx, vocab[term]] = tf

    # Apply IDF
    doc_freq = np.array((matrix > 0).sum(axis=0)).flatten()
    idf = np.log(n_docs / (1 + doc_freq))
    matrix = matrix.multiply(idf)

    return matrix.tocsr(), vocab

def main():
    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'
    raw_dir = data_dir / 'classic4_raw'

    data_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("Classic4 Dataset Downloader")
    print("=" * 70)
    print()

    # Download and extract all collections
    all_documents = []

    for name, url in DATASETS.items():
        print(f"\n--- Processing {name.upper()} collection ---")

        tar_file = raw_dir / f"{name}.tar.gz"
        extract_dir = raw_dir / name
        extract_dir.mkdir(exist_ok=True)

        # Download if not exists
        if not tar_file.exists():
            if not download_file(url, tar_file):
                print(f"Failed to download {name}, skipping...")
                continue
        else:
            print(f"✓ Already downloaded: {tar_file}")

        # Extract if not already extracted
        if not any(extract_dir.iterdir()):
            if not extract_tar_gz(tar_file, extract_dir):
                print(f"Failed to extract {name}, skipping...")
                continue
        else:
            print(f"✓ Already extracted: {extract_dir}")

        # Load documents
        documents = load_collection(extract_dir)
        print(f"✓ Loaded {len(documents)} documents from {name}")
        all_documents.extend(documents)

    if not all_documents:
        print("\n✗ No documents loaded. Exiting.")
        return 1

    print("\n" + "=" * 70)
    print(f"Total documents: {len(all_documents)}")
    print("=" * 70)
    print()

    # Build term-document matrix
    print("Building term-document matrix...")
    matrix, vocab = build_vocabulary(all_documents)

    print(f"\nMatrix shape: {matrix.shape[0]} docs × {matrix.shape[1]} terms")
    print(f"Sparsity: {100 * (1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])):.2f}%")

    # Save sparse matrix in Matrix Market format
    from scipy.io import mmwrite
    mtx_file = data_dir / 'classic4.mtx'
    print(f"\nSaving sparse matrix to {mtx_file}...")
    mmwrite(mtx_file, matrix)
    print("✓ Saved")

    # Create a dense subset for benchmarking (first 1000 docs)
    subset_size = min(1000, matrix.shape[0])
    dense_subset = matrix[:subset_size].toarray()

    subset_file = data_dir / 'classic4_subset_1000.npy'
    print(f"\nSaving dense subset ({subset_size} docs) to {subset_file}...")
    np.save(subset_file, dense_subset)
    print("✓ Saved")

    # Save document metadata
    import json
    metadata = {
        'total_docs': len(all_documents),
        'vocab_size': len(vocab),
        'collections': {name: sum(1 for d in all_documents if d['collection'] == name)
                       for name in DATASETS.keys()},
        'matrix_shape': list(matrix.shape),
        'sparsity': float(1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
    }

    metadata_file = data_dir / 'classic4_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Saved metadata to {metadata_file}")

    print("\n" + "=" * 70)
    print("Dataset preparation complete!")
    print("=" * 70)
    print(f"\nFiles created:")
    print(f"  - {mtx_file}")
    print(f"  - {subset_file}")
    print(f"  - {metadata_file}")
    print(f"\nYou can now run the benchmark with:")
    print(f"  cargo bench --bench classic4_benchmark")
    print()

    return 0

if __name__ == '__main__':
    sys.exit(main())
