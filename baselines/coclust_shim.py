"""
Minimal shim for coclust package functions used by NMTFcoclust.
Replaces `from ..initialization import random_init` and
`from ..io.input_checking import check_positive`.

This avoids installing the full coclust package as a dependency.
"""

import numpy as np
from sklearn.utils import check_random_state
from scipy.sparse import issparse, dok_matrix, lil_matrix


def random_init(n_clusters, n_cols, random_state=None):
    """Random one-hot cluster assignment matrix (n_cols x n_clusters)."""
    random_state = check_random_state(random_state)
    assignments = random_state.randint(n_clusters, size=n_cols)
    W = np.zeros((n_cols, n_clusters))
    W[np.arange(n_cols), assignments] = 1
    return W


def check_positive(X):
    """Raise ValueError if matrix contains negative values."""
    if isinstance(X, dok_matrix):
        values = np.array(list(X.values()))
    elif isinstance(X, lil_matrix):
        values = np.array([v for e in X.data for v in e])
    elif not issparse(X):
        values = X
    else:
        values = X.data
    if (np.asarray(values) < 0).any():
        raise ValueError("The matrix contains negative values.")
    return X
