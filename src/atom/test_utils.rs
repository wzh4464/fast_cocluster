/// Shared test utilities for atom co-clustering methods
use ndarray::Array2;

/// Create a synthetic block-diagonal matrix with clear 2x2 co-cluster structure.
/// Block (0,0) and (1,1) have high values (5.0), off-diagonal blocks have low values (0.1).
/// Returns a 20x20 matrix: rows 0-9 → cluster 0, rows 10-19 → cluster 1,
/// cols 0-9 → cluster 0, cols 10-19 → cluster 1.
pub fn make_block_diagonal() -> Array2<f64> {
    let n = 20;
    let mut x = Array2::from_elem((n, n), 0.1);
    // Block (0,0): rows 0-9, cols 0-9
    for i in 0..10 {
        for j in 0..10 {
            x[[i, j]] = 5.0;
        }
    }
    // Block (1,1): rows 10-19, cols 10-19
    for i in 10..20 {
        for j in 10..20 {
            x[[i, j]] = 5.0;
        }
    }
    x
}

/// Check that row labels correctly separate the two blocks.
/// Rows 0-9 should have one label, rows 10-19 should have a different label.
pub fn check_block_labels(labels: &[usize], block_size: usize) -> bool {
    if labels.len() != block_size * 2 {
        return false;
    }
    let label_a = labels[0];
    let label_b = labels[block_size];

    // All rows in block 0 should have the same label
    let block0_consistent = labels[..block_size].iter().all(|&l| l == label_a);
    // All rows in block 1 should have the same label
    let block1_consistent = labels[block_size..].iter().all(|&l| l == label_b);
    // Two blocks should have different labels
    let different = label_a != label_b;

    block0_consistent && block1_consistent && different
}
