//! # Pipeline Integration
//!
//! Adapters and utilities for integrating DiMergeCo with the existing Pipeline system.

/**
 * File: /src/dimerge_co/pipeline_integration.rs
 * Created Date: Monday, January 27th 2026
 * Author: Zihan
 * -----
 * Last Modified: Monday, 27th January 2026
 * Modified By: Zihan Wu <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
 * 2026-01-27		Zihan	Created pipeline integration adapters for DiMergeCo
 */

use crate::dimerge_co::parallel_coclusterer::LocalClusterer;
use crate::dimerge_co::types::Partition;
use crate::matrix::Matrix;
use crate::pipeline::Clusterer;
use crate::submatrix::Submatrix;
use ndarray::Array2;
use std::error::Error;

/// Adapter to use an existing Clusterer as a LocalClusterer for DiMergeCo
///
/// This wrapper allows any implementation of the `Clusterer` trait (e.g., SVDClusterer)
/// to be used as a `LocalClusterer` in DiMergeCo's parallel pipeline.
pub struct ClustererAdapter<C: Clusterer> {
    inner: C,
}

impl<C: Clusterer> ClustererAdapter<C> {
    /// Create a new adapter wrapping an existing clusterer
    pub fn new(clusterer: C) -> Self {
        Self { inner: clusterer }
    }
}

impl<C: Clusterer> LocalClusterer for ClustererAdapter<C> {
    fn cluster_local<'a>(
        &self,
        matrix: &'a Array2<f64>,
    ) -> Result<Vec<Submatrix<'a, f64>>, Box<dyn Error>> {
        // Wrap Array2 in Matrix for compatibility with Clusterer trait
        let matrix_wrapper = Matrix {
            data: matrix.clone(), // TODO: This involves a copy - optimize if needed
            rows: matrix.nrows(),
            cols: matrix.ncols(),
        };

        // Call the wrapped clusterer
        let result = self.inner.cluster(&matrix_wrapper)?;

        // The result already has the correct lifetime 'a tied to matrix_wrapper
        // But matrix_wrapper is about to be dropped, so we need to re-map to the original matrix

        // Extract row/col indices from each submatrix and recreate from original matrix
        let remapped: Vec<Submatrix<'a, f64>> = result
            .into_iter()
            .filter_map(|sub| {
                // Create new submatrix from the original matrix using the same indices
                Submatrix::from_indices(
                    matrix,
                    &sub.row_indices,
                    &sub.col_indices,
                )
            })
            .collect();

        Ok(remapped)
    }
}

/// Helper to extract partition data as owned Matrix
///
/// This creates a new Matrix containing only the rows/cols specified by the partition.
/// Used when we need to materialize partition data for local clustering.
pub fn extract_partition_matrix(
    matrix: &Array2<f64>,
    partition: &Partition,
) -> Array2<f64> {
    let n_rows = partition.row_indices.len();
    let n_cols = partition.col_indices.len();

    let mut result = Array2::zeros((n_rows, n_cols));

    for (i, &row_idx) in partition.row_indices.iter().enumerate() {
        for (j, &col_idx) in partition.col_indices.iter().enumerate() {
            result[[i, j]] = matrix[[row_idx, col_idx]];
        }
    }

    result
}

/// Improved local clustering that works with partition indices
///
/// This function applies local clustering on each partition by:
/// 1. Extracting partition data into owned matrices
/// 2. Running local clustering on each partition independently
/// 3. Mapping results back to original matrix indices
pub fn cluster_partitions_parallel<'a, L: LocalClusterer>(
    original_matrix: &'a Array2<f64>,
    partitions: &[Partition],
    local_clusterer: &L,
) -> Result<Vec<Vec<Submatrix<'a, f64>>>, Box<dyn Error + Send + Sync>> {
    use rayon::prelude::*;

    let results: Result<Vec<Vec<Submatrix<'a, f64>>>, Box<dyn Error + Send + Sync>> = partitions
        .par_iter()
        .map(|partition| -> Result<Vec<Submatrix<'a, f64>>, Box<dyn Error + Send + Sync>> {
            // Extract partition data
            let partition_matrix = extract_partition_matrix(original_matrix, partition);

            // Cluster on partition
            let local_clusters = local_clusterer.cluster_local(&partition_matrix)
                .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())) as Box<dyn Error + Send + Sync>)?;

            // Map back to original matrix indices
            let mapped: Vec<Submatrix<'a, f64>> = local_clusters
                .into_iter()
                .filter_map(|sub| {
                    // Translate local partition indices to global matrix indices
                    let global_rows: Vec<usize> = sub
                        .row_indices
                        .iter()
                        .map(|&local_idx| partition.row_indices[local_idx])
                        .collect();

                    let global_cols: Vec<usize> = sub
                        .col_indices
                        .iter()
                        .map(|&local_idx| partition.col_indices[local_idx])
                        .collect();

                    // Create submatrix referencing original matrix
                    Submatrix::from_indices(original_matrix, &global_rows, &global_cols)
                })
                .collect();

            Ok(mapped)
        })
        .collect();

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_extract_partition_matrix() {
        let matrix = Array2::from_shape_vec(
            (4, 4),
            (0..16).map(|x| x as f64).collect(),
        )
        .unwrap();

        let partition = Partition {
            row_indices: vec![0, 2],
            col_indices: vec![1, 3],
            id: 0,
        };

        let result = extract_partition_matrix(&matrix, &partition);

        assert_eq!(result.dim(), (2, 2));
        assert_eq!(result[[0, 0]], matrix[[0, 1]]);
        assert_eq!(result[[0, 1]], matrix[[0, 3]]);
        assert_eq!(result[[1, 0]], matrix[[2, 1]]);
        assert_eq!(result[[1, 1]], matrix[[2, 3]]);
    }
}
