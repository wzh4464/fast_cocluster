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
    use crate::pipeline::Clusterer;
    use crate::submatrix::Submatrix;
    use ndarray::Array2;

    // Mock local clusterer for testing
    struct MockLocalClusterer {
        num_clusters: usize,
    }

    impl LocalClusterer for MockLocalClusterer {
        fn cluster_local<'a>(
            &self,
            matrix: &'a Array2<f64>,
        ) -> Result<Vec<Submatrix<'a, f64>>, Box<dyn std::error::Error>> {
            let n_rows = matrix.nrows();
            let n_cols = matrix.ncols();

            if n_rows == 0 || n_cols == 0 {
                return Ok(vec![]);
            }

            // Create simple clusters by splitting matrix
            let mut clusters = Vec::new();
            let rows_per_cluster = (n_rows / self.num_clusters).max(1);

            for i in 0..self.num_clusters.min(n_rows / rows_per_cluster) {
                let row_start = i * rows_per_cluster;
                let row_end = ((i + 1) * rows_per_cluster).min(n_rows);
                let rows: Vec<usize> = (row_start..row_end).collect();
                let cols: Vec<usize> = (0..n_cols).collect();

                if let Some(sub) = Submatrix::from_indices(matrix, &rows, &cols) {
                    clusters.push(sub);
                }
            }

            Ok(clusters)
        }
    }

    // Mock Clusterer for testing adapter
    struct MockClusterer;

    impl Clusterer for MockClusterer {
        fn cluster<'matrix_life>(
            &self,
            matrix: &'matrix_life Matrix<f64>,
        ) -> Result<Vec<Submatrix<'matrix_life, f64>>, Box<dyn std::error::Error>> {
            let n_rows = matrix.data.nrows();
            let n_cols = matrix.data.ncols();

            if n_rows < 2 || n_cols < 2 {
                return Ok(vec![]);
            }

            // Return a simple cluster
            let rows = vec![0, 1];
            let cols = vec![0, 1];
            if let Some(sub) = Submatrix::from_indices(&matrix.data, &rows, &cols) {
                Ok(vec![sub])
            } else {
                Ok(vec![])
            }
        }

        fn name(&self) -> &str {
            "MockClusterer"
        }
    }

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

    #[test]
    fn test_extract_partition_matrix_single_element() {
        let matrix = Array2::from_shape_vec(
            (3, 3),
            (0..9).map(|x| x as f64).collect(),
        )
        .unwrap();

        let partition = Partition {
            row_indices: vec![1],
            col_indices: vec![2],
            id: 0,
        };

        let result = extract_partition_matrix(&matrix, &partition);

        assert_eq!(result.dim(), (1, 1));
        assert_eq!(result[[0, 0]], matrix[[1, 2]]);
    }

    #[test]
    fn test_extract_partition_matrix_full_rows() {
        let matrix = Array2::from_shape_vec(
            (3, 4),
            (0..12).map(|x| x as f64).collect(),
        )
        .unwrap();

        let partition = Partition {
            row_indices: vec![0, 1, 2],
            col_indices: vec![0, 1, 2, 3],
            id: 0,
        };

        let result = extract_partition_matrix(&matrix, &partition);

        assert_eq!(result.dim(), (3, 4));
        // Should be identical to original
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(result[[i, j]], matrix[[i, j]]);
            }
        }
    }

    #[test]
    fn test_clusterer_adapter_basic() {
        let adapter = ClustererAdapter::new(MockClusterer);
        let matrix = Array2::from_shape_vec(
            (4, 4),
            (0..16).map(|x| x as f64).collect(),
        )
        .unwrap();

        let result = adapter.cluster_local(&matrix).unwrap();

        // MockClusterer returns 1 cluster with rows [0,1], cols [0,1]
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].row_indices, vec![0, 1]);
        assert_eq!(result[0].col_indices, vec![0, 1]);
    }

    #[test]
    fn test_clusterer_adapter_empty_matrix() {
        let adapter = ClustererAdapter::new(MockClusterer);
        let matrix = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();

        let result = adapter.cluster_local(&matrix).unwrap();

        // MockClusterer returns empty for small matrices
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_cluster_partitions_parallel_single_partition() {
        let matrix = Array2::from_shape_vec(
            (6, 4),
            (0..24).map(|x| x as f64).collect(),
        )
        .unwrap();

        let partition = Partition {
            row_indices: vec![0, 1, 2],
            col_indices: vec![0, 1],
            id: 0,
        };

        let local_clusterer = MockLocalClusterer { num_clusters: 2 };

        let results = cluster_partitions_parallel(&matrix, &[partition], &local_clusterer).unwrap();

        assert_eq!(results.len(), 1); // One partition
        assert!(results[0].len() > 0); // Should have clusters
    }

    #[test]
    fn test_cluster_partitions_parallel_multiple_partitions() {
        let matrix = Array2::from_shape_vec(
            (8, 6),
            (0..48).map(|x| x as f64).collect(),
        )
        .unwrap();

        let partitions = vec![
            Partition {
                row_indices: vec![0, 1, 2],
                col_indices: vec![0, 1, 2],
                id: 0,
            },
            Partition {
                row_indices: vec![3, 4, 5],
                col_indices: vec![3, 4, 5],
                id: 1,
            },
        ];

        let local_clusterer = MockLocalClusterer { num_clusters: 1 };

        let results = cluster_partitions_parallel(&matrix, &partitions, &local_clusterer).unwrap();

        assert_eq!(results.len(), 2); // Two partitions
        
        // Each partition should have at least one cluster
        for partition_result in &results {
            assert!(partition_result.len() > 0);
        }

        // Verify indices are correctly mapped back to original matrix
        for cluster in &results[0] {
            for &row_idx in &cluster.row_indices {
                assert!(row_idx < 3); // First partition rows
            }
        }
    }

    #[test]
    fn test_cluster_partitions_parallel_empty_partitions() {
        let matrix = Array2::from_shape_vec(
            (4, 4),
            (0..16).map(|x| x as f64).collect(),
        )
        .unwrap();

        let partitions: Vec<Partition> = vec![];
        let local_clusterer = MockLocalClusterer { num_clusters: 2 };

        let results = cluster_partitions_parallel(&matrix, &partitions, &local_clusterer).unwrap();

        assert_eq!(results.len(), 0); // No partitions, no results
    }

    #[test]
    fn test_cluster_partitions_parallel_preserves_indices() {
        let matrix = Array2::from_shape_vec(
            (6, 6),
            (0..36).map(|x| x as f64).collect(),
        )
        .unwrap();

        let partition = Partition {
            row_indices: vec![1, 3, 5], // Non-contiguous
            col_indices: vec![0, 2, 4], // Non-contiguous
            id: 0,
        };

        let local_clusterer = MockLocalClusterer { num_clusters: 1 };

        let results = cluster_partitions_parallel(&matrix, &[partition], &local_clusterer).unwrap();

        assert_eq!(results.len(), 1);
        
        // Verify all returned indices are from the partition
        for cluster in &results[0] {
            for &row_idx in &cluster.row_indices {
                assert!(vec![1, 3, 5].contains(&row_idx));
            }
            for &col_idx in &cluster.col_indices {
                assert!(vec![0, 2, 4].contains(&col_idx));
            }
        }
    }
}
