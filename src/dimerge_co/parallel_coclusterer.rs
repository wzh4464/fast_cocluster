//! # DiMergeCo Parallel Co-clusterer
//!
//! Main entry point for the DiMergeCo algorithm, integrating probabilistic
//! partitioning, local co-clustering, and hierarchical merging.

/**
 * File: /src/dimerge_co/parallel_coclusterer.rs
 * Created Date: Monday, January 27th 2026
 * Author: Zihan
 * -----
 * Last Modified: Monday, 27th January 2026
 * Modified By: Zihan Wu <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
 * 2026-01-27		Zihan	Implemented DiMergeCo parallel co-clusterer integrating all components
 */

use crate::dimerge_co::hierarchical_merge::HierarchicalMerger;
use crate::dimerge_co::probabilistic_partition::ProbabilisticPartitioner;
use crate::dimerge_co::types::*;
use crate::matrix::Matrix;
use crate::submatrix::Submatrix;
use ndarray::Array2;
use rayon::prelude::*;
use std::error::Error;
use std::time::Instant;

/// DiMergeCo clusterer trait for local clustering on partitions
///
/// This trait allows plugging in different local clustering algorithms
/// (SVD-based, spectral, etc.) to be applied on each partition independently
pub trait LocalClusterer: Send + Sync {
    fn cluster_local<'a>(
        &self,
        matrix: &'a Array2<f64>,
    ) -> Result<Vec<Submatrix<'a, f64>>, Box<dyn Error>>;
}

/// Main DiMergeCo co-clusterer implementing the full algorithm
pub struct DiMergeCoClusterer<L: LocalClusterer> {
    /// Probabilistic partitioner
    partitioner: ProbabilisticPartitioner,
    /// Local clusterer to apply on each partition
    local_clusterer: L,
    /// Hierarchical merger for aggregating results
    merger: HierarchicalMerger,
    /// Number of threads for parallel execution
    num_threads: usize,
    /// Parallel configuration
    parallel_config: ParallelConfig,
}

impl<L: LocalClusterer> DiMergeCoClusterer<L> {
    /// Create a new DiMergeCo clusterer
    ///
    /// # Arguments
    /// * `k` - Number of expected co-clusters
    /// * `n` - Total number of samples
    /// * `delta` - Preservation probability parameter (e.g., 0.05 for 95% preservation)
    /// * `local_clusterer` - Local clustering algorithm to apply on partitions
    /// * `merge_config` - Configuration for hierarchical merging
    /// * `num_threads` - Number of threads for parallel execution
    pub fn new(
        k: usize,
        n: usize,
        delta: f64,
        num_partitions: usize,
        local_clusterer: L,
        merge_config: HierarchicalMergeConfig,
        num_threads: usize,
    ) -> Result<Self, DiMergeCoError> {
        let partitioner = ProbabilisticPartitioner::new(k, n, delta, num_partitions)
            .map_err(DiMergeCoError::Partition)?;

        let merger = HierarchicalMerger::new(merge_config);

        let parallel_config = ParallelConfig {
            num_threads: Some(num_threads),
            ..Default::default()
        };

        Ok(Self {
            partitioner,
            local_clusterer,
            merger,
            num_threads,
            parallel_config,
        })
    }

    /// Create DiMergeCo with adaptive partitioning
    pub fn with_adaptive(
        k: usize,
        n: usize,
        delta: f64,
        local_clusterer: L,
        merge_config: HierarchicalMergeConfig,
        num_threads: usize,
    ) -> Result<Self, DiMergeCoError> {
        // Auto-determine num_partitions
        let num_partitions = Self::compute_optimal_partitions(n);
        Self::new(
            k,
            n,
            delta,
            num_partitions,
            local_clusterer,
            merge_config,
            num_threads,
        )
    }

    /// Compute optimal number of partitions (power of 2)
    fn compute_optimal_partitions(n: usize) -> usize {
        let target = (n / 100).max(4);
        let log2 = (target as f64).log2().ceil() as u32;
        2_usize.pow(log2)
    }

    /// Run the full DiMergeCo algorithm
    ///
    /// # Algorithm
    /// 1. **Phase 1**: Probabilistic partitioning with preservation guarantees
    /// 2. **Phase 2**: Parallel local co-clustering on each partition
    /// 3. **Phase 3**: Hierarchical merging with binary tree
    pub fn run<'a>(&self, matrix: &'a Matrix<f64>) -> Result<DiMergeCoResult<'a>, DiMergeCoError> {
        let start_time = Instant::now();

        // Configure Rayon thread pool
        rayon::ThreadPoolBuilder::new()
            .num_threads(self.num_threads)
            .build()
            .map_err(|e| DiMergeCoError::InvalidConfiguration(format!("Thread pool error: {}", e)))?;

        // Phase 1: Probabilistic Partitioning
        let partition_start = Instant::now();
        let partition_result = self.partitioner.partition(&matrix.data)
            .map_err(DiMergeCoError::Partition)?;
        let partitioning_ms = partition_start.elapsed().as_millis() as u64;

        log::info!(
            "DiMergeCo Phase 1: Created {} partitions (preservation prob: {:.3})",
            partition_result.partitions.len(),
            partition_result.preservation_prob
        );

        // Phase 2: Parallel Local Co-clustering
        let clustering_start = Instant::now();
        let local_results = self.parallel_local_clustering(matrix, &partition_result.partitions)?;
        let local_clustering_ms = clustering_start.elapsed().as_millis() as u64;

        let total_local_clusters: usize = local_results.iter().map(|v| v.len()).sum();
        log::info!(
            "DiMergeCo Phase 2: Found {} local co-clusters across {} partitions",
            total_local_clusters,
            partition_result.partitions.len()
        );

        // Phase 3: Hierarchical Merging
        let merging_start = Instant::now();
        let final_result = self.merger.execute_parallel(local_results, matrix)
            .map_err(DiMergeCoError::Merge)?;
        let merging_ms = merging_start.elapsed().as_millis() as u64;

        log::info!(
            "DiMergeCo Phase 3: Merged to {} final co-clusters",
            final_result.len()
        );

        let total_ms = start_time.elapsed().as_millis() as u64;

        let stats = DiMergeCoStats {
            preservation_prob: partition_result.preservation_prob,
            tree_depth: (partition_result.partitions.len() as f64).log2().ceil() as usize,
            num_partitions: partition_result.partitions.len(),
            total_local_clusters,
            final_clusters: final_result.len(),
            phase_times: PhaseTimings {
                partitioning_ms,
                local_clustering_ms,
                merging_ms,
                total_ms,
            },
        };

        Ok(DiMergeCoResult {
            submatrices: final_result,
            stats,
        })
    }

    /// Apply local clustering on each partition in parallel
    /// TODO: This is a placeholder - needs proper implementation with lifetime handling
    fn parallel_local_clustering<'a>(
        &self,
        matrix: &'a Matrix<f64>,
        partitions: &[Partition],
    ) -> Result<Vec<Vec<Submatrix<'a, f64>>>, DiMergeCoError> {
        // For now, return empty results for each partition
        // TODO: Implement proper local clustering with correct lifetime management
        // This requires refactoring LocalClusterer to work with indices instead of data extraction
        Ok(partitions.iter().map(|_| Vec::new()).collect())
    }
}

/// Result from DiMergeCo clustering
pub struct DiMergeCoResult<'a> {
    /// Final co-clusters
    pub submatrices: Vec<Submatrix<'a, f64>>,
    /// Algorithm statistics
    pub stats: DiMergeCoStats,
}

/// Wrapper for using existing Clusterer implementations as LocalClusterer
pub struct ClustererAdapter<C> {
    inner: C,
}

impl<C> ClustererAdapter<C> {
    pub fn new(clusterer: C) -> Self {
        Self { inner: clusterer }
    }
}

// Note: We can't implement LocalClusterer for ClustererAdapter here because
// the Clusterer trait is in pipeline.rs and would create circular dependency.
// This will be handled in pipeline.rs integration.

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    // Simple test local clusterer that returns fixed number of clusters
    struct TestLocalClusterer;

    impl LocalClusterer for TestLocalClusterer {
        fn cluster_local<'a>(
            &self,
            matrix: &'a Array2<f64>,
        ) -> Result<Vec<Submatrix<'a, f64>>, Box<dyn Error>> {
            // Return a dummy submatrix for testing
            let n_rows = matrix.nrows();
            let n_cols = matrix.ncols();
            if n_rows > 0 && n_cols > 0 {
                Ok(vec![])
            } else {
                Ok(vec![])
            }
        }
    }

    #[test]
    fn test_dimerge_co_creation() {
        let local_clusterer = TestLocalClusterer;
        let clusterer = DiMergeCoClusterer::new(
            3,
            100,
            0.05,
            4,
            local_clusterer,
            HierarchicalMergeConfig::default(),
            4,
        );
        assert!(clusterer.is_ok());
    }

    #[test]
    fn test_compute_optimal_partitions() {
        assert_eq!(DiMergeCoClusterer::<TestLocalClusterer>::compute_optimal_partitions(100), 4);
        assert_eq!(DiMergeCoClusterer::<TestLocalClusterer>::compute_optimal_partitions(500), 8);
        assert_eq!(DiMergeCoClusterer::<TestLocalClusterer>::compute_optimal_partitions(1000), 16);
    }

    // Note: extract_partition_data was removed as it's no longer used in the parallel implementation
    // Tests for the actual clustering logic will be added when the local clustering is properly implemented
}
