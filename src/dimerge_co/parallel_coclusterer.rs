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
    /// Number of T_p random partitioning iterations (paper: Theorem 3)
    num_iterations: usize,
    /// Number of row blocks per random partition
    m_blocks: usize,
    /// Number of column blocks per random partition
    n_blocks: usize,
}

impl<L: LocalClusterer> DiMergeCoClusterer<L> {
    /// Create a new DiMergeCo clusterer
    ///
    /// # Arguments
    /// * `k` - Number of expected co-clusters
    /// * `n` - Total number of samples
    /// * `delta` - Preservation probability parameter (e.g., 0.05 for 95% preservation)
    /// * `num_partitions` - Number of partitions (power of 2)
    /// * `local_clusterer` - Local clustering algorithm to apply on partitions
    /// * `merge_config` - Configuration for hierarchical merging
    /// * `num_threads` - Number of threads for parallel execution
    /// * `num_iterations` - Number of T_p random partitioning iterations (paper Theorem 3)
    pub fn new(
        k: usize,
        n: usize,
        delta: f64,
        num_partitions: usize,
        local_clusterer: L,
        merge_config: HierarchicalMergeConfig,
        num_threads: usize,
        num_iterations: usize,
    ) -> Result<Self, DiMergeCoError> {
        let partitioner = ProbabilisticPartitioner::new(k, n, delta, num_partitions)
            .map_err(DiMergeCoError::Partition)?;

        let merger = HierarchicalMerger::new(merge_config);

        let parallel_config = ParallelConfig {
            num_threads: Some(num_threads),
            ..Default::default()
        };

        // Compute block grid dimensions: m_blocks × n_blocks ≈ num_partitions
        let sqrt_p = (num_partitions as f64).sqrt().ceil() as usize;
        let m_blocks = sqrt_p.max(2);
        let n_blocks = sqrt_p.max(2);

        Ok(Self {
            partitioner,
            local_clusterer,
            merger,
            num_threads,
            parallel_config,
            num_iterations: num_iterations.max(1),
            m_blocks,
            n_blocks,
        })
    }

    /// Create DiMergeCo with adaptive partitioning (uses T_p=10 iterations by default)
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
            10, // default T_p=10 iterations
        )
    }

    /// Compute optimal number of partitions (power of 2)
    fn compute_optimal_partitions(n: usize) -> usize {
        let target = (n / 100).max(4);
        let log2 = (target as f64).log2().ceil() as u32;
        2_usize.pow(log2)
    }

    /// Run the full DiMergeCo algorithm with T_p random partitioning iterations
    ///
    /// # Algorithm (paper Theorem 3)
    /// 1. **Phase 1+2**: T_p iterations of random block partitioning + local co-clustering
    ///    - Each iteration uses a different random seed for independent partitions
    ///    - All local co-clusters are collected across iterations
    /// 2. **Phase 3**: Single hierarchical merge on all collected co-clusters
    ///
    /// Preservation guarantee: P >= 1 - K * exp(-gamma * T_p)
    pub fn run<'a>(&self, matrix: &'a Matrix<f64>) -> Result<DiMergeCoResult<'a>, DiMergeCoError> {
        let start_time = Instant::now();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.num_threads)
            .build()
            .map_err(|e| DiMergeCoError::InvalidConfiguration(format!("Thread pool error: {}", e)))?;

        log::info!(
            "DiMergeCo: {} threads, matrix {}x{}, T_p={} iterations, {}x{} blocks",
            self.num_threads, matrix.rows, matrix.cols,
            self.num_iterations, self.m_blocks, self.n_blocks
        );

        let (final_result, stats) = pool.install(|| -> Result<_, DiMergeCoError> {
            // Phase 1+2: T_p iterations of random partitioning + local clustering
            let phase12_start = Instant::now();
            let mut all_local_results: Vec<Vec<Submatrix<'a, f64>>> = Vec::new();
            let mut total_partitions = 0;

            for iter in 0..self.num_iterations {
                // Random block partitioning with different seed each iteration
                let partition_result = self.partitioner.partition_random_blocks(
                    &matrix.data,
                    self.m_blocks,
                    self.n_blocks,
                    iter as u64,
                ).map_err(DiMergeCoError::Partition)?;

                let n_parts = partition_result.partitions.len();
                total_partitions += n_parts;

                log::info!(
                    "DiMergeCo iter {}/{}: {} partitions from {}x{} blocks",
                    iter + 1, self.num_iterations, n_parts, self.m_blocks, self.n_blocks
                );

                // Local clustering on this iteration's partitions
                let local_results = self.parallel_local_clustering(matrix, &partition_result.partitions)?;
                let iter_clusters: usize = local_results.iter().map(|v| v.len()).sum();
                log::info!(
                    "DiMergeCo iter {}/{}: {} local co-clusters",
                    iter + 1, self.num_iterations, iter_clusters
                );

                all_local_results.extend(local_results);
            }

            let phase12_ms = phase12_start.elapsed().as_millis() as u64;
            let total_local_clusters: usize = all_local_results.iter().map(|v| v.len()).sum();

            log::info!(
                "DiMergeCo Phases 1+2: {} iterations, {} total partitions, {} local co-clusters [{} ms]",
                self.num_iterations, total_partitions, total_local_clusters, phase12_ms
            );

            // Phase 3: Hierarchical Merging (once, on all collected co-clusters)
            let merging_start = Instant::now();
            let final_result = self.merger.execute_parallel(all_local_results, matrix)
                .map_err(DiMergeCoError::Merge)?;
            let merging_ms = merging_start.elapsed().as_millis() as u64;

            log::info!(
                "DiMergeCo Phase 3: Merged to {} final co-clusters [{} ms]",
                final_result.len(), merging_ms
            );

            let total_ms = start_time.elapsed().as_millis() as u64;

            let stats = DiMergeCoStats {
                preservation_prob: 1.0 - (-0.5 * self.num_iterations as f64).exp(),
                tree_depth: (total_partitions as f64).log2().ceil() as usize,
                num_partitions: total_partitions,
                total_local_clusters,
                final_clusters: final_result.len(),
                phase_times: PhaseTimings {
                    partitioning_ms: phase12_ms,
                    local_clustering_ms: 0, // combined with partitioning
                    merging_ms,
                    total_ms,
                },
            };

            Ok((final_result, stats))
        })?;

        Ok(DiMergeCoResult {
            submatrices: final_result,
            stats,
        })
    }

    /// Apply local clustering on each partition in parallel
    ///
    /// Uses the pipeline_integration::cluster_partitions_parallel function to properly
    /// handle lifetimes by mapping local partition results to global matrix indices.
    fn parallel_local_clustering<'a>(
        &self,
        matrix: &'a Matrix<f64>,
        partitions: &[Partition],
    ) -> Result<Vec<Vec<Submatrix<'a, f64>>>, DiMergeCoError> {
        use crate::dimerge_co::pipeline_integration::cluster_partitions_parallel;

        cluster_partitions_parallel(&matrix.data, partitions, &self.local_clusterer)
            .map_err(|e| DiMergeCoError::InvalidConfiguration(format!("Local clustering failed: {}", e)))
    }
}

/// Result from DiMergeCo clustering
pub struct DiMergeCoResult<'a> {
    /// Final co-clusters
    pub submatrices: Vec<Submatrix<'a, f64>>,
    /// Algorithm statistics
    pub stats: DiMergeCoStats,
}

/// Implementation of Clusterer trait for DiMergeCo
impl<L: LocalClusterer> crate::pipeline::Clusterer for DiMergeCoClusterer<L> {
    fn cluster<'matrix_life>(
        &self,
        matrix: &'matrix_life Matrix<f64>,
    ) -> Result<Vec<Submatrix<'matrix_life, f64>>, Box<dyn std::error::Error>> {
        let result = self.run(matrix)?;
        Ok(result.submatrices)
    }

    fn name(&self) -> &str {
        "DiMergeCo"
    }
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
            1, // T_p=1 for basic test
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
