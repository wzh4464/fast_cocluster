//! # DiMergeCo Data Structures
//!
//! Core data structures for the DiMergeCo divide-merge co-clustering algorithm.

/**
 * File: /src/dimerge_co/types.rs
 * Created Date: Monday, January 27th 2026
 * Author: Zihan
 * -----
 * Last Modified: Monday, 27th January 2026
 * Modified By: Zihan Wu <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
 * 2026-01-27		Zihan	Created data structures for DiMergeCo algorithm
 */

use crate::submatrix::Submatrix;
use ndarray::Array2;
use std::error::Error;
use std::fmt;

/// Configuration parameters for probabilistic partitioning
///
/// # Mathematical Basis
/// - Threshold: τ = √(k/n)
/// - Preservation: P(preserve co-clusters) ≥ 1-δ when spectral gap σ_k - σ_{k+1} > τ
#[derive(Debug, Clone)]
pub struct PartitionParams {
    /// Number of expected co-clusters
    pub k: usize,
    /// Total number of samples
    pub n: usize,
    /// Partitioning threshold τ = √(k/n)
    pub tau: f64,
    /// Preservation probability parameter (e.g., 0.05 for 95% preservation)
    pub delta: f64,
    /// Number of partitions to create (should be power of 2 for binary tree)
    pub num_partitions: usize,
    /// Minimum partition size (rows, cols) to avoid degenerate cases
    pub min_partition_size: (usize, usize),
}

impl PartitionParams {
    /// Create new partition parameters with automatic threshold calculation
    pub fn new(k: usize, n: usize, delta: f64, num_partitions: usize) -> Self {
        let tau = (k as f64 / n as f64).sqrt();
        Self {
            k,
            n,
            tau,
            delta,
            num_partitions,
            min_partition_size: (10, 10),
        }
    }

    /// Validate that num_partitions is a power of 2
    pub fn validate(&self) -> Result<(), PartitionError> {
        if self.num_partitions == 0 || (self.num_partitions & (self.num_partitions - 1)) != 0 {
            return Err(PartitionError::InvalidPartitionCount(self.num_partitions));
        }
        if self.k == 0 || self.n == 0 {
            return Err(PartitionError::InvalidParameters);
        }
        Ok(())
    }
}

/// Represents a single partition of the data matrix
#[derive(Debug, Clone)]
pub struct Partition {
    /// Row indices in the original matrix
    pub row_indices: Vec<usize>,
    /// Column indices in the original matrix
    pub col_indices: Vec<usize>,
    /// Partition ID for tracking
    pub id: usize,
}

impl Partition {
    pub fn new(row_indices: Vec<usize>, col_indices: Vec<usize>, id: usize) -> Self {
        Self {
            row_indices,
            col_indices,
            id,
        }
    }

    pub fn size(&self) -> (usize, usize) {
        (self.row_indices.len(), self.col_indices.len())
    }
}

/// Result of the probabilistic partitioning phase
#[derive(Debug)]
pub struct PartitionResult {
    /// The created partitions
    pub partitions: Vec<Partition>,
    /// Threshold used for partitioning
    pub threshold: f64,
    /// Estimated preservation probability based on spectral gap
    pub preservation_prob: f64,
    /// Singular values from SVD (for validation)
    pub singular_values: Vec<f64>,
}

/// Node in the hierarchical merge tree
#[derive(Debug)]
pub struct MergeNode<'a> {
    /// Left child (None for leaf nodes)
    pub left: Option<Box<MergeNode<'a>>>,
    /// Right child (None for leaf nodes)
    pub right: Option<Box<MergeNode<'a>>>,
    /// Co-clustering result at this node
    pub cocluster_result: Vec<Submatrix<'a, f64>>,
    /// Metadata about this node
    pub metadata: NodeMetadata,
}

impl<'a> MergeNode<'a> {
    /// Create a leaf node
    pub fn leaf(result: Vec<Submatrix<'a, f64>>, partition_id: usize) -> Self {
        Self {
            left: None,
            right: None,
            cocluster_result: result,
            metadata: NodeMetadata {
                depth: 0,
                num_clusters: 0,
                partition_id: Some(partition_id),
                merge_score: None,
            },
        }
    }

    /// Create an internal node by merging two children
    pub fn internal(
        left: MergeNode<'a>,
        right: MergeNode<'a>,
        merged_result: Vec<Submatrix<'a, f64>>,
        merge_score: Option<f64>,
    ) -> Self {
        let depth = left.metadata.depth.max(right.metadata.depth) + 1;
        Self {
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            cocluster_result: merged_result,
            metadata: NodeMetadata {
                depth,
                num_clusters: 0,
                partition_id: None,
                merge_score,
            },
        }
    }

    /// Compute the depth of the tree
    pub fn depth(&self) -> usize {
        self.metadata.depth
    }
}

/// Metadata for a merge tree node
#[derive(Debug, Clone)]
pub struct NodeMetadata {
    /// Depth in the tree (0 for leaves)
    pub depth: usize,
    /// Number of clusters at this node
    pub num_clusters: usize,
    /// Partition ID (only for leaf nodes)
    pub partition_id: Option<usize>,
    /// Score of the merge operation (for internal nodes)
    pub merge_score: Option<f64>,
}

/// Configuration for hierarchical merging
#[derive(Debug, Clone)]
pub struct HierarchicalMergeConfig {
    /// Strategy to use when merging clusters
    pub merge_strategy: MergeStrategy,
    /// Threshold for determining which clusters to merge (strategy-dependent)
    pub merge_threshold: f64,
    /// Whether to re-score merged clusters
    pub rescore_merged: bool,
    /// Level of parallelism for merging (number of concurrent merges)
    pub parallel_level: usize,
}

impl Default for HierarchicalMergeConfig {
    fn default() -> Self {
        Self {
            merge_strategy: MergeStrategy::Adaptive,
            merge_threshold: 0.5,
            rescore_merged: true,
            parallel_level: 4,
        }
    }
}

/// Strategy for merging clusters from two nodes
#[derive(Debug, Clone)]
pub enum MergeStrategy {
    /// Take union of all clusters, remove duplicates
    Union,
    /// Keep only clusters that overlap significantly
    Intersection {
        /// Minimum overlap ratio to keep a cluster
        overlap_threshold: f64,
    },
    /// Combine clusters with weighted scoring
    Weighted {
        /// Weight for left child clusters
        left_weight: f64,
        /// Weight for right child clusters
        right_weight: f64,
    },
    /// Adaptively choose strategy based on cluster properties
    Adaptive,
}

/// Configuration for parallel execution
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Enable parallel execution
    pub enabled: bool,
    /// Minimum number of items to use parallelization (avoid overhead)
    pub min_items_for_parallel: usize,
    /// Number of threads (None = use all available cores)
    pub num_threads: Option<usize>,
    /// Chunk size for parallel iterators
    pub chunk_size: Option<usize>,
    /// Enable parallel k-means
    pub kmeans_parallel: bool,
    /// Enable parallel scoring
    pub scoring_parallel: bool,
    /// Enable parallel normalization
    pub normalization_parallel: bool,
    /// Enable parallel submatrix creation
    pub submatrix_creation_parallel: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_items_for_parallel: 100,
            num_threads: Some(num_cpus::get()),
            chunk_size: Some(num_cpus::get() * 4),
            kmeans_parallel: true,
            scoring_parallel: true,
            normalization_parallel: true,
            submatrix_creation_parallel: true,
        }
    }
}

impl ParallelConfig {
    /// Check if parallelization should be used for a given workload size
    pub fn should_parallelize(&self, num_items: usize) -> bool {
        self.enabled && num_items >= self.min_items_for_parallel
    }
}

/// Statistics from a DiMergeCo run
#[derive(Debug, Clone)]
pub struct DiMergeCoStats {
    /// Measured preservation probability
    pub preservation_prob: f64,
    /// Depth of the merge tree
    pub tree_depth: usize,
    /// Number of partitions created
    pub num_partitions: usize,
    /// Total number of local co-clusters found
    pub total_local_clusters: usize,
    /// Final number of merged clusters
    pub final_clusters: usize,
    /// Time spent in each phase (ms)
    pub phase_times: PhaseTimings,
}

/// Timing information for each DiMergeCo phase
#[derive(Debug, Clone, Default)]
pub struct PhaseTimings {
    pub partitioning_ms: u64,
    pub local_clustering_ms: u64,
    pub merging_ms: u64,
    pub total_ms: u64,
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during partitioning
#[derive(Debug)]
pub enum PartitionError {
    InvalidPartitionCount(usize),
    InvalidParameters,
    SvdFailed(String),
    InsufficientData,
    Other(String),
}

impl fmt::Display for PartitionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PartitionError::InvalidPartitionCount(n) => {
                write!(f, "Invalid partition count {} (must be power of 2)", n)
            }
            PartitionError::InvalidParameters => write!(f, "Invalid partition parameters"),
            PartitionError::SvdFailed(msg) => write!(f, "SVD computation failed: {}", msg),
            PartitionError::InsufficientData => write!(f, "Insufficient data for partitioning"),
            PartitionError::Other(msg) => write!(f, "Partition error: {}", msg),
        }
    }
}

impl Error for PartitionError {}

/// Errors that can occur during merging
#[derive(Debug)]
pub enum MergeError {
    EmptyInput,
    InconsistentDimensions,
    MergeStrategyFailed(String),
    Other(String),
}

impl fmt::Display for MergeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MergeError::EmptyInput => write!(f, "Cannot merge empty cluster sets"),
            MergeError::InconsistentDimensions => {
                write!(f, "Cluster dimensions are inconsistent")
            }
            MergeError::MergeStrategyFailed(msg) => write!(f, "Merge strategy failed: {}", msg),
            MergeError::Other(msg) => write!(f, "Merge error: {}", msg),
        }
    }
}

impl Error for MergeError {}

/// Errors from the overall DiMergeCo algorithm
#[derive(Debug)]
pub enum DiMergeCoError {
    Partition(PartitionError),
    Merge(MergeError),
    LocalClustering(String),
    InvalidConfiguration(String),
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_creation() {
        let partition = Partition {
            row_indices: vec![0, 1, 2],
            col_indices: vec![3, 4, 5],
            id: 0,
        };

        assert_eq!(partition.row_indices.len(), 3);
        assert_eq!(partition.col_indices.len(), 3);
        assert_eq!(partition.id, 0);
    }

    #[test]
    fn test_partition_size() {
        let partition = Partition {
            row_indices: vec![0, 2, 4, 6],
            col_indices: vec![1, 3],
            id: 1,
        };

        let (rows, cols) = partition.size();
        assert_eq!(rows, 4);
        assert_eq!(cols, 2);
    }

    #[test]
    fn test_partition_params_creation() {
        let params = PartitionParams {
            k: 5,
            n: 100,
            tau: 0.2236,  // sqrt(5/100)
            delta: 0.05,
            num_partitions: 8,
            min_partition_size: (10, 10),
        };

        assert_eq!(params.k, 5);
        assert_eq!(params.n, 100);
        assert!((params.tau - 0.2236).abs() < 1e-4);
        assert_eq!(params.num_partitions, 8);
    }

    #[test]
    fn test_partition_result_creation() {
        let partition1 = Partition {
            row_indices: vec![0, 1],
            col_indices: vec![0, 1],
            id: 0,
        };
        let partition2 = Partition {
            row_indices: vec![2, 3],
            col_indices: vec![2, 3],
            id: 1,
        };

        let result = PartitionResult {
            partitions: vec![partition1, partition2],
            threshold: 0.3,
            preservation_prob: 0.95,
            singular_values: vec![5.0, 3.0, 1.0],
        };

        assert_eq!(result.partitions.len(), 2);
        assert_eq!(result.threshold, 0.3);
        assert_eq!(result.preservation_prob, 0.95);
        assert_eq!(result.singular_values.len(), 3);
    }

    #[test]
    fn test_merge_strategy_union() {
        let strategy = MergeStrategy::Union;
        match strategy {
            MergeStrategy::Union => assert!(true),
            _ => panic!("Expected Union strategy"),
        }
    }

    #[test]
    fn test_merge_strategy_intersection() {
        let strategy = MergeStrategy::Intersection {
            overlap_threshold: 0.5,
        };
        match strategy {
            MergeStrategy::Intersection { overlap_threshold } => {
                assert_eq!(overlap_threshold, 0.5);
            }
            _ => panic!("Expected Intersection strategy"),
        }
    }

    #[test]
    fn test_merge_strategy_weighted() {
        let strategy = MergeStrategy::Weighted {
            left_weight: 0.6,
            right_weight: 0.4,
        };
        match strategy {
            MergeStrategy::Weighted {
                left_weight,
                right_weight,
            } => {
                assert_eq!(left_weight, 0.6);
                assert_eq!(right_weight, 0.4);
            }
            _ => panic!("Expected Weighted strategy"),
        }
    }

    #[test]
    fn test_hierarchical_merge_config_default() {
        let config = HierarchicalMergeConfig::default();
        
        match config.merge_strategy {
            MergeStrategy::Adaptive => assert!(true),
            _ => panic!("Expected Adaptive as default strategy"),
        }
        assert_eq!(config.merge_threshold, 0.5);
        assert_eq!(config.rescore_merged, true);
        assert_eq!(config.parallel_level, 2);
    }

    #[test]
    fn test_node_metadata_creation() {
        let metadata = NodeMetadata {
            depth: 2,
            num_clusters: 10,
            partition_id: Some(5),
            merge_score: Some(0.85),
        };

        assert_eq!(metadata.depth, 2);
        assert_eq!(metadata.num_clusters, 10);
        assert_eq!(metadata.partition_id.unwrap(), 5);
        assert_eq!(metadata.merge_score.unwrap(), 0.85);
    }

    #[test]
    fn test_parallel_config_default() {
        let config = ParallelConfig::default();
        
        assert_eq!(config.enabled, true);
        assert_eq!(config.min_items_for_parallel, 100);
        assert!(config.num_threads.is_some());
        assert_eq!(config.kmeans_parallel, true);
        assert_eq!(config.scoring_parallel, true);
    }

    #[test]
    fn test_parallel_config_custom() {
        let config = ParallelConfig {
            enabled: true,
            min_items_for_parallel: 50,
            num_threads: Some(8),
            chunk_size: Some(32),
            kmeans_parallel: false,
            scoring_parallel: true,
            normalization_parallel: true,
            submatrix_creation_parallel: true,
        };

        assert_eq!(config.num_threads.unwrap(), 8);
        assert_eq!(config.min_items_for_parallel, 50);
        assert_eq!(config.kmeans_parallel, false);
    }

    #[test]
    fn test_dimerge_co_stats_creation() {
        let phase_times = PhaseTimings {
            partitioning_ms: 100,
            local_clustering_ms: 500,
            merging_ms: 200,
            total_ms: 800,
        };

        let stats = DiMergeCoStats {
            preservation_prob: 0.95,
            tree_depth: 3,
            num_partitions: 8,
            total_local_clusters: 40,
            final_clusters: 10,
            phase_times,
        };

        assert_eq!(stats.preservation_prob, 0.95);
        assert_eq!(stats.tree_depth, 3);
        assert_eq!(stats.num_partitions, 8);
        assert_eq!(stats.final_clusters, 10);
        assert_eq!(stats.phase_times.total_ms, 800);
    }

    #[test]
    fn test_partition_error_display() {
        let error = PartitionError::SvdFailed("Test SVD error".to_string());
        let error_str = format!("{}", error);
        assert!(error_str.contains("SVD"));
        assert!(error_str.contains("Test SVD error"));
    }

    #[test]
    fn test_merge_error_display() {
        let error = MergeError::MergeStrategyFailed("Size mismatch".to_string());
        let error_str = format!("{}", error);
        assert!(error_str.contains("merge"));
        assert!(error_str.contains("Size mismatch"));
    }

    #[test]
    fn test_dimerge_co_error_from_partition_error() {
        let partition_err = PartitionError::InvalidPartitionCount(0);
        let dimerge_err: DiMergeCoError = partition_err.into();
        
        match dimerge_err {
            DiMergeCoError::Partition(_) => assert!(true),
            _ => panic!("Expected Partition error variant"),
        }
    }

    #[test]
    fn test_dimerge_co_error_from_merge_error() {
        let merge_err = MergeError::EmptyInput;
        let dimerge_err: DiMergeCoError = merge_err.into();
        
        match dimerge_err {
            DiMergeCoError::Merge(_) => assert!(true),
            _ => panic!("Expected Merge error variant"),
        }
    }

    #[test]
    fn test_dimerge_co_error_display() {
        let error = DiMergeCoError::InvalidConfiguration("Test config error".to_string());
        let error_str = format!("{}", error);
        assert!(error_str.contains("Invalid configuration"));
        assert!(error_str.contains("Test config error"));
    }
}

impl fmt::Display for DiMergeCoError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DiMergeCoError::Partition(e) => write!(f, "Partitioning error: {}", e),
            DiMergeCoError::Merge(e) => write!(f, "Merging error: {}", e),
            DiMergeCoError::LocalClustering(msg) => write!(f, "Local clustering error: {}", msg),
            DiMergeCoError::InvalidConfiguration(msg) => write!(f, "Invalid configuration: {}", msg),
        }
    }
}

impl Error for DiMergeCoError {}

impl From<PartitionError> for DiMergeCoError {
    fn from(e: PartitionError) -> Self {
        DiMergeCoError::Partition(e)
    }
}

impl From<MergeError> for DiMergeCoError {
    fn from(e: MergeError) -> Self {
        DiMergeCoError::Merge(e)
    }
}
