//! # DiMergeCo: Divide-Merge Co-clustering
//!
//! Implementation of the DiMergeCo algorithm for scalable co-clustering
//! with theoretical guarantees.
//!
//! ## Algorithm Overview
//!
//! DiMergeCo consists of three phases:
//! 1. **Probabilistic Partitioning**: Divide with preservation probability ≥ 1-δ
//! 2. **Local Co-clustering**: Apply SVD-based clustering independently (parallel)
//! 3. **Hierarchical Merging**: Aggregate with binary tree, O(log n) complexity
//!
//! ## Theoretical Guarantees
//! - Preservation: Co-clusters preserved with prob ≥ 1-δ
//! - Complexity: Communication O(log n) vs O(n)
//! - Convergence: Bounded error reduction
//!
//! ## References
//! Wu, Z., et al. (2024). "DiMergeCo: Divide-Merge Co-clustering for Large-Scale Data."
//! IEEE International Conference on Systems, Man, and Cybernetics (SMC).

/**
 * File: /src/dimerge_co/mod.rs
 * Created Date: Monday, January 27th 2026
 * Author: Zihan
 * -----
 * Last Modified: Monday, 27th January 2026
 * Modified By: Zihan Wu <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
 * 2026-01-27		Zihan	Created DiMergeCo module for divide-merge co-clustering
 */

pub mod types;
pub mod probabilistic_partition;
pub mod hierarchical_merge;
pub mod parallel_coclusterer;
pub mod theoretical_validation;
pub mod pipeline_integration;

pub use types::*;
pub use probabilistic_partition::ProbabilisticPartitioner;
pub use hierarchical_merge::HierarchicalMerger;
pub use parallel_coclusterer::{DiMergeCoClusterer, DiMergeCoResult, LocalClusterer};
pub use theoretical_validation::TheoreticalValidator;
pub use pipeline_integration::{ClustererAdapter, cluster_partitions_parallel};
