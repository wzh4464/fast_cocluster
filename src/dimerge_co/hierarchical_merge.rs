//! # Hierarchical Merging
//!
//! Implements binary tree hierarchical merging for co-cluster aggregation.
//!
//! ## Mathematical Basis
//! - Tree depth: log₂(P) for P partitions
//! - Complexity: O(log n) vs O(n) traditional sequential merging

/**
 * File: /src/dimerge_co/hierarchical_merge.rs
 * Created Date: Monday, January 27th 2026
 * Author: Zihan
 * -----
 * Last Modified: Monday, 27th January 2026
 * Modified By: Zihan Wu <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
 * 2026-01-27		Zihan	Implemented hierarchical binary tree merging with O(log n) complexity
 */

use crate::dimerge_co::types::*;
use crate::matrix::Matrix;
use crate::scoring::Scorer;
use crate::submatrix::Submatrix;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// Hierarchical merger using binary tree structure
pub struct HierarchicalMerger {
    config: HierarchicalMergeConfig,
}

impl HierarchicalMerger {
    /// Create a new hierarchical merger with configuration
    pub fn new(config: HierarchicalMergeConfig) -> Self {
        Self { config }
    }

    /// Build the merge tree from partition results
    ///
    /// # Algorithm
    /// 1. Pad results to power of 2 if needed
    /// 2. Build binary tree bottom-up with parallel merging
    /// 3. Recursively merge left and right subtrees
    pub fn build_merge_tree<'a>(
        &self,
        partition_results: Vec<Vec<Submatrix<'a, f64>>>,
    ) -> Result<MergeNode<'a>, MergeError> {
        if partition_results.is_empty() {
            return Err(MergeError::EmptyInput);
        }

        // Pad to power of 2 if needed
        let padded_results = self.pad_to_power_of_two(partition_results);

        // Build tree recursively
        self.build_tree_recursive(padded_results, 0)
    }

    /// Execute the merge and return final result
    pub fn execute_parallel<'a>(
        &self,
        partition_results: Vec<Vec<Submatrix<'a, f64>>>,
        original_matrix: &'a Matrix<f64>,
    ) -> Result<Vec<Submatrix<'a, f64>>, MergeError> {
        let tree = self.build_merge_tree(partition_results)?;
        Ok(tree.cocluster_result)
    }

    /// Pad results to next power of 2 for balanced binary tree
    fn pad_to_power_of_two<'a>(
        &self,
        mut results: Vec<Vec<Submatrix<'a, f64>>>,
    ) -> Vec<Vec<Submatrix<'a, f64>>> {
        if results.is_empty() {
            return results;
        }

        let len = results.len();
        let next_power = len.next_power_of_two();

        // Pad with empty results
        while results.len() < next_power {
            results.push(Vec::new());
        }

        results
    }

    /// Build tree recursively with parallel merging
    fn build_tree_recursive<'a>(
        &self,
        results: Vec<Vec<Submatrix<'a, f64>>>,
        level: usize,
    ) -> Result<MergeNode<'a>, MergeError> {
        if results.len() == 1 {
            // Leaf node
            return Ok(MergeNode::leaf(results.into_iter().next().unwrap(), level));
        }

        // Split in half
        let mid = results.len() / 2;
        let (left_results, right_results) = results.split_at(mid);

        // Determine if we should parallelize at this level
        let should_parallel = level < self.config.parallel_level;

        let (left_node, right_node) = if should_parallel {
            // PARALLEL: Build left and right subtrees concurrently
            rayon::join(
                || self.build_tree_recursive(left_results.to_vec(), level + 1),
                || self.build_tree_recursive(right_results.to_vec(), level + 1),
            )
        } else {
            // Sequential for deeper levels
            let left = self.build_tree_recursive(left_results.to_vec(), level + 1);
            let right = self.build_tree_recursive(right_results.to_vec(), level + 1);
            (left, right)
        };

        let left_node = left_node?;
        let right_node = right_node?;

        // Merge the two children
        let (merged, merge_score) = self.merge_clusters(
            &left_node.cocluster_result,
            &right_node.cocluster_result,
        )?;

        Ok(MergeNode::internal(
            left_node,
            right_node,
            merged,
            Some(merge_score),
        ))
    }

    /// Merge clusters from two nodes according to the strategy
    fn merge_clusters<'a>(
        &self,
        left_clusters: &[Submatrix<'a, f64>],
        right_clusters: &[Submatrix<'a, f64>],
    ) -> Result<(Vec<Submatrix<'a, f64>>, f64), MergeError> {
        match &self.config.merge_strategy {
            MergeStrategy::Union => self.merge_union(left_clusters, right_clusters),
            MergeStrategy::Intersection { overlap_threshold } => {
                self.merge_intersection(left_clusters, right_clusters, *overlap_threshold)
            }
            MergeStrategy::Weighted {
                left_weight,
                right_weight,
            } => self.merge_weighted(left_clusters, right_clusters, *left_weight, *right_weight),
            MergeStrategy::Adaptive => self.merge_adaptive(left_clusters, right_clusters),
        }
    }

    /// Union merge: combine all clusters, remove duplicates
    /// Optimized with parallel signature computation
    fn merge_union<'a>(
        &self,
        left_clusters: &[Submatrix<'a, f64>],
        right_clusters: &[Submatrix<'a, f64>],
    ) -> Result<(Vec<Submatrix<'a, f64>>, f64), MergeError> {
        // Parallel compute signatures for all clusters
        let left_sigs: Vec<u64> = left_clusters
            .par_iter()
            .map(|c| self.cluster_signature_hash(c))
            .collect();

        let right_sigs: Vec<u64> = right_clusters
            .par_iter()
            .map(|c| self.cluster_signature_hash(c))
            .collect();

        // Build result with deduplication
        let mut seen_signatures: HashSet<u64> = HashSet::with_capacity(left_clusters.len() + right_clusters.len());
        let mut merged: Vec<Submatrix<'a, f64>> = Vec::with_capacity(left_clusters.len() + right_clusters.len());

        // Add all left clusters
        for (cluster, sig) in left_clusters.iter().zip(left_sigs.iter()) {
            if seen_signatures.insert(*sig) {
                merged.push(cluster.clone());
            }
        }

        // Add non-duplicate right clusters
        for (cluster, sig) in right_clusters.iter().zip(right_sigs.iter()) {
            if seen_signatures.insert(*sig) {
                merged.push(cluster.clone());
            }
        }

        let total = left_clusters.len() + right_clusters.len();
        let score = if total > 0 { merged.len() as f64 / total as f64 } else { 0.0 };
        Ok((merged, score))
    }

    /// Intersection merge: keep only overlapping clusters
    /// Optimized with parallel comparison of cluster pairs
    fn merge_intersection<'a>(
        &self,
        left_clusters: &[Submatrix<'a, f64>],
        right_clusters: &[Submatrix<'a, f64>],
        overlap_threshold: f64,
    ) -> Result<(Vec<Submatrix<'a, f64>>, f64), MergeError> {
        let num_comparisons = left_clusters.len() * right_clusters.len();

        if num_comparisons == 0 {
            return Ok((Vec::new(), 0.0));
        }

        // Parallel: compute all overlaps and collect matching pairs
        let results: Vec<(Option<Submatrix<'a, f64>>, f64)> = left_clusters
            .par_iter()
            .flat_map(|left_cluster| {
                right_clusters
                    .par_iter()
                    .map(move |right_cluster| {
                        let overlap = self.compute_overlap(left_cluster, right_cluster);
                        if overlap >= overlap_threshold {
                            (self.compute_union_cluster(left_cluster, right_cluster), overlap)
                        } else {
                            (None, overlap)
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // Aggregate results
        let total_overlap: f64 = results.iter().map(|(_, o)| o).sum();
        let merged: Vec<Submatrix<'a, f64>> = results
            .into_iter()
            .filter_map(|(cluster, _)| cluster)
            .collect();

        let score = total_overlap / num_comparisons as f64;
        Ok((merged, score))
    }

    /// Weighted merge: combine with scoring weights
    fn merge_weighted<'a>(
        &self,
        left_clusters: &[Submatrix<'a, f64>],
        right_clusters: &[Submatrix<'a, f64>],
        left_weight: f64,
        right_weight: f64,
    ) -> Result<(Vec<Submatrix<'a, f64>>, f64), MergeError> {
        // For now, use union merge and return weighted score
        // TODO: Implement weighted scoring integration when Scorer is available
        let (merged, _) = self.merge_union(left_clusters, right_clusters)?;
        let score = left_weight * left_clusters.len() as f64 + right_weight * right_clusters.len() as f64;
        Ok((merged, score))
    }

    /// Adaptive merge: choose strategy based on cluster properties
    fn merge_adaptive<'a>(
        &self,
        left_clusters: &[Submatrix<'a, f64>],
        right_clusters: &[Submatrix<'a, f64>],
    ) -> Result<(Vec<Submatrix<'a, f64>>, f64), MergeError> {
        // Heuristic: if cluster counts are similar, use union; otherwise use weighted
        let ratio = (left_clusters.len() as f64) / (right_clusters.len().max(1) as f64);

        if ratio > 0.5 && ratio < 2.0 {
            // Similar sizes - use union
            self.merge_union(left_clusters, right_clusters)
        } else {
            // Different sizes - use weighted merge favoring larger set
            let (left_weight, right_weight) = if left_clusters.len() > right_clusters.len() {
                (0.7, 0.3)
            } else {
                (0.3, 0.7)
            };
            self.merge_weighted(left_clusters, right_clusters, left_weight, right_weight)
        }
    }

    /// Compute a signature for cluster deduplication (legacy, returns cloned vecs)
    fn cluster_signature(&self, cluster: &Submatrix<f64>) -> (Vec<usize>, Vec<usize>) {
        (
            cluster.row_indices.clone(),
            cluster.col_indices.clone(),
        )
    }

    /// Compute a lightweight hash signature for cluster deduplication
    /// Much faster than cloning full index vectors
    fn cluster_signature_hash(&self, cluster: &Submatrix<f64>) -> u64 {
        let mut hasher = DefaultHasher::new();
        cluster.row_indices.hash(&mut hasher);
        cluster.col_indices.hash(&mut hasher);
        hasher.finish()
    }

    /// Compute overlap between two clusters (Jaccard index)
    fn compute_overlap(&self, cluster1: &Submatrix<f64>, cluster2: &Submatrix<f64>) -> f64 {
        let rows1: HashSet<_> = cluster1.row_indices.iter().collect();
        let rows2: HashSet<_> = cluster2.row_indices.iter().collect();
        let cols1: HashSet<_> = cluster1.col_indices.iter().collect();
        let cols2: HashSet<_> = cluster2.col_indices.iter().collect();

        let row_intersection = rows1.intersection(&rows2).count();
        let row_union = rows1.union(&rows2).count();
        let col_intersection = cols1.intersection(&cols2).count();
        let col_union = cols1.union(&cols2).count();

        if row_union == 0 || col_union == 0 {
            return 0.0;
        }

        let row_jaccard = row_intersection as f64 / row_union as f64;
        let col_jaccard = col_intersection as f64 / col_union as f64;

        // Geometric mean of row and column Jaccard
        (row_jaccard * col_jaccard).sqrt()
    }

    /// Compute union of two overlapping clusters
    /// Note: This is a simplified version that just returns the first cluster
    /// A full implementation would need access to the original matrix
    fn compute_union_cluster<'a>(
        &self,
        cluster1: &Submatrix<'a, f64>,
        cluster2: &Submatrix<'a, f64>,
    ) -> Option<Submatrix<'a, f64>> {
        // For now, just return a copy of cluster1
        // TODO: Implement proper union when we have access to original matrix
        Some(cluster1.clone())
    }

    /// Compute tree depth for validation
    pub fn compute_tree_depth<'a>(tree: &MergeNode<'a>) -> usize {
        tree.depth()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn create_test_submatrix<'a>(matrix_data: &'a Array2<f64>, rows: Vec<usize>, cols: Vec<usize>) -> Submatrix<'a, f64> {
        Submatrix::from_indices(matrix_data, &rows, &cols).unwrap()
    }

    #[test]
    fn test_pad_to_power_of_two() {
        let merger = HierarchicalMerger::new(HierarchicalMergeConfig::default());
        let results = vec![vec![], vec![], vec![]]; // 3 items
        let padded = merger.pad_to_power_of_two(results);
        assert_eq!(padded.len(), 4); // Next power of 2
    }

    #[test]
    fn test_merge_union() {
        let data = Array2::from_shape_vec((5, 5), (0..25).map(|x| x as f64).collect()).unwrap();

        let merger = HierarchicalMerger::new(HierarchicalMergeConfig::default());

        let left = vec![
            create_test_submatrix(&data, vec![0, 1], vec![0, 1]),
            create_test_submatrix(&data, vec![2, 3], vec![2, 3]),
        ];

        let right = vec![
            create_test_submatrix(&data, vec![0, 1], vec![0, 1]), // Duplicate
            create_test_submatrix(&data, vec![3, 4], vec![3, 4]), // New
        ];

        let (merged, score) = merger.merge_union(&left, &right).unwrap();
        assert_eq!(merged.len(), 3); // 2 unique + 1 new
        assert!(score > 0.0);
    }

    #[test]
    fn test_compute_overlap() {
        let data = Array2::from_shape_vec((5, 5), (0..25).map(|x| x as f64).collect()).unwrap();

        let merger = HierarchicalMerger::new(HierarchicalMergeConfig::default());

        let cluster1 = create_test_submatrix(&data, vec![0, 1, 2], vec![0, 1]);
        let cluster2 = create_test_submatrix(&data, vec![1, 2, 3], vec![0, 1]);

        let overlap = merger.compute_overlap(&cluster1, &cluster2);
        assert!(overlap > 0.0);
        assert!(overlap <= 1.0);
    }

    #[test]
    fn test_build_merge_tree() {
        let data = Array2::from_shape_vec((10, 10), (0..100).map(|x| x as f64).collect()).unwrap();

        let merger = HierarchicalMerger::new(HierarchicalMergeConfig::default());

        let partition_results = vec![
            vec![create_test_submatrix(&data, vec![0, 1], vec![0, 1])],
            vec![create_test_submatrix(&data, vec![2, 3], vec![2, 3])],
            vec![create_test_submatrix(&data, vec![4, 5], vec![4, 5])],
            vec![create_test_submatrix(&data, vec![6, 7], vec![6, 7])],
        ];

        let tree = merger.build_merge_tree(partition_results).unwrap();
        assert_eq!(HierarchicalMerger::compute_tree_depth(&tree), 2); // log2(4) = 2
    }
}
