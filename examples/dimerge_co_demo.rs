//! # DiMergeCo Example
//!
//! Demonstrates basic usage of the DiMergeCo divide-merge co-clustering algorithm.

/**
 * File: /examples/dimerge_co_demo.rs
 * Created Date: Monday, January 27th 2026
 * Author: Zihan
 * -----
 * Last Modified: Monday, 27th January 2026
 * Modified By: Zihan Wu <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
 * 2026-01-27		Zihan	Created DiMergeCo demonstration example
 */

use fast_cocluster::dimerge_co::*;
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn main() {
    env_logger::init();

    println!("=== DiMergeCo Demonstration ===\n");

    // Create a synthetic matrix with co-cluster structure
    let matrix_data = create_synthetic_matrix(100, 80, 3);

    println!("Matrix shape: {:?}", matrix_data.dim());
    println!("Expected co-clusters: 3\n");

    // Test probabilistic partitioning
    test_probabilistic_partitioning(&matrix_data);

    // Test hierarchical merging
    test_hierarchical_merging();

    // Test theoretical validation
    test_theoretical_validation();

    println!("\n=== Demo Complete ===");
}

/// Create a synthetic matrix with planted co-cluster structure
fn create_synthetic_matrix(n_rows: usize, n_cols: usize, n_clusters: usize) -> Array2<f64> {
    // Create random matrix
    let mut matrix = Array2::random((n_rows, n_cols), Uniform::new(0.0, 1.0));

    // Add structure for co-clusters
    let rows_per_cluster = n_rows / n_clusters;
    let cols_per_cluster = n_cols / n_clusters;

    for k in 0..n_clusters {
        let row_start = k * rows_per_cluster;
        let row_end = ((k + 1) * rows_per_cluster).min(n_rows);
        let col_start = k * cols_per_cluster;
        let col_end = ((k + 1) * cols_per_cluster).min(n_cols);

        // Add high correlation within each cluster
        for i in row_start..row_end {
            for j in col_start..col_end {
                matrix[[i, j]] += 2.0; // Boost values in co-clusters
            }
        }
    }

    matrix
}

/// Test probabilistic partitioning
fn test_probabilistic_partitioning(matrix: &Array2<f64>) {
    println!("--- Testing Probabilistic Partitioning ---");

    let k = 3;  // Expected number of co-clusters
    let n = matrix.nrows();
    let delta = 0.05;  // 95% preservation probability
    let num_partitions = 4;  // Power of 2

    match ProbabilisticPartitioner::new(k, n, delta, num_partitions) {
        Ok(partitioner) => {
            match partitioner.partition(matrix) {
                Ok(result) => {
                    println!("✓ Created {} partitions", result.partitions.len());
                    println!("✓ Threshold τ = {:.4}", result.threshold);
                    println!("✓ Preservation probability = {:.3}", result.preservation_prob);
                    println!("✓ Singular values: {:?}", &result.singular_values[..3.min(result.singular_values.len())]);

                    // Show partition sizes
                    for (i, partition) in result.partitions.iter().enumerate() {
                        let (rows, cols) = partition.size();
                        println!("  Partition {}: {}×{} elements", i, rows, cols);
                    }
                }
                Err(e) => println!("✗ Partitioning failed: {}", e),
            }
        }
        Err(e) => println!("✗ Failed to create partitioner: {}", e),
    }

    println!();
}

/// Test hierarchical merging
fn test_hierarchical_merging() {
    println!("--- Testing Hierarchical Merging ---");

    let config = HierarchicalMergeConfig {
        merge_strategy: MergeStrategy::Union,
        merge_threshold: 0.5,
        rescore_merged: true,
        parallel_level: 2,
    };

    let _merger = HierarchicalMerger::new(config);

    println!("✓ Created hierarchical merger");
    println!("  Strategy: Union");
    println!("  Parallel level: 2");
    println!("  Expected tree depth: log₂(4) = 2");
    println!();
}

/// Test theoretical validation
fn test_theoretical_validation() {
    println!("--- Testing Theoretical Validation ---");

    // Test spectral gap validation
    let singular_values = vec![10.0, 5.0, 2.5, 0.8, 0.3];
    let k = 3;
    let tau = 0.5;

    let gap_valid = TheoreticalValidator::validate_spectral_gap(&singular_values, k, tau);
    println!("✓ Spectral gap validation: {}", if gap_valid { "PASS" } else { "FAIL" });
    println!("  Gap (σ₃ - σ₄) = {:.1} {} τ = {:.1}",
             singular_values[k-1] - singular_values[k],
             if gap_valid { ">" } else { "≤" },
             tau);

    // Test convergence validation
    let iterations = vec![1.0, 0.5, 0.25, 0.125];
    let bound_fn = |i: usize| 2.0_f64.powi(-(i as i32));

    let convergence = TheoreticalValidator::validate_convergence_bounds(&iterations, bound_fn);
    println!("✓ Convergence validation: {}", if convergence.passed { "PASS" } else { "FAIL" });
    println!("  Iterations: {}", convergence.errors.len());
    println!("  Final error: {:.4} (bound: {:.4})",
             convergence.errors.last().unwrap(),
             convergence.bounds.last().unwrap());

    println!();
}
