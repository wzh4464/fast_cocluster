//! # DiMergeCo Pipeline Integration Demo
//!
//! Demonstrates using DiMergeCo through the Pipeline API

/**
 * File: /examples/dimerge_co_pipeline_demo.rs
 * Created Date: Monday, January 27th 2026
 * Author: Zihan
 * -----
 * Last Modified: Monday, 27th January 2026
 * Modified By: Zihan Wu <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
 * 2026-01-27		Zihan	Created DiMergeCo pipeline integration demo
 */

use fast_cocluster::dimerge_co::ClustererAdapter;
use fast_cocluster::matrix::Matrix;
use fast_cocluster::pipeline::{CoclusterPipeline, SVDClusterer};
use fast_cocluster::scoring::PearsonScorer;
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("=== DiMergeCo Pipeline Integration Demo ===\n");

    // Create a synthetic matrix with co-cluster structure
    let matrix_data = create_synthetic_matrix(200, 150, 5);
    let matrix = Matrix {
        data: matrix_data,
        rows: 200,
        cols: 150,
    };

    println!("Matrix shape: {}×{}", matrix.rows, matrix.cols);
    println!("Expected co-clusters: 5\n");

    // Example 1: Using SVDClusterer as local clusterer via adapter
    println!("--- Example 1: DiMergeCo with SVD Local Clusterer ---");
    let local_clusterer = ClustererAdapter::new(SVDClusterer::new(3, 0.1));

    let pipeline = CoclusterPipeline::builder()
        .with_dimerge_co(
            5,           // k: expected number of co-clusters
            200,         // n: total samples
            0.05,        // δ: 95% preservation probability
            local_clusterer,
            4,           // 4 threads
        )?
        .with_scorer(Box::new(PearsonScorer::new(3, 3)))
        .min_score(0.5)
        .build()?;

    let result = pipeline.run(&matrix)?;
    println!("✓ Found {} co-clusters", result.submatrices.len());
    println!("  Top 3 by score:");
    for (i, sub) in result.submatrices.iter().take(3).enumerate() {
        println!(
            "    {}. {}×{} (rows: {:?}, cols: {:?})",
            i + 1,
            sub.row_indices.len(),
            sub.col_indices.len(),
            &sub.row_indices[..sub.row_indices.len().min(5)],
            &sub.col_indices[..sub.col_indices.len().min(5)]
        );
    }

    println!("\n--- Example 2: DiMergeCo with Explicit Configuration ---");

    use fast_cocluster::dimerge_co::{HierarchicalMergeConfig, MergeStrategy};

    let local_clusterer2 = ClustererAdapter::new(SVDClusterer::new(3, 0.1));

    let custom_merge_config = HierarchicalMergeConfig {
        merge_strategy: MergeStrategy::Union,
        merge_threshold: 0.6,
        rescore_merged: true,
        parallel_level: 3,
    };

    let pipeline2 = CoclusterPipeline::builder()
        .with_dimerge_co_explicit(
            5,                      // k
            200,                    // n
            0.05,                   // δ
            8,                      // num_partitions (power of 2)
            local_clusterer2,
            custom_merge_config,
            4,                      // threads
        )?
        .with_scorer(Box::new(PearsonScorer::new(3, 3)))
        .min_score(0.4)
        .build()?;

    let result2 = pipeline2.run(&matrix)?;
    println!("✓ Found {} co-clusters with Union merge strategy", result2.submatrices.len());

    println!("\n=== Demo Complete ===");
    Ok(())
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
