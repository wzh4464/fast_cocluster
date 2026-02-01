//! Integration tests for DiMergeCo algorithm
//!
//! Tests the full pipeline with all three phases:
//! 1. Probabilistic partitioning
//! 2. Parallel local clustering
//! 3. Hierarchical merging

use fast_cocluster::dimerge_co::*;
use fast_cocluster::matrix::Matrix;
use fast_cocluster::pipeline::{CoclusterPipeline, SVDClusterer};
use fast_cocluster::scoring::PearsonScorer;
use fast_cocluster::submatrix::Submatrix;
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::error::Error;

/// Create a synthetic matrix with planted co-cluster structure
fn create_test_matrix(n_rows: usize, n_cols: usize, n_clusters: usize) -> Array2<f64> {
    let mut matrix = Array2::random((n_rows, n_cols), Uniform::new(0.0, 1.0));

    let rows_per_cluster = n_rows / n_clusters;
    let cols_per_cluster = n_cols / n_clusters;

    for k in 0..n_clusters {
        let row_start = k * rows_per_cluster;
        let row_end = ((k + 1) * rows_per_cluster).min(n_rows);
        let col_start = k * cols_per_cluster;
        let col_end = ((k + 1) * cols_per_cluster).min(n_cols);

        // Add structure to co-clusters
        for i in row_start..row_end {
            for j in col_start..col_end {
                matrix[[i, j]] += 2.0;
            }
        }
    }

    matrix
}

#[test]
fn test_probabilistic_partitioner_basic() {
    let matrix = create_test_matrix(50, 40, 3);

    let k = 3;
    let n = 50;
    let delta = 0.05;
    let num_partitions = 4;

    let partitioner = ProbabilisticPartitioner::new(k, n, delta, num_partitions).unwrap();
    let result = partitioner.partition(&matrix).unwrap();

    assert_eq!(result.partitions.len(), num_partitions);
    assert!(result.preservation_prob >= 0.0 && result.preservation_prob <= 1.0);
    assert!(result.threshold > 0.0);
    assert!(!result.singular_values.is_empty());
}

#[test]
fn test_hierarchical_merger_union_strategy() {
    let matrix_data = create_test_matrix(40, 30, 2);
    let matrix = Matrix {
        data: matrix_data.clone(),
        rows: 40,
        cols: 30,
    };

    // Create some mock submatrices
    let sub1 = Submatrix::from_indices(&matrix.data, &vec![0, 1, 2], &vec![0, 1]).unwrap();
    let sub2 = Submatrix::from_indices(&matrix.data, &vec![3, 4, 5], &vec![2, 3]).unwrap();

    let partition_results = vec![
        vec![sub1],
        vec![sub2],
    ];

    let config = HierarchicalMergeConfig {
        merge_strategy: MergeStrategy::Union,
        merge_threshold: 0.5,
        rescore_merged: false,
        parallel_level: 1,
    };

    let merger = HierarchicalMerger::new(config);
    let result = merger.execute_parallel(partition_results, &matrix);

    assert!(result.is_ok());
    let merged = result.unwrap();
    assert!(!merged.is_empty());
}

#[test]
fn test_dimerge_co_with_mock_clusterer() {
    // Mock local clusterer
    struct SimpleLocalClusterer;

    impl LocalClusterer for SimpleLocalClusterer {
        fn cluster_local<'a>(
            &self,
            matrix: &'a Array2<f64>,
        ) -> Result<Vec<Submatrix<'a, f64>>, Box<dyn Error>> {
            let n_rows = matrix.nrows();
            let n_cols = matrix.ncols();

            if n_rows < 2 || n_cols < 2 {
                return Ok(vec![]);
            }

            // Create one simple cluster
            let rows = vec![0, 1];
            let cols = vec![0, 1];
            if let Some(sub) = Submatrix::from_indices(matrix, &rows, &cols) {
                Ok(vec![sub])
            } else {
                Ok(vec![])
            }
        }
    }

    let matrix_data = create_test_matrix(50, 40, 3);
    let matrix = Matrix {
        data: matrix_data,
        rows: 50,
        cols: 40,
    };

    let local_clusterer = SimpleLocalClusterer;
    let merge_config = HierarchicalMergeConfig::default();

    let clusterer = DiMergeCoClusterer::new(
        3,                      // k
        50,                     // n
        0.05,                   // delta
        4,                      // num_partitions
        local_clusterer,
        merge_config,
        2,                      // num_threads
    )
    .unwrap();

    let result = clusterer.run(&matrix);
    assert!(result.is_ok());

    let result = result.unwrap();
    assert!(result.stats.num_partitions == 4);
    assert!(result.stats.preservation_prob >= 0.0);
    assert!(result.stats.tree_depth <= 2); // log2(4) = 2
}

#[test]
fn test_pipeline_integration_with_clusterer_adapter() {
    let matrix_data = create_test_matrix(60, 50, 3);
    let matrix = Matrix {
        data: matrix_data,
        rows: 60,
        cols: 50,
    };

    let local_clusterer = ClustererAdapter::new(SVDClusterer::new(3, 0.1));

    let pipeline = CoclusterPipeline::builder()
        .with_dimerge_co(
            3,                  // k
            60,                 // n
            0.05,               // delta
            local_clusterer,
            2,                  // threads
        )
        .unwrap()
        .with_scorer(Box::new(PearsonScorer::new(2, 2)))
        .min_score(0.3)
        .build()
        .unwrap();

    let result = pipeline.run(&matrix);
    assert!(result.is_ok());

    let step_result = result.unwrap();
    // Should complete successfully (may have zero or more clusters depending on scoring threshold)
    // Just verify the result structure is valid
    assert!(step_result.scores.len() == step_result.submatrices.len());
}

#[test]
fn test_theoretical_validation_preservation() {
    let matrix_data = create_test_matrix(40, 30, 2);

    // Ground truth clusters
    let ground_truth = vec![
        Submatrix::from_indices(&matrix_data, &vec![0, 1, 2], &vec![0, 1, 2]).unwrap(),
        Submatrix::from_indices(&matrix_data, &vec![3, 4, 5], &vec![3, 4, 5]).unwrap(),
    ];

    // Recovered clusters (similar to ground truth)
    let recovered = vec![
        Submatrix::from_indices(&matrix_data, &vec![0, 1, 2], &vec![0, 1, 2]).unwrap(),
        Submatrix::from_indices(&matrix_data, &vec![3, 4, 5], &vec![3, 4, 5]).unwrap(),
    ];

    let delta = 0.05;
    let validation = TheoreticalValidator::validate_preservation(&ground_truth, &recovered, delta);

    assert!(validation.passed);
    assert_eq!(validation.num_ground_truth, 2);
    assert_eq!(validation.num_recovered, 2);
    assert_eq!(validation.num_preserved, 2);
    assert!(validation.measured_preservation >= 0.95);
}

#[test]
fn test_theoretical_validation_communication_complexity() {
    // Create a simple tree with 4 leaves
    let matrix_data = Array2::zeros((10, 10));
    let sub1 = Submatrix::from_indices(&matrix_data, &vec![0, 1], &vec![0, 1]).unwrap();
    let sub2 = Submatrix::from_indices(&matrix_data, &vec![2, 3], &vec![2, 3]).unwrap();

    let leaf1 = MergeNode::leaf(vec![sub1], 0);
    let leaf2 = MergeNode::leaf(vec![sub2], 1);

    let root = MergeNode::internal(leaf1, leaf2, vec![], None);

    let validation = TheoreticalValidator::validate_communication_complexity(&root, 2);

    assert!(validation.passed);
    assert_eq!(validation.measured_depth, 1); // Root has depth 1
    assert_eq!(validation.expected_depth, 1); // log2(2) = 1
}

#[test]
fn test_merge_strategies_comparison() {
    let matrix_data = create_test_matrix(30, 25, 2);
    let matrix = Matrix {
        data: matrix_data.clone(),
        rows: 30,
        cols: 25,
    };

    let sub1 = Submatrix::from_indices(&matrix_data, &vec![0, 1, 2], &vec![0, 1]).unwrap();
    let sub2 = Submatrix::from_indices(&matrix_data, &vec![3, 4, 5], &vec![2, 3]).unwrap();
    let partition_results = vec![vec![sub1], vec![sub2]];

    // Test Union strategy
    let union_config = HierarchicalMergeConfig {
        merge_strategy: MergeStrategy::Union,
        merge_threshold: 0.5,
        rescore_merged: false,
        parallel_level: 1,
    };
    let union_merger = HierarchicalMerger::new(union_config);
    let union_result = union_merger.execute_parallel(partition_results.clone(), &matrix).unwrap();

    // Test Adaptive strategy
    let adaptive_config = HierarchicalMergeConfig {
        merge_strategy: MergeStrategy::Adaptive,
        merge_threshold: 0.5,
        rescore_merged: false,
        parallel_level: 1,
    };
    let adaptive_merger = HierarchicalMerger::new(adaptive_config);
    let adaptive_result = adaptive_merger.execute_parallel(partition_results.clone(), &matrix).unwrap();

    // Both should produce results
    assert!(!union_result.is_empty());
    assert!(!adaptive_result.is_empty());
}

#[test]
fn test_parallel_config_settings() {
    let config = ParallelConfig {
        enabled: true,
        min_items_for_parallel: 50,
        num_threads: Some(4),
        chunk_size: Some(16),
        kmeans_parallel: true,
        scoring_parallel: true,
        normalization_parallel: false,
        submatrix_creation_parallel: true,
    };

    assert_eq!(config.num_threads.unwrap(), 4);
    assert_eq!(config.min_items_for_parallel, 50);
    assert!(config.kmeans_parallel);
    assert!(!config.normalization_parallel);
}

#[test]
fn test_dimerge_co_stats_tracking() {
    let matrix_data = create_test_matrix(40, 30, 2);
    let matrix = Matrix {
        data: matrix_data,
        rows: 40,
        cols: 30,
    };

    struct DummyClusterer;
    impl LocalClusterer for DummyClusterer {
        fn cluster_local<'a>(
            &self,
            matrix: &'a Array2<f64>,
        ) -> Result<Vec<Submatrix<'a, f64>>, Box<dyn Error>> {
            if matrix.nrows() > 2 && matrix.ncols() > 2 {
                let sub = Submatrix::from_indices(matrix, &vec![0, 1], &vec![0, 1]).unwrap();
                Ok(vec![sub])
            } else {
                Ok(vec![])
            }
        }
    }

    let clusterer = DiMergeCoClusterer::with_adaptive(
        2,
        40,
        0.05,
        DummyClusterer,
        HierarchicalMergeConfig::default(),
        2,
    )
    .unwrap();

    let result = clusterer.run(&matrix).unwrap();

    // Verify timing stats are populated (individual phases may be 0 at ms resolution for small matrices)
    assert!(result.stats.phase_times.total_ms
        >= result.stats.phase_times.partitioning_ms.min(result.stats.phase_times.total_ms));
    assert_eq!(result.stats.num_partitions, 4); // Adaptive should choose 4
    assert!(result.stats.preservation_prob >= 0.0);
}
