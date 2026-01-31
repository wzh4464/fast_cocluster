//! # Theoretical Validation
//!
//! Validates theoretical guarantees of the DiMergeCo algorithm:
//! - Preservation probability ≥ 1-δ
//! - Communication complexity O(log n)
//! - Convergence bounds

/**
 * File: /src/dimerge_co/theoretical_validation.rs
 * Created Date: Monday, January 27th 2026
 * Author: Zihan
 * -----
 * Last Modified: Monday, 27th January 2026
 * Modified By: Zihan Wu <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
 * 2026-01-27		Zihan	Implemented theoretical validation for DiMergeCo guarantees
 */

use crate::dimerge_co::types::*;
use crate::submatrix::Submatrix;
use std::collections::HashSet;

/// Result of preservation validation
#[derive(Debug, Clone)]
pub struct PreservationValidation {
    /// Whether validation passed (measured_preservation ≥ 1-δ)
    pub passed: bool,
    /// Measured preservation ratio
    pub measured_preservation: f64,
    /// Expected minimum (1-δ)
    pub expected_minimum: f64,
    /// Number of ground truth clusters
    pub num_ground_truth: usize,
    /// Number of recovered clusters
    pub num_recovered: usize,
    /// Number of correctly preserved clusters
    pub num_preserved: usize,
}

/// Result of complexity validation
#[derive(Debug, Clone)]
pub struct ComplexityValidation {
    /// Whether validation passed (depth ≤ log₂(num_partitions))
    pub passed: bool,
    /// Measured tree depth
    pub measured_depth: usize,
    /// Expected maximum depth
    pub expected_depth: usize,
    /// Number of partitions
    pub num_partitions: usize,
}

/// Result of convergence validation
#[derive(Debug, Clone)]
pub struct ConvergenceValidation {
    /// Whether validation passed (error within bounds)
    pub passed: bool,
    /// Measured error at each iteration
    pub errors: Vec<f64>,
    /// Expected error bounds at each iteration
    pub bounds: Vec<f64>,
}

/// Theoretical validator for DiMergeCo guarantees
pub struct TheoreticalValidator;

impl TheoreticalValidator {
    /// Validate that co-cluster preservation probability ≥ 1-δ
    ///
    /// Compares ground truth clusters with recovered clusters using Jaccard similarity
    pub fn validate_preservation(
        ground_truth: &[Submatrix<f64>],
        recovered: &[Submatrix<f64>],
        delta: f64,
    ) -> PreservationValidation {
        let expected_minimum = 1.0 - delta;
        let num_ground_truth = ground_truth.len();
        let num_recovered = recovered.len();

        // For each ground truth cluster, find best matching recovered cluster
        let mut num_preserved = 0;

        for gt_cluster in ground_truth {
            let mut best_match: f64 = 0.0;

            for rec_cluster in recovered {
                let similarity = Self::jaccard_similarity(gt_cluster, rec_cluster);
                best_match = best_match.max(similarity);
            }

            // Consider preserved if similarity > threshold (e.g., 0.7)
            if best_match >= 0.7 {
                num_preserved += 1;
            }
        }

        let measured_preservation = if num_ground_truth > 0 {
            num_preserved as f64 / num_ground_truth as f64
        } else {
            1.0
        };

        let passed = measured_preservation >= expected_minimum;

        PreservationValidation {
            passed,
            measured_preservation,
            expected_minimum,
            num_ground_truth,
            num_recovered,
            num_preserved,
        }
    }

    /// Validate O(log n) communication complexity via tree depth
    pub fn validate_communication_complexity<'a>(
        tree: &MergeNode<'a>,
        num_partitions: usize,
    ) -> ComplexityValidation {
        let measured_depth = tree.depth();
        let expected_depth = if num_partitions > 0 {
            (num_partitions as f64).log2().ceil() as usize
        } else {
            0
        };

        let passed = measured_depth <= expected_depth;

        ComplexityValidation {
            passed,
            measured_depth,
            expected_depth,
            num_partitions,
        }
    }

    /// Validate convergence bounds
    ///
    /// Checks that error decreases according to theoretical bound function
    pub fn validate_convergence_bounds<F>(
        iterations: &[f64],
        bound_function: F,
    ) -> ConvergenceValidation
    where
        F: Fn(usize) -> f64,
    {
        let errors = iterations.to_vec();
        let bounds: Vec<f64> = (0..iterations.len())
            .map(|i| bound_function(i))
            .collect();

        // Check if all errors are within bounds
        let passed = errors
            .iter()
            .zip(bounds.iter())
            .all(|(error, bound)| error <= bound);

        ConvergenceValidation {
            passed,
            errors,
            bounds,
        }
    }

    /// Compute Jaccard similarity between two clusters
    fn jaccard_similarity(cluster1: &Submatrix<f64>, cluster2: &Submatrix<f64>) -> f64 {
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

    /// Validate spectral gap condition: σ_k - σ_{k+1} > τ
    pub fn validate_spectral_gap(
        singular_values: &[f64],
        k: usize,
        tau: f64,
    ) -> bool {
        if singular_values.len() <= k {
            return false;
        }

        let gap = singular_values[k - 1] - singular_values[k];
        gap > tau
    }

    /// Comprehensive validation report
    pub fn generate_validation_report(
        preservation: &PreservationValidation,
        complexity: &ComplexityValidation,
        convergence: Option<&ConvergenceValidation>,
    ) -> String {
        let mut report = String::new();

        report.push_str("=== DiMergeCo Theoretical Validation Report ===\n\n");

        // Preservation validation
        report.push_str("1. Preservation Guarantee:\n");
        report.push_str(&format!(
            "   Status: {}\n",
            if preservation.passed { "PASS ✓" } else { "FAIL ✗" }
        ));
        report.push_str(&format!(
            "   Measured: {:.3} (expected: ≥{:.3})\n",
            preservation.measured_preservation, preservation.expected_minimum
        ));
        report.push_str(&format!(
            "   Preserved: {}/{} clusters\n\n",
            preservation.num_preserved, preservation.num_ground_truth
        ));

        // Complexity validation
        report.push_str("2. Communication Complexity:\n");
        report.push_str(&format!(
            "   Status: {}\n",
            if complexity.passed { "PASS ✓" } else { "FAIL ✗" }
        ));
        report.push_str(&format!(
            "   Tree Depth: {} (expected: ≤{})\n",
            complexity.measured_depth, complexity.expected_depth
        ));
        report.push_str(&format!(
            "   Partitions: {}\n\n",
            complexity.num_partitions
        ));

        // Convergence validation (if provided)
        if let Some(conv) = convergence {
            report.push_str("3. Convergence Bounds:\n");
            report.push_str(&format!(
                "   Status: {}\n",
                if conv.passed { "PASS ✓" } else { "FAIL ✗" }
            ));
            report.push_str(&format!(
                "   Iterations: {}\n",
                conv.errors.len()
            ));
            if !conv.errors.is_empty() {
                report.push_str(&format!(
                    "   Final Error: {:.6} (bound: {:.6})\n\n",
                    conv.errors.last().unwrap(),
                    conv.bounds.last().unwrap()
                ));
            }
        }

        report.push_str("===========================================\n");

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn create_test_cluster(matrix_data: &Array2<f64>, rows: Vec<usize>, cols: Vec<usize>) -> Submatrix<'_, f64> {
        Submatrix::from_indices(matrix_data, &rows, &cols).unwrap()
    }

    #[test]
    fn test_jaccard_similarity_identical() {
        let data = Array2::from_shape_vec((5, 5), (0..25).map(|x| x as f64).collect()).unwrap();

        let cluster1 = create_test_cluster(&data, vec![0, 1, 2], vec![0, 1]);
        let cluster2 = create_test_cluster(&data, vec![0, 1, 2], vec![0, 1]);

        let similarity = TheoreticalValidator::jaccard_similarity(&cluster1, &cluster2);
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_similarity_partial_overlap() {
        let data = Array2::from_shape_vec((5, 5), (0..25).map(|x| x as f64).collect()).unwrap();

        let cluster1 = create_test_cluster(&data, vec![0, 1, 2], vec![0, 1]);
        let cluster2 = create_test_cluster(&data, vec![1, 2, 3], vec![0, 1]);

        let similarity = TheoreticalValidator::jaccard_similarity(&cluster1, &cluster2);
        assert!(similarity > 0.0 && similarity < 1.0);
    }

    #[test]
    fn test_validate_preservation() {
        let data = Array2::from_shape_vec((10, 10), (0..100).map(|x| x as f64).collect()).unwrap();

        let ground_truth = vec![
            create_test_cluster(&data, vec![0, 1], vec![0, 1]),
            create_test_cluster(&data, vec![2, 3], vec![2, 3]),
        ];

        let recovered = vec![
            create_test_cluster(&data, vec![0, 1], vec![0, 1]),  // Perfect match
            create_test_cluster(&data, vec![2, 3], vec![2, 3]),  // Perfect match
        ];

        let validation = TheoreticalValidator::validate_preservation(&ground_truth, &recovered, 0.05);
        assert!(validation.passed);
        assert_eq!(validation.num_preserved, 2);
    }

    #[test]
    fn test_validate_communication_complexity() {
        let tree = MergeNode::leaf(vec![], 0);
        let validation = TheoreticalValidator::validate_communication_complexity(&tree, 1);
        assert!(validation.passed);
        assert_eq!(validation.measured_depth, 0);
    }

    #[test]
    fn test_validate_convergence_bounds() {
        let iterations = vec![1.0, 0.5, 0.25, 0.125];
        let bound_fn = |i: usize| 2.0_f64.powi(-(i as i32));

        let validation = TheoreticalValidator::validate_convergence_bounds(&iterations, bound_fn);
        assert!(validation.passed);
    }

    #[test]
    fn test_validate_spectral_gap() {
        let singular_values = vec![10.0, 5.0, 2.0, 0.5, 0.1];
        let k = 3;
        let tau = 1.0;

        let result = TheoreticalValidator::validate_spectral_gap(&singular_values, k, tau);
        assert!(result); // gap = 2.0 - 0.5 = 1.5 > 1.0
    }

    #[test]
    fn test_generate_validation_report() {
        let preservation = PreservationValidation {
            passed: true,
            measured_preservation: 0.95,
            expected_minimum: 0.95,
            num_ground_truth: 10,
            num_recovered: 10,
            num_preserved: 10,
        };

        let complexity = ComplexityValidation {
            passed: true,
            measured_depth: 2,
            expected_depth: 2,
            num_partitions: 4,
        };

        let report = TheoreticalValidator::generate_validation_report(&preservation, &complexity, None);
        assert!(report.contains("PASS ✓"));
        assert!(report.contains("0.950"));
    }
}
