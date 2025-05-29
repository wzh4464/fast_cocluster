/**
 * File: ./src/probability.rs
 * Created Date: Thursday, May 29th 2025
 * Author: Zihan
 * -----
 * Last Modified: Thursday, 29th May 2025 10:55:05 am
 * Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
**/

use std::collections::HashSet;
use crate::partitioner::{CoCluster, PartitionResult};

/// Calculate joint probability P(M_ij^k < T_m, N_ij^k < T_n)
pub fn joint_probability(m_k: usize, n_k: usize, m_total: usize, n_total: usize,
                        phi_i: usize, psi_j: usize, t_m: usize, t_n: usize) -> f64 {
    // Calculate s_i^(k) and t_j^(k)
    let s_i_k = (m_k as f64 / m_total as f64) - ((t_m - 1) as f64 / phi_i as f64);
    let t_j_k = (n_k as f64 / n_total as f64) - ((t_n - 1) as f64 / psi_j as f64);
    
    // Apply Hoeffding's inequality
    let exponent = -2.0 * s_i_k.powi(2) * phi_i as f64 - 2.0 * t_j_k.powi(2) * psi_j as f64;
    
    // Handle NaN/Infinity cases more robustly
    if exponent.is_nan() || exponent.is_infinite() {
        if s_i_k > 0.0 && t_j_k > 0.0 { // if s_i_k or t_j_k is positive, exp will be large negative -> 0
            return 0.0;
        } else { // if s_i_k or t_j_k is negative, exp will be large positive -> extremely large, effectively infinity (bad for prob)
                 // This case should ideally be handled by s_k.max(0.0) logic before this.
                 // If it still results in positive exponent here, it implies an issue or large values.
                 // A probability cannot be > 1.0. exp() of a large positive is >> 1.
                 // However, the formula itself defines P as exp(negative_terms), so it should be <= 1.
                 // If exponent becomes positive due to negative s_i_k/t_j_k (squared makes them positive),
                 // and if phi_i/psi_j are somehow negative (which they shouldn't be),
                 // this is an issue. Assuming phi_i, psi_j always non-negative.
                 // If s_i_k or t_j_k were negative, their squares are positive.
                 // So exponent = -2 * (positive) * (positive) - 2 * (positive) * (positive) = negative.
                 // exp(negative) is between 0 and 1.
                 // So, NaN/Infinity here is likely from phi_i/psi_j being zero or problematic m_k/n_k etc.
            return 0.0; // A safe default for invalid calculations leading to NaN/Inf in exponent for probability
        }
    }
    exponent.exp()
}

/// Calculate overall detection probability
pub fn detection_probability(coclusters: &[CoCluster], partition: &PartitionResult, 
                           t_p: usize, t_m: usize, t_n: usize) -> f64 {
    let m_total: usize = partition.block_rows.iter().sum();
    let n_total: usize = partition.block_cols.iter().sum();
    
    let m_blocks_len = partition.block_rows.len() as f64;
    let n_blocks_len = partition.block_cols.len() as f64;
    
    if m_blocks_len == 0.0 || n_blocks_len == 0.0 {
        return 0.0; // Avoid division by zero if there are no blocks
    }

    let phi_avg = partition.block_rows.iter().sum::<usize>() as f64 / m_blocks_len;
    let psi_avg = partition.block_cols.iter().sum::<usize>() as f64 / n_blocks_len;
    
    let mut min_exponent = f64::INFINITY;
    
    if coclusters.is_empty() {
        return 1.0; // Or 0.0, depending on convention for no coclusters to detect
    }

    for cocluster in coclusters {
        // Calculate minimum s^(k) and t^(k)
        let s_k = compute_min_s_k(cocluster, &partition.block_rows, m_total, t_m);
        let t_k = compute_min_t_k(cocluster, &partition.block_cols, n_total, t_n);
        
        let exponent = 2.0 * t_p as f64 * 
            (phi_avg * m_blocks_len * s_k.powi(2) + psi_avg * n_blocks_len * t_k.powi(2));
        
        min_exponent = min_exponent.min(exponent);
    }
    
    if min_exponent == f64::INFINITY {
        0.0 // Should ideally not happen if coclusters is not empty
    } else {
        1.0 - (-min_exponent).exp()
    }
}

fn compute_min_s_k(cocluster: &CoCluster, block_rows: &[usize], 
                   m_total: usize, t_m: usize) -> f64 {
    let mut min_s = f64::INFINITY;
    
    if m_total == 0 { return 0.0; }

    for &phi_i in block_rows {
        if phi_i == 0 { continue; } // Avoid division by zero
        let s_i_k = (cocluster.row_size as f64 / m_total as f64) - 
                   ((t_m - 1) as f64 / phi_i as f64);
        min_s = min_s.min(s_i_k);
    }
    
    if min_s == f64::INFINITY { 0.0 } else { min_s.max(0.0) }
}

fn compute_min_t_k(cocluster: &CoCluster, block_cols: &[usize], 
                   n_total: usize, t_n: usize) -> f64 {
    let mut min_t = f64::INFINITY;

    if n_total == 0 { return 0.0; }
    
    for &psi_j in block_cols {
        if psi_j == 0 { continue; } // Avoid division by zero
        let t_j_k = (cocluster.col_size as f64 / n_total as f64) - 
                   ((t_n - 1) as f64 / psi_j as f64);
        min_t = min_t.min(t_j_k);
    }
    
    if min_t == f64::INFINITY { 0.0 } else { min_t.max(0.0) }
}

/// Calculate probability bound for a single co-cluster
pub fn single_cocluster_bound(cocluster: &CoCluster, m_total: usize, n_total: usize,
                             phi: usize, psi: usize, _t_m: usize, _t_n: usize) -> f64 {
    if m_total == 0 || n_total == 0 || phi == 0 || psi == 0 {
        return 0.0;
    }
    let p_row = 1.0 - (-(cocluster.row_size as f64 / m_total as f64).powi(2) * 
                       2.0 * phi as f64).exp();
    let p_col = 1.0 - (-(cocluster.col_size as f64 / n_total as f64).powi(2) * 
                       2.0 * psi as f64).exp();
    
    p_row * p_col
}

#[cfg(test)]
mod tests {
    use super::*; // Imports CoCluster, PartitionResult, and probability functions
    use crate::partitioner::{CoCluster as PartitionerCoCluster, PartitionResult as PartitionerPartitionResult}; // To create instances for testing

    // Helper to create a dummy CoCluster for testing probability functions
    fn mock_cocluster(row_size: usize, col_size: usize) -> PartitionerCoCluster {
        PartitionerCoCluster {
            row_indices: (0..row_size).collect(),
            col_indices: (0..col_size).collect(),
            row_size,
            col_size,
        }
    }

    // Helper to create a dummy PartitionResult for testing probability functions
    fn mock_partition_result(block_rows: Vec<usize>, block_cols: Vec<usize>) -> PartitionerPartitionResult {
        PartitionerPartitionResult {
            block_rows,
            block_cols,
            iterations: 1, // Dummy value
            probability: 0.0, // Dummy value, will be calculated by detection_probability
        }
    }

    #[test]
    fn test_joint_probability_basic() {
        // Test with some arbitrary realistic values
        let prob = joint_probability(50, 50, 1000, 1000, 100, 100, 10, 10);
        // Expected value depends on the formula, this is more of a smoke test
        // that it runs and produces a value in the expected range (0 to 1)
        assert!(prob >= 0.0 && prob <= 1.0, "Joint probability out of bounds: {}", prob);
        
        // Test case where s_i_k or t_j_k might be negative, leading to large exp value (near 0 prob)
        // Here m_k/m_total is small, and (t_m-1)/phi_i is larger
        let prob_low_overlap = joint_probability(10, 10, 1000, 1000, 100, 100, 50, 50);
        assert!(prob_low_overlap >= 0.0 && prob_low_overlap <= 1.0, "Joint probability (low overlap) out of bounds: {}", prob_low_overlap);
        // Expect prob_low_overlap to be very small
        assert!(prob_low_overlap < 0.0001, "Expected very low probability for low overlap, got: {}", prob_low_overlap);
    }

    #[test]
    fn test_detection_probability_basic() {
        let coclusters = vec![mock_cocluster(50, 50)];
        let partition = mock_partition_result(vec![100, 100, 100], vec![100, 100, 100]);
        let t_p = 5;
        let t_m = 10;
        let t_n = 10;
        let prob = detection_probability(&coclusters, &partition, t_p, t_m, t_n);
        assert!(prob >= 0.0 && prob <= 1.0, "Detection probability out of bounds: {}", prob);
    }

    #[test]
    fn test_detection_probability_no_coclusters() {
        let coclusters: Vec<PartitionerCoCluster> = Vec::new();
        let partition = mock_partition_result(vec![100], vec![100]);
        let prob = detection_probability(&coclusters, &partition, 1, 10, 10);
        // Conventionally, if there are no co-clusters, the probability of detecting them might be considered 1 (nothing to fail to detect)
        // or 0 (nothing to detect). The implementation returns 1.0.
        assert_eq!(prob, 1.0, "Detection probability for no coclusters should be 1.0");
    }

    #[test]
    fn test_detection_probability_varying_tp() {
        let coclusters = vec![mock_cocluster(20, 20)];
        let partition = mock_partition_result(vec![50, 50], vec![50, 50]);
        let t_m = 5;
        let t_n = 5;

        let prob_t1 = detection_probability(&coclusters, &partition, 1, t_m, t_n);
        let prob_t10 = detection_probability(&coclusters, &partition, 10, t_m, t_n);
        // Check for NaN before comparison
        assert!(!prob_t1.is_nan(), "prob_t1 is NaN");
        assert!(!prob_t10.is_nan(), "prob_t10 is NaN");
        assert!(prob_t10 >= prob_t1, "Probability should generally increase with t_p ({:.4} vs {:.4})", prob_t10, prob_t1);
    }
    
    #[test]
    fn test_compute_min_s_k_t_k() {
        let cocluster = mock_cocluster(100, 80); // row_size, col_size
        let m_total = 1000;
        let n_total = 800;
        let t_m = 10;
        let t_n = 10;

        // Test compute_min_s_k
        let block_rows1 = vec![50, 50, 50]; // phi_i values
        let s_k1 = compute_min_s_k(&cocluster, &block_rows1, m_total, t_m);
        // s_i_k = (100/1000) - (9/50) = 0.1 - 0.18 = -0.08. max(0, -0.08) = 0
        assert!((s_k1 - 0.0).abs() < 1e-9, "s_k1 calculation error, got {}", s_k1);

        let block_rows2 = vec![200, 200]; // Larger phi_i should yield positive s_k if possible
        // s_i_k_for_200 = (100/1000) - (9/200) = 0.1 - 0.045 = 0.055
        let s_k2 = compute_min_s_k(&cocluster, &block_rows2, m_total, t_m);
        assert!((s_k2 - 0.055).abs() < 1e-9, "s_k2 calculation error, got {}", s_k2);

        // Test compute_min_t_k
        let block_cols1 = vec![40, 40, 40]; // psi_j values
        let t_k1 = compute_min_t_k(&cocluster, &block_cols1, n_total, t_n);
        // t_j_k = (80/800) - (9/40) = 0.1 - 0.225 = -0.125. max(0, -0.125) = 0
        assert!((t_k1 - 0.0).abs() < 1e-9, "t_k1 calculation error, got {}", t_k1);

        let block_cols2 = vec![100, 100, 100, 100];
        // t_j_k_for_100 = (80/800) - (9/100) = 0.1 - 0.09 = 0.01
        let t_k2 = compute_min_t_k(&cocluster, &block_cols2, n_total, t_n);
        assert!((t_k2 - 0.01).abs() < 1e-9, "t_k2 calculation error, got {}", t_k2);
    }

    #[test]
    fn test_single_cocluster_bound() {
        let cocluster = mock_cocluster(100, 50);
        let bound = single_cocluster_bound(&cocluster, 1000, 500, 200, 100, 10, 10);
        assert!(bound >= 0.0 && bound <= 1.0, "Single cocluster bound out of range: {}", bound);
        // Example: p_row = 1 - exp(-(100/1000)^2 * 2 * 200) = 1 - exp(-0.01 * 400) = 1 - exp(-4) approx 1 - 0.0183 = 0.9817
        // p_col = 1 - exp(-(50/500)^2 * 2 * 100) = 1 - exp(-0.01 * 200) = 1 - exp(-2) approx 1 - 0.1353 = 0.8647
        // bound approx 0.9817 * 0.8647 = 0.8488
        let p_row_val = 1.0 - (-(cocluster.row_size as f64 / 1000.0).powi(2) * 2.0 * 200.0 as f64).exp();
        let p_col_val = 1.0 - (-(cocluster.col_size as f64 / 500.0).powi(2) * 2.0 * 100.0 as f64).exp();
        let expected_bound = p_row_val * p_col_val;
        assert!((bound - expected_bound).abs() < 1e-9, "single_cocluster_bound calculation error. Expected {}, got {}", expected_bound, bound);
    }

     #[test]
    fn test_detection_probability_with_zero_blocks() {
        let coclusters = vec![mock_cocluster(50, 50)];
        let partition = mock_partition_result(vec![], vec![]); // Zero blocks
        let prob = detection_probability(&coclusters, &partition, 1, 10, 10);
        assert_eq!(prob, 0.0, "Detection probability with zero blocks should be 0.0");
    }

    #[test]
    fn test_compute_min_s_k_t_k_with_empty_blocks() {
        let cocluster = mock_cocluster(100, 80);
        let m_total = 1000;
        let n_total = 800;
        let t_m = 10;
        let t_n = 10;

        let s_k_empty = compute_min_s_k(&cocluster, &[], m_total, t_m);
        assert_eq!(s_k_empty, 0.0, "s_k with empty blocks should be 0.0");

        let t_k_empty = compute_min_t_k(&cocluster, &[], n_total, t_n);
        assert_eq!(t_k_empty, 0.0, "t_k with empty blocks should be 0.0");
    }

     #[test]
    fn test_compute_min_s_k_t_k_with_zero_block_sizes() {
        let cocluster = mock_cocluster(100, 80);
        let m_total = 1000;
        let n_total = 800;
        let t_m = 10;
        let t_n = 10;

        let s_k_zero_block = compute_min_s_k(&cocluster, &[0, 50, 0], m_total, t_m);
         // (100/1000) - (9/50) = 0.1 - 0.18 = -0.08 -> 0
        assert!((s_k_zero_block - 0.0).abs() < 1e-9, "s_k with zero block size error, got {}", s_k_zero_block);

        let t_k_zero_block = compute_min_t_k(&cocluster, &[0, 0, 40], n_total, t_n);
        // (80/800) - (9/40) = 0.1 - 0.225 = -0.125 -> 0
        assert!((t_k_zero_block - 0.0).abs() < 1e-9, "t_k with zero block size error, got {}", t_k_zero_block);
    }
} 