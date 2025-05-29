use ndarray::Array2;
// use std::collections::HashMap; // HashMap is not used in the new code
use std::error::Error;

// Structs are now defined here as per the user's complete implementation
#[derive(Debug, Clone)]
pub struct CoCluster {
    pub row_indices: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub row_size: usize,
    pub col_size: usize,
}

#[derive(Debug)] // Removed Clone as it's not strictly needed by partitioner and probability can be recomputed
pub struct PartitionResult {
    pub block_rows: Vec<usize>,
    pub block_cols: Vec<usize>,
    pub iterations: usize,
    pub probability: f64, // Added probability field
}

// Import functions from the probability module
// Assuming probability.rs is in the same crate src directory
use crate::probability;

#[derive(Clone)]
pub struct ProbabilisticPartitioner {
    pub t_m: usize,     // Minimum row threshold
    pub t_n: usize,     // Minimum column threshold
    pub t_max: usize,   // Maximum iterations
    pub p_thresh: f64,  // Probability threshold
}

impl ProbabilisticPartitioner {
    pub fn new(t_m: usize, t_n: usize, t_max: usize, p_thresh: f64) -> Self {
        Self { t_m, t_n, t_max, p_thresh }
    }

    pub fn partition(&self, matrix: &Array2<f64>, coclusters: &[CoCluster]) 
        -> Result<PartitionResult, Box<dyn Error>> {
        let (m, n) = matrix.dim();
        
        if self.t_m == 0 || self.t_n == 0 {
            return Err(Box::from("t_m and t_n must be greater than 0."));
        }

        // Step 1: Initialize minimum blocks to avoid fragmentation
        let m_blocks = (m + self.t_m - 1) / self.t_m;
        let n_blocks = (n + self.t_n - 1) / self.t_n;

        if m_blocks == 0 || n_blocks == 0 {
            return Err(Box::from("Matrix dimensions are too small for the given thresholds, resulting in zero blocks."));
        }
        
        // Step 2: Uniform initialization
        let mut phi_i = self.initialize_uniform_blocks(m, m_blocks);
        let mut psi_j = self.initialize_uniform_blocks(n, n_blocks);
        
        // Step 3: Compute detectability parameters
        let mut s_k = vec![0.0; coclusters.len()];
        let mut t_k = vec![0.0; coclusters.len()];
        
        if !coclusters.is_empty() {
            self.compute_detectability_params(coclusters, &phi_i, &psi_j, &mut s_k, &mut t_k, m, n);
        }
        
        // Step 4: Initial detection probability
        let mut t_p = 1;
        // Pass t_m and t_n to detection_probability
        let temp_partition_result_for_prob = PartitionResult { 
            block_rows: phi_i.clone(), block_cols: psi_j.clone(), iterations: 0, probability: 0.0 
        };
        let mut prob = probability::detection_probability(
            coclusters, 
            &temp_partition_result_for_prob, 
            t_p, 
            self.t_m, 
            self.t_n
        );
        
        // Step 5: Main optimization loop
        while prob < self.p_thresh && t_p <= self.t_max {
            let mut adjusted = false;
            
            if coclusters.is_empty() { // No adjustments if no coclusters
                break;
            }

            for (k_idx, cocluster) in coclusters.iter().enumerate() {
                if cocluster.row_size >= self.t_m && cocluster.col_size >= self.t_n {
                    let overlapping_blocks = self.find_overlapping_blocks(cocluster, &phi_i, &psi_j);
                    
                    for (i_block_idx, j_block_idx) in overlapping_blocks {
                        // Ensure s_k[k_idx] and t_k[k_idx] are not zero to avoid division by zero or excessively large new_phi/new_psi
                        let s_k_val = s_k[k_idx];
                        let t_k_val = t_k[k_idx];

                        if s_k_val <= 0.0 || t_k_val <= 0.0 { continue; }
                        if m == 0 || n == 0 { continue; }

                        let new_phi_val = ((cocluster.row_size as f64 * self.t_m as f64) /
                                           (m as f64 * s_k_val)).ceil() as usize;
                        let new_psi_val = ((cocluster.col_size as f64 * self.t_n as f64) /
                                           (n as f64 * t_k_val)).ceil() as usize;
                        
                        if i_block_idx < phi_i.len() && new_phi_val > phi_i[i_block_idx] {
                            phi_i[i_block_idx] = new_phi_val.min(m); // Cap at total matrix dimension
                            adjusted = true;
                        }
                        if j_block_idx < psi_j.len() && new_psi_val > psi_j[j_block_idx] {
                            psi_j[j_block_idx] = new_psi_val.min(n); // Cap at total matrix dimension
                            adjusted = true;
                        }
                    }
                }
            }
            
            if adjusted {
                self.redistribute_blocks(&mut phi_i, m);
                self.redistribute_blocks(&mut psi_j, n);
                self.compute_detectability_params(coclusters, &phi_i, &psi_j, &mut s_k, &mut t_k, m, n);
            }
            
            let current_partition_for_prob = PartitionResult { 
                block_rows: phi_i.clone(), block_cols: psi_j.clone(), iterations: t_p, probability: prob 
            };
            prob = probability::detection_probability(coclusters, &current_partition_for_prob, t_p, self.t_m, self.t_n);
            
            if prob < self.p_thresh && t_p < self.t_max { // Avoid incrementing t_p if it's already t_max
                let delta_t = self.compute_delta_t(&s_k, &t_k, &phi_i, &psi_j, prob);
                t_p = (t_p + delta_t.max(1)).min(self.t_max); // Ensure t_p increases by at least 1, unless at max
            } else if prob >= self.p_thresh {
                break; // Exit loop if probability threshold is met
            } else { // t_p >= self.t_max and prob < self.p_thresh
                 break; // Exit loop if max iterations reached
            }
        }
        
        Ok(PartitionResult {
            block_rows: phi_i,
            block_cols: psi_j,
            iterations: t_p,
            probability: prob,
        })
    }
    
    fn initialize_uniform_blocks(&self, total_size: usize, num_blocks: usize) -> Vec<usize> {
        if num_blocks == 0 { return Vec::new(); }
        let base_size = total_size / num_blocks;
        let remainder = total_size % num_blocks;
        
        let mut blocks = vec![base_size; num_blocks];
        for i in 0..remainder {
            blocks[i] += 1;
        }
        blocks
    }
    
    fn compute_detectability_params(&self, coclusters: &[CoCluster], phi_i: &[usize], 
                                   psi_j: &[usize], s_k: &mut [f64], t_k: &mut [f64],
                                   m_total_dim: usize, n_total_dim: usize) {
        // m_total_dim and n_total_dim are the overall matrix dimensions, not sum of current blocks if redistributed
        // The paper implies m and n in s_i_k formula are total dimensions of the original matrix.
        
        for (k, cocluster) in coclusters.iter().enumerate() {
            let mut min_s = f64::INFINITY;
            let mut min_t = f64::INFINITY;
            
            if m_total_dim == 0 || n_total_dim == 0 { 
                s_k[k] = 0.0;
                t_k[k] = 0.0;
                continue;
            }

            for &phi_val in phi_i.iter() {
                if phi_val == 0 { continue; } // Avoid division by zero
                let s_i_k_val = (cocluster.row_size as f64 / m_total_dim as f64) - 
                               ((self.t_m.saturating_sub(1)) as f64 / phi_val as f64);
                min_s = min_s.min(s_i_k_val);
            }
            
            for &psi_val in psi_j.iter() {
                if psi_val == 0 { continue; } // Avoid division by zero
                let t_j_k_val = (cocluster.col_size as f64 / n_total_dim as f64) - 
                               ((self.t_n.saturating_sub(1)) as f64 / psi_val as f64);
                min_t = min_t.min(t_j_k_val);
            }
            
            s_k[k] = if min_s == f64::INFINITY { 0.0 } else { min_s.max(0.0) };
            t_k[k] = if min_t == f64::INFINITY { 0.0 } else { min_t.max(0.0) };
        }
    }
        
    fn find_overlapping_blocks(&self, cocluster: &CoCluster, phi_i: &[usize], 
                              psi_j: &[usize]) -> Vec<(usize, usize)> {
        let mut overlapping = Vec::new();
        let mut current_row_offset = 0;
        
        let mut overlapping_row_block_indices = Vec::new();
        for (i, &block_row_size) in phi_i.iter().enumerate() {
            let block_row_end = current_row_offset + block_row_size;
            // Check if any row index of the cocluster falls within this block
            if cocluster.row_indices.iter().any(|&r_idx| r_idx >= current_row_offset && r_idx < block_row_end) {
                overlapping_row_block_indices.push(i);
            }
            current_row_offset = block_row_end;
        }
        
        let mut current_col_offset = 0;
        let mut overlapping_col_block_indices = Vec::new();
        for (j, &block_col_size) in psi_j.iter().enumerate() {
            let block_col_end = current_col_offset + block_col_size;
            // Check if any col index of the cocluster falls within this block
            if cocluster.col_indices.iter().any(|&c_idx| c_idx >= current_col_offset && c_idx < block_col_end) {
                overlapping_col_block_indices.push(j);
            }
            current_col_offset = block_col_end;
        }
        
        for &r_block_idx in &overlapping_row_block_indices {
            for &c_block_idx in &overlapping_col_block_indices {
                overlapping.push((r_block_idx, c_block_idx));
            }
        }
        
        overlapping
    }
    
    fn redistribute_blocks(&self, blocks: &mut Vec<usize>, total_size: usize) {
        if blocks.is_empty() { return; }
        let current_sum: usize = blocks.iter().sum();
        
        if current_sum == 0 && total_size == 0 { return; } // Nothing to do
        if current_sum == 0 && total_size > 0 { // Distribute total_size uniformly if current_sum is 0
            let base = total_size / blocks.len();
            let rem = total_size % blocks.len();
            for i in 0..blocks.len() {
                blocks[i] = base + if i < rem { 1 } else { 0 };
            }
            return;
        }

        if current_sum != total_size {
            let scale_factor = total_size as f64 / current_sum as f64;
            let mut new_calculated_sum = 0;
            
            for i in 0..blocks.len() {
                blocks[i] = (blocks[i] as f64 * scale_factor).round() as usize;
                // Ensure block size is at least 1 if total_size > 0, can be complex if total_size < blocks.len()
                // For simplicity, this might make blocks zero. A better strategy might be needed.
                // If we enforce minimum block size, it should be handled here.
                // blocks[i] = blocks[i].max(1); // Example: ensure min size of 1, but this can violate total_size
            }

            // Adjust sum due to rounding
            let mut final_sum_after_scaling: usize = blocks.iter().sum();
            let mut diff = total_size as i64 - final_sum_after_scaling as i64;

            // Distribute difference prioritizing non-zero blocks or proportionally
            // This is a simple iterative adjustment for the difference.
            let mut iter = 0;
            while diff != 0 && iter < blocks.len() * 2 { // Limit iterations to avoid infinite loops
                for i in 0..blocks.len() {
                    if diff == 0 { break; }
                    if diff > 0 {
                        blocks[i] += 1;
                        diff -= 1;
                    } else { // diff < 0
                        if blocks[i] > 0 { // Prevent going below zero, ideally should be min_threshold (e.g. t_m)
                            blocks[i] -= 1;
                            diff += 1;
                        }
                    }
                }
                iter += 1;
            }
            // If diff is still not zero, it implies an issue, possibly due to min block size constraints (not fully implemented here)
            // For now, the last block takes the remainder to ensure sum is exact if possible under constraints.
            // This simplified redistribution might not perfectly match the paper's intent without further specific rules.
             final_sum_after_scaling = blocks.iter().sum();
             if final_sum_after_scaling != total_size && !blocks.is_empty() {
                let last_idx = blocks.len() -1;
                let remainder_to_add = total_size as i64 - final_sum_after_scaling as i64 + blocks[last_idx] as i64;
                if remainder_to_add >=0 {
                     blocks[last_idx] = remainder_to_add as usize;
                } else {
                    // This case means other blocks summed up to more than total_size, even after trying to reduce them.
                    // This implies a conflict with minimum block sizes or an issue in scaling.
                    // A more robust strategy would be needed here.
                }
             }
        }
    }
    
    fn compute_delta_t(&self, s_k: &[f64], t_k: &[f64], phi_i: &[usize], 
                      psi_j: &[usize], current_prob: f64) -> usize {
        if phi_i.is_empty() || psi_j.is_empty() { return 1; }

        let m_blocks_len = phi_i.len() as f64;
        let n_blocks_len = psi_j.len() as f64;
        let phi_avg = phi_i.iter().sum::<usize>() as f64 / m_blocks_len;
        let psi_avg = psi_j.iter().sum::<usize>() as f64 / n_blocks_len;
        
        let mut max_term: f64 = 0.0;
        if s_k.is_empty() { // If no coclusters, no term to maximize
            return 1; // Default increase if no info to base delta_t on
        }
        for (s_val, t_val) in s_k.iter().zip(t_k) {
            let term = phi_avg * m_blocks_len * s_val.powi(2) + psi_avg * n_blocks_len * t_val.powi(2);
            if term.is_finite() {
                max_term = max_term.max(term);
            }
        }
        
        if max_term > 1e-9 { // Check against a small epsilon to avoid division by zero or near-zero
            // Ensure arguments to ln are positive
            let target_ln_arg = 1.0 - self.p_thresh;
            let current_ln_arg = 1.0 - current_prob;

            if target_ln_arg <= 0.0 || current_ln_arg <= 0.0 {
                return 1; // Cannot compute delta_t if probabilities are 1.0 or invalid
            }

            let required_exp_log = target_ln_arg.ln();
            let current_exp_log = current_ln_arg.ln();
            
            // Delta_t = (ln(1-P_thresh) - ln(1-P_current)) / (-2 * max_term_val)
            let delta = (required_exp_log - current_exp_log) / (-2.0 * max_term);
            if delta.is_finite() && delta > 0.0 {
                delta.ceil() as usize
            } else {
                1 // Default if calculation leads to non-positive or non-finite delta
            }
        } else {
            1 // Default increase if max_term is too small
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    // We will use CoCluster and PartitionResult defined in this file for partitioner tests
    fn create_dummy_matrix(rows: usize, cols: usize) -> Array2<f64> {
        Array2::zeros((rows, cols)) // Content doesn't matter for some partitioner logic
    }
    fn create_dummy_coclusters(num: usize, r_size: usize, c_size: usize, m_total: usize, n_total: usize) -> Vec<CoCluster> {
        let mut coclusters = Vec::new();
        for i in 0..num {
            let row_indices: Vec<usize> = (i * r_size .. (i * r_size + r_size).min(m_total)).collect();
            let col_indices: Vec<usize> = (i * c_size .. (i * c_size + c_size).min(n_total)).collect();
            coclusters.push(CoCluster {
                row_indices,
                col_indices,
                row_size: r_size,
                col_size: c_size,
            });
        }
        coclusters
    }
    #[test]
    fn test_initialize_uniform_blocks() {
        let partitioner = ProbabilisticPartitioner::new(10, 10, 5, 0.9);
        let blocks1 = partitioner.initialize_uniform_blocks(100, 4);
        assert_eq!(blocks1, vec![25, 25, 25, 25]);
        assert_eq!(blocks1.iter().sum::<usize>(), 100);
        let blocks2 = partitioner.initialize_uniform_blocks(103, 4);
        assert_eq!(blocks2, vec![26, 26, 26, 25]);
        assert_eq!(blocks2.iter().sum::<usize>(), 103);
        let blocks3 = partitioner.initialize_uniform_blocks(10, 3);
        assert_eq!(blocks3, vec![4, 3, 3]);
        assert_eq!(blocks3.iter().sum::<usize>(), 10);
        let blocks_empty = partitioner.initialize_uniform_blocks(100, 0);
        assert!(blocks_empty.is_empty());
    }
    #[test]
    fn test_compute_detectability_params() {
        let partitioner = ProbabilisticPartitioner::new(5, 5, 5, 0.9);
        let coclusters = vec![
            CoCluster { row_indices: (0..20).collect(), col_indices: (0..20).collect(), row_size: 20, col_size: 20 },
            CoCluster { row_indices: (50..70).collect(), col_indices: (50..70).collect(), row_size: 20, col_size: 20 },
        ];
        let phi_i = vec![50, 50]; // Total rows = 100
        let psi_j = vec![50, 50]; // Total cols = 100
        let mut s_k = vec![0.0; 2];
        let mut t_k = vec![0.0; 2];
        partitioner.compute_detectability_params(&coclusters, &phi_i, &psi_j, &mut s_k, &mut t_k, 100, 100);
        // For cocluster 0 (20x20) in 100x100 matrix, t_m=5, phi_i=50:
        // s_val = (20/100) - (4/50) = 0.2 - 0.08 = 0.12
        assert!((s_k[0] - 0.12).abs() < 1e-9, "s_k[0] expected 0.12, got {}", s_k[0]);
        assert!((t_k[0] - 0.12).abs() < 1e-9, "t_k[0] expected 0.12, got {}", t_k[0]);
    }
    #[test]
    fn test_redistribute_blocks() {
        let partitioner = ProbabilisticPartitioner::new(10, 10, 5, 0.9);
        let mut blocks1 = vec![20, 30, 60]; // sum 110
        partitioner.redistribute_blocks(&mut blocks1, 100); // target 100
        assert_eq!(blocks1.iter().sum::<usize>(), 100, "Redistribute 1 failed sum check");
        // Check individual elements if specific rounding is expected, e.g., roughly proportional scaling
        // Expected scaling: 20*100/110=18.18->18, 30*100/110=27.27->27, 60*100/110=54.54->55. Sum=100
        // The current implementation might give slightly different rounding due to iterative adjustment.
        // For example output of current code: [18, 27, 55] or similar.
        // This test mainly checks if the sum is correct after redistribution.
        let mut blocks2 = vec![10, 10, 10]; // sum 30
        partitioner.redistribute_blocks(&mut blocks2, 60); // target 60
        assert_eq!(blocks2.iter().sum::<usize>(), 60, "Redistribute 2 failed sum check");
        // Expected: [20, 20, 20]
        assert_eq!(blocks2, vec![20,20,20], "Redistribute 2 failed element check");
        let mut blocks3 = vec![10, 15, 25]; // sum 50
        partitioner.redistribute_blocks(&mut blocks3, 50); // target 50 (no change)
        assert_eq!(blocks3.iter().sum::<usize>(), 50, "Redistribute 3 failed sum check");
        assert_eq!(blocks3, vec![10, 15, 25], "Redistribute 3 failed element check (no change)");
        let mut blocks_empty: Vec<usize> = vec![];
        partitioner.redistribute_blocks(&mut blocks_empty, 100);
        assert!(blocks_empty.is_empty(), "Redistribute empty blocks failed");
        let mut blocks_to_zero = vec![10,20];
        partitioner.redistribute_blocks(&mut blocks_to_zero, 0);
        assert_eq!(blocks_to_zero.iter().sum::<usize>(), 0, "Redistribute to zero sum failed");
        // Could be [0,0]
    }
    #[test]
    fn test_compute_delta_t() {
        let partitioner = ProbabilisticPartitioner::new(10, 10, 20, 0.95);
        let s_k = vec![0.1, 0.12];
        let t_k = vec![0.1, 0.12];
        let phi_i = vec![100,100];
        let psi_j = vec![100,100];
        let current_prob = 0.8;
        let delta_t = partitioner.compute_delta_t(&s_k, &t_k, &phi_i, &psi_j, current_prob);
        // This is a smoke test, exact value depends on complex formula involving logs
        // We expect delta_t to be a positive integer
        assert!(delta_t >= 1, "delta_t should be at least 1, got {}", delta_t);
        let current_prob_high = 0.99; // Already above threshold
        let delta_t_high_prob = partitioner.compute_delta_t(&s_k, &t_k, &phi_i, &psi_j, current_prob_high);
        // If current_prob >= p_thresh, ln(1-current_prob) might be ln of negative or zero.
        // Or required_exp_log - current_exp_log becomes positive or zero.
        // The formula is (ln(1-P_thresh) - ln(1-P_current)) / (-2 * max_term_val)
        // If 1-P_current < 1-P_thresh (i.e. P_current > P_thresh), numerator is negative.
        // Combined with -2*max_term in denominator, delta can be positive. 
        // However, the loop for t_p usually stops if prob >= p_thresh.
        // Test if it returns 1 when prob is high.
        assert!(delta_t_high_prob >= 1, "delta_t for high prob should be >=1, got {}", delta_t_high_prob);
        let s_k_zero = vec![0.0, 0.0];
        let t_k_zero = vec![0.0, 0.0];
        let delta_t_zero_st = partitioner.compute_delta_t(&s_k_zero, &t_k_zero, &phi_i, &psi_j, current_prob);
        assert_eq!(delta_t_zero_st, 1, "delta_t with zero s_k, t_k should be 1");
    }
    #[test]
    fn test_partitioner_basic_run() {
        let matrix = create_dummy_matrix(200, 200);
        // Create one large cocluster for simplicity in this basic test
        let coclusters = vec![CoCluster { 
            row_indices: (0..100).collect(), col_indices: (0..100).collect(), row_size: 100, col_size: 100 
        }];
        let partitioner = ProbabilisticPartitioner::new(10, 10, 5, 0.90); // t_max=5, p_thresh=0.90
        let result = partitioner.partition(&matrix, &coclusters);
        assert!(result.is_ok(), "Partitioning failed: {:?}", result.err());
        let partition_result = result.unwrap();
        assert!(partition_result.probability >= 0.0, "Result probability is invalid: {}", partition_result.probability);
        assert!(!partition_result.block_rows.is_empty());
        assert!(!partition_result.block_cols.is_empty());
        assert_eq!(partition_result.block_rows.iter().sum::<usize>(), 200);
        assert_eq!(partition_result.block_cols.iter().sum::<usize>(), 200);
        assert!(partition_result.iterations > 0 && partition_result.iterations <= 5); // iterations should be within [1, t_max]
    }
    #[test]
    fn test_partitioner_no_coclusters() {
        let matrix = create_dummy_matrix(100,100);
        let coclusters: Vec<CoCluster> = vec![];
        let partitioner = ProbabilisticPartitioner::new(10,10,5,0.9);
        let result = partitioner.partition(&matrix, &coclusters).unwrap();
        // Expect default partitioning, prob might be 1.0 or some other default for no coclusters
        // The current main loop for t_p might not run if coclusters is empty.
        // The compute_detectability_params won't run. detection_probability with empty coclusters returns 1.0.
        // So prob will be 1.0, loop `while prob < self.p_thresh` won't run, t_p = 1.
        assert_eq!(result.iterations, 1); 
        assert!((result.probability - 1.0).abs() < 1e-9); // Probability of detecting zero things is 1.
        assert_eq!(result.block_rows.iter().sum::<usize>(), 100);
        assert_eq!(result.block_cols.iter().sum::<usize>(), 100);
    }
    #[test]
    fn test_partitioner_small_matrix_error() {
        let matrix_too_small = create_dummy_matrix(5, 5);
        let coclusters = create_dummy_coclusters(1, 2, 2, 5, 5);
        // t_m=10, t_n=10 means m_blocks or n_blocks will be 0 for 5x5 matrix.
        let partitioner = ProbabilisticPartitioner::new(10, 10, 5, 0.9); 
        let result = partitioner.partition(&matrix_too_small, &coclusters);
        assert!(result.is_err(), "Partitioning should fail for matrix smaller than t_m/t_n thresholds leading to 0 blocks");
    }
    #[test]
    fn test_find_overlapping_blocks() { 
        let partitioner = ProbabilisticPartitioner::new(10,10,5,0.9); 
        let phi_i = vec![20, 30, 50]; // Row blocks: [0-19], [20-49], [50-99]
        let psi_j = vec![25, 25, 25, 25]; // Col blocks: [0-24], [25-49], [50-74], [75-99]
        // CoCluster 1: overlaps block (0,0) and (0,1)
        let cc1 = CoCluster { 
            row_indices: vec![5, 15], col_indices: vec![10, 30], row_size: 2, col_size: 2 
        };
        let overlaps1 = partitioner.find_overlapping_blocks(&cc1, &phi_i, &psi_j);
        assert_eq!(overlaps1.len(), 2);
        assert!(overlaps1.contains(&(0,0)));
        assert!(overlaps1.contains(&(0,1)));
        // CoCluster 2: overlaps block (1,2)
        let cc2 = CoCluster { 
            row_indices: vec![25, 40], col_indices: vec![60], row_size: 2, col_size: 1
        };
        let overlaps2 = partitioner.find_overlapping_blocks(&cc2, &phi_i, &psi_j);
        assert_eq!(overlaps2.len(), 1);
        assert!(overlaps2.contains(&(1,2)));
        // CoCluster 3: no overlap (indices outside defined blocks, though block sums to 100x100 matrix)
        // This test assumes cocluster indices are within matrix dimension which phi_i/psi_j cover.
        // find_overlapping_blocks itself doesn't check matrix bounds, just block coverage.
        let cc3 = CoCluster { 
             row_indices: vec![100, 101], col_indices: vec![100, 101], row_size: 2, col_size: 2
        };
        let overlaps3 = partitioner.find_overlapping_blocks(&cc3, &phi_i, &psi_j);
        assert!(overlaps3.is_empty());
        // CoCluster 4: overlaps multiple row and column blocks
        let cc4 = CoCluster {
            row_indices: vec![15, 25, 55], // overlaps row blocks 0, 1, 2
            col_indices: vec![20, 40, 70, 90], // overlaps col blocks 0, 1, 2, 3
            row_size: 3, col_size: 4
        };
        let overlaps4 = partitioner.find_overlapping_blocks(&cc4, &phi_i, &psi_j);
        assert_eq!(overlaps4.len(), 3 * 4); // 3 row blocks * 4 col blocks = 12 pairs
        assert!(overlaps4.contains(&(0,0)));
        assert!(overlaps4.contains(&(1,1)));
        assert!(overlaps4.contains(&(2,3)));
    }
} 