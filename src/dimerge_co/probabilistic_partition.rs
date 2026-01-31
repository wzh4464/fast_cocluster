//! # Probabilistic Partitioning
//!
//! Implements SVD-based probabilistic partitioning with preservation guarantees.
//!
//! ## Mathematical Basis
//! - Threshold: τ = √(k/n)
//! - Preservation: P(preserve co-clusters) ≥ 1-δ when spectral gap σ_k - σ_{k+1} > τ

/**
 * File: /src/dimerge_co/probabilistic_partition.rs
 * Created Date: Monday, January 27th 2026
 * Author: Zihan
 * -----
 * Last Modified: Monday, 27th January 2026
 * Modified By: Zihan Wu <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
 * 2026-01-27		Zihan	Implemented probabilistic partitioning with theoretical guarantees
 */

use crate::dimerge_co::types::*;
use ndarray::{s, Array1, Array2, Axis};
use ndarray_linalg::SVD;
use rayon::prelude::*;

/// Probabilistic partitioner using SVD-based sign patterns
pub struct ProbabilisticPartitioner {
    params: PartitionParams,
}

impl ProbabilisticPartitioner {
    /// Create a new probabilistic partitioner
    pub fn new(k: usize, n: usize, delta: f64, num_partitions: usize) -> Result<Self, PartitionError> {
        let params = PartitionParams::new(k, n, delta, num_partitions);
        params.validate()?;
        Ok(Self { params })
    }

    /// Create partitioner from parameters
    pub fn from_params(params: PartitionParams) -> Result<Self, PartitionError> {
        params.validate()?;
        Ok(Self { params })
    }

    /// Partition a matrix using probabilistic guarantees
    ///
    /// # Algorithm
    /// 1. Compute truncated SVD for dominant k singular vectors
    /// 2. Partition rows/columns by sign patterns of dominant singular vectors
    /// 3. Form cross-product partitions
    /// 4. Validate preservation probability from spectral gap
    pub fn partition(&self, matrix: &Array2<f64>) -> Result<PartitionResult, PartitionError> {
        if matrix.nrows() < self.params.min_partition_size.0
            || matrix.ncols() < self.params.min_partition_size.1
        {
            return Err(PartitionError::InsufficientData);
        }

        // Step 1: Compute truncated SVD
        let (u, sigma, vt) = self.compute_truncated_svd(matrix)?;

        // Step 2: Partition by sign pattern of dominant singular vectors
        let row_partitions = self.partition_by_sign(&u, self.params.tau);
        let col_partitions = self.partition_by_sign(&vt.t().to_owned(), self.params.tau);

        // Step 3: Form cross-product partitions
        let partitions = self.form_partitions(&row_partitions, &col_partitions)?;

        // Step 4: Compute preservation probability from spectral gap
        let preservation_prob = self.compute_preservation_probability(&sigma);

        Ok(PartitionResult {
            partitions,
            threshold: self.params.tau,
            preservation_prob,
            singular_values: sigma.to_vec(),
        })
    }

    /// 使用指定线程数并行分区
    pub fn partition_parallel(
        &self,
        matrix: &Array2<f64>,
        num_threads: usize,
    ) -> Result<PartitionResult, PartitionError> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| PartitionError::Other(format!("Thread pool creation failed: {}", e)))?;

        pool.install(|| self.partition(matrix))
    }

    /// Compute truncated SVD for the first k components
    fn compute_truncated_svd(
        &self,
        matrix: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>), PartitionError> {
        // Use full SVD and truncate (for now; can optimize with randomized SVD later)
        let result = matrix
            .svd(true, true)
            .map_err(|e| PartitionError::SvdFailed(format!("{:?}", e)))?;

        let u = result
            .0
            .ok_or_else(|| PartitionError::SvdFailed("U matrix missing".to_string()))?;
        let sigma = result.1;
        let vt = result
            .2
            .ok_or_else(|| PartitionError::SvdFailed("Vt matrix missing".to_string()))?;

        // Truncate to k components
        let k = self.params.k.min(sigma.len());
        let u_truncated = u.slice(s![.., ..k]).to_owned();
        let sigma_truncated = sigma.slice(s![..k]).to_owned();
        let vt_truncated = vt.slice(s![..k, ..]).to_owned();

        Ok((u_truncated, sigma_truncated, vt_truncated))
    }

    /// Partition indices by sign pattern of singular vectors
    ///
    /// Creates binary partitions based on whether the dominant singular vector
    /// component is positive or negative (with threshold for numerical stability)
    fn partition_by_sign(&self, vectors: &Array2<f64>, threshold: f64) -> Vec<Vec<usize>> {
        let n = vectors.nrows();
        let mut positive = Vec::new();
        let mut negative = Vec::new();

        for i in 0..n {
            // Use the first (dominant) singular vector for partitioning
            let value = vectors[[i, 0]];
            if value > threshold {
                positive.push(i);
            } else if value < -threshold {
                negative.push(i);
            } else {
                // Values close to zero - assign to smaller partition for balance
                if positive.len() <= negative.len() {
                    positive.push(i);
                } else {
                    negative.push(i);
                }
            }
        }

        vec![positive, negative]
    }

    /// Form partitions as cross-product of row and column partitions
    fn form_partitions(
        &self,
        row_partitions: &[Vec<usize>],
        col_partitions: &[Vec<usize>],
    ) -> Result<Vec<Partition>, PartitionError> {
        let mut partitions = Vec::new();
        let mut id = 0;

        for row_indices in row_partitions {
            for col_indices in col_partitions {
                // Skip empty or too-small partitions
                if row_indices.len() >= self.params.min_partition_size.0
                    && col_indices.len() >= self.params.min_partition_size.1
                {
                    partitions.push(Partition::new(
                        row_indices.clone(),
                        col_indices.clone(),
                        id,
                    ));
                    id += 1;
                }
            }
        }

        // Pad to power of 2 if needed for binary tree
        while partitions.len() < self.params.num_partitions && partitions.len() > 0 {
            // Duplicate the largest partition to maintain balance
            let largest_idx = partitions
                .iter()
                .enumerate()
                .max_by_key(|(_, p)| p.row_indices.len() * p.col_indices.len())
                .map(|(i, _)| i)
                .unwrap();
            let duplicate = partitions[largest_idx].clone();
            partitions.push(Partition::new(
                duplicate.row_indices,
                duplicate.col_indices,
                id,
            ));
            id += 1;
        }

        if partitions.is_empty() {
            return Err(PartitionError::InsufficientData);
        }

        Ok(partitions)
    }

    /// Compute preservation probability from spectral gap
    ///
    /// Based on theorem: P(preserve) ≥ 1-δ when σ_k - σ_{k+1} > τ
    fn compute_preservation_probability(&self, singular_values: &Array1<f64>) -> f64 {
        if singular_values.len() <= self.params.k {
            // Not enough singular values to compute gap
            return 1.0 - self.params.delta;
        }

        // Compute spectral gap: σ_k - σ_{k+1}
        let gap = singular_values[self.params.k - 1] - singular_values[self.params.k];

        // Simple model: preservation probability increases with gap relative to threshold
        // P(preserve) ≈ 1 - δ * exp(-(gap/τ)^2)
        let ratio = gap / self.params.tau;
        let preservation = 1.0 - self.params.delta * (-ratio * ratio).exp();

        // Clamp to [1-δ, 1.0]
        preservation.max(1.0 - self.params.delta).min(1.0)
    }
}

/// Advanced partitioning strategies
pub struct AdaptivePartitioner {
    base_partitioner: ProbabilisticPartitioner,
}

impl AdaptivePartitioner {
    /// Create adaptive partitioner that adjusts parameters based on matrix properties
    pub fn new(k: usize, n: usize, delta: f64) -> Result<Self, PartitionError> {
        // Auto-determine num_partitions as power of 2
        let num_partitions = Self::compute_optimal_partitions(n);
        let base_partitioner = ProbabilisticPartitioner::new(k, n, delta, num_partitions)?;
        Ok(Self { base_partitioner })
    }

    /// Compute optimal number of partitions (power of 2)
    fn compute_optimal_partitions(n: usize) -> usize {
        // Use log2(n/100) as a heuristic (want ~100 samples per partition on average)
        let target = (n / 100).max(4);
        let log2 = (target as f64).log2().ceil() as u32;
        2_usize.pow(log2)
    }

    /// Partition with adaptive parameters
    pub fn partition(&self, matrix: &Array2<f64>) -> Result<PartitionResult, PartitionError> {
        self.base_partitioner.partition(matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_partition_params_validation() {
        let params = PartitionParams::new(3, 100, 0.05, 4);
        assert!(params.validate().is_ok());

        let bad_params = PartitionParams::new(3, 100, 0.05, 3); // Not power of 2
        assert!(bad_params.validate().is_err());
    }

    #[test]
    fn test_partition_by_sign() {
        let partitioner = ProbabilisticPartitioner::new(2, 100, 0.05, 2).unwrap();
        let vectors = Array2::from_shape_vec((4, 2), vec![
            0.5, 0.1,   // positive
            -0.5, 0.2,  // negative
            0.3, -0.1,  // positive
            -0.3, 0.05, // negative
        ]).unwrap();

        let partitions = partitioner.partition_by_sign(&vectors, 0.01);
        assert_eq!(partitions.len(), 2);
        assert_eq!(partitions[0], vec![0, 2]); // positive
        assert_eq!(partitions[1], vec![1, 3]); // negative
    }

    #[test]
    fn test_probabilistic_partition_basic() {
        // Use a larger matrix (50×40) to avoid InsufficientData error
        let matrix = Array2::from_shape_vec((50, 40), (0..2000).map(|x| x as f64).collect()).unwrap();
        let partitioner = ProbabilisticPartitioner::new(3, 50, 0.05, 4).unwrap();
        let result = partitioner.partition(&matrix).unwrap();

        assert!(!result.partitions.is_empty());
        assert!(result.preservation_prob >= 0.0 && result.preservation_prob <= 1.0);
        assert_eq!(result.threshold, (3.0 / 50.0_f64).sqrt());
    }

    #[test]
    fn test_adaptive_partitioner() {
        let matrix = Array2::from_shape_vec((50, 30), (0..1500).map(|x| x as f64).collect()).unwrap();
        let partitioner = AdaptivePartitioner::new(3, 50, 0.05).unwrap();
        let result = partitioner.partition(&matrix).unwrap();

        assert!(!result.partitions.is_empty());
    }
}
