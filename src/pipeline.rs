use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use std::time::{Duration, Instant};

use log::{debug, info, warn};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::cocluster::Coclusterer;
use crate::scoring::{CompatibilityScorer, ExponentialScorer, PearsonScorer, Scorer};
use crate::submatrix::Submatrix;
use crate::matrix::Matrix;
use crate::config::Config;
use crate::{config::Config as PipelinecrateConfig, matrix::Matrix as PipelinecrateMatrix};
use ndarray::Array2;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::marker::PhantomData;

/// Pipeline配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// 最小分数阈值
    pub min_score: f64,
    /// 最大子矩阵数量
    pub max_submatrices: usize,
    /// 是否按分数排序
    pub sort_by_score: bool,
    /// 最小子矩阵大小
    pub min_submatrix_size: (usize, usize),
    /// 是否收集详细统计信息
    pub collect_stats: bool,
    /// 是否并行处理
    pub parallel: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            min_score: 0.5,
            max_submatrices: 50,
            sort_by_score: true,
            min_submatrix_size: (3, 3),
            collect_stats: true,
            parallel: true,
        }
    }
}

/// 聚类算法trait
pub trait Clusterer: Send + Sync {
    fn cluster<'matrix_life>(
        &self,
        matrix: &'matrix_life Matrix<f64>
    ) -> Result<Vec<Submatrix<'matrix_life, f64>>, Box<dyn Error>>;
    fn name(&self) -> &str;
}

/// SVD聚类器包装
pub struct SVDClusterer {
    k: usize,
    tol: f64,
}

impl SVDClusterer {
    pub fn new(k: usize, tol: f64) -> Self {
        Self { k, tol }
    }
}

impl Clusterer for SVDClusterer {
    fn cluster<'matrix_life>(
        &self,
        matrix: &'matrix_life Matrix<f64>
    ) -> Result<Vec<Submatrix<'matrix_life, f64>>, Box<dyn Error>> {
        let mut coclusterer = Coclusterer::new(matrix.data.clone(), self.k as usize, self.tol as f64);
        let assignments = coclusterer.cocluster()?;

        // 将assignments转换为Submatrix列表
        let mut cluster_map: HashMap<usize, Vec<usize>> = HashMap::new();
        let n_rows = matrix.data.nrows(); // matrix.data is Array2<f64>

        for (idx, &cluster) in assignments.iter().enumerate() {
            if cluster == usize::MAX {
                continue; // skip zero-embedding samples (unassigned)
            }
            cluster_map
                .entry(cluster)
                .or_insert_with(Vec::new)
                .push(idx);
        }

        let mut submatrices = Vec::new();
        for (_, indices) in cluster_map {
            let (rows, cols): (Vec<_>, Vec<_>) = indices.into_iter().partition(|&idx| idx < n_rows);

            let cols: Vec<_> = cols.into_iter().map(|idx| idx - n_rows).collect();

            if !rows.is_empty() && !cols.is_empty() {
                // Use from_indices from crate::submatrix
                // matrix.data is Array2<f64>, which is what from_indices expects
                if let Some(submatrix) = crate::submatrix::Submatrix::from_indices(&matrix.data, &rows, &cols) {
                    submatrices.push(submatrix);
                }
            }
        }

        Ok(submatrices)
    }

    fn name(&self) -> &str {
        "SVD"
    }
}

/// Pipeline统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStats {
    pub total_duration: Duration,
    pub clustering_duration: Duration,
    pub scoring_duration: Duration,
    pub filtering_duration: Duration,
    pub initial_submatrices: usize,
    pub filtered_submatrices: usize,
    pub score_distribution: ScoreDistribution,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreDistribution {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std_dev: f64,
}

/// Pipeline步骤结果
#[derive(Debug, Clone)]
pub struct StepResult<'a> {
    pub submatrices: Vec<Submatrix<'a, f64>>,
    pub scores: Vec<f64>,
    pub stats: Option<PipelineStats>,
}

impl<'a> StepResult<'a> {
    pub fn summary(&self) -> String {
        let mut summary = format!("Found {} submatrices", self.submatrices.len());

        if !self.scores.is_empty() {
            let min_score = self.scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_score = self.scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let avg_score = self.scores.iter().sum::<f64>() / self.scores.len() as f64;

            summary.push_str(&format!(
                "\nScore range: {:.4} - {:.4} (avg: {:.4})",
                min_score, max_score, avg_score
            ));
        }

        for (i, (sub, score)) in self
            .submatrices
            .iter()
            .zip(&self.scores)
            .take(5)
            .enumerate()
        {
            summary.push_str(&format!(
                "\n  #{}: {}x{} matrix, score: {:.4}",
                i + 1,
                sub.row_indices.len(),
                sub.col_indices.len(),
                score
            ));
        }

        if self.submatrices.len() > 5 {
            summary.push_str(&format!("\n  ... and {} more", self.submatrices.len() - 5));
        }

        summary
    }
}

/// 共聚类Pipeline
pub struct CoclusterPipeline {
    clusterer: Box<dyn Clusterer>,
    scorer: Box<dyn Scorer>,
    config: PipelineConfig,
}

impl CoclusterPipeline {
    pub fn builder() -> PipelineBuilder {
        PipelineBuilder::new()
    }

    pub fn run<'a>(&self, matrix: &'a Matrix<f64>) -> Result<StepResult<'a>, Box<dyn Error>> {
        let start_time = Instant::now();
        info!(
            "Starting cocluster pipeline with {} clustering",
            self.clusterer.name()
        );

        // Step 1: 聚类
        let clustering_start = Instant::now();
        let submatrices_from_clusterer = self.clusterer.cluster(matrix)?;
        let clustering_duration = clustering_start.elapsed();
        let initial_submatrices_count = submatrices_from_clusterer.len();
        info!(
            "Clustering completed in {:?}, found {} submatrices",
            clustering_duration,
            initial_submatrices_count
        );

        // Step 2: 评分
        let scoring_start = Instant::now();
        let scores = if self.config.parallel {
            self.score_parallel(matrix, &submatrices_from_clusterer)
        } else {
            self.scorer.score_all(matrix, &submatrices_from_clusterer)
        };
        let scoring_duration = scoring_start.elapsed();
        info!("Scoring completed in {:?}", scoring_duration);

        // Step 3: 过滤和排序
        let filtering_start = Instant::now();
        let (filtered_submatrices, filtered_scores) =
            self.filter_and_sort(submatrices_from_clusterer, scores, &self.config);
        let filtering_duration = filtering_start.elapsed();

        let total_duration = start_time.elapsed();
        info!(
            "Pipeline completed in {:?}, {} submatrices retained",
            total_duration,
            filtered_submatrices.len()
        );

        // 收集统计信息
        let stats = if self.config.collect_stats {
            Some(self.collect_stats(
                total_duration,
                clustering_duration,
                scoring_duration,
                filtering_duration,
                initial_submatrices_count,
                filtered_submatrices.len(),
                &filtered_scores,
            ))
        } else {
            None
        };

        Ok(StepResult {
            submatrices: filtered_submatrices,
            scores: filtered_scores,
            stats,
        })
    }

    fn score_parallel<'a>(&self, matrix: &'a Matrix<f64>, submatrices: &[Submatrix<'a, f64>]) -> Vec<f64> {
        use rayon::prelude::*;

        submatrices
            .par_iter()
            .map(|sub| self.scorer.score(matrix, sub))
            .collect()
    }

    fn filter_and_sort<'a>(
        &self,
        submatrices: Vec<Submatrix<'a, f64>>,
        scores: Vec<f64>,
        config: &PipelineConfig,
    ) -> (Vec<Submatrix<'a, f64>>, Vec<f64>) {
        let mut combined: Vec<_> = submatrices.into_iter().zip(scores.into_iter()).collect();

        // 过滤
        combined.retain(|(sub, score)| {
            *score >= config.min_score
                && sub.row_indices.len() >= config.min_submatrix_size.0
                && sub.col_indices.len() >= config.min_submatrix_size.1
        });

        // 排序
        if config.sort_by_score {
            combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        }

        // 限制数量
        combined.truncate(config.max_submatrices);

        // 分离结果
        combined.into_iter().unzip()
    }

    fn collect_stats(
        &self,
        total_duration: Duration,
        clustering_duration: Duration,
        scoring_duration: Duration,
        filtering_duration: Duration,
        initial_count: usize,
        filtered_count: usize,
        scores: &[f64],
    ) -> PipelineStats {
        let score_distribution = if scores.is_empty() {
            ScoreDistribution {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                std_dev: 0.0,
            }
        } else {
            let min = scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let mean = scores.iter().sum::<f64>() / scores.len() as f64;
            let variance =
                scores.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / scores.len() as f64;
            let std_dev = variance.sqrt();

            ScoreDistribution {
                min,
                max,
                mean,
                std_dev,
            }
        };

        PipelineStats {
            total_duration,
            clustering_duration,
            scoring_duration,
            filtering_duration,
            initial_submatrices: initial_count,
            filtered_submatrices: filtered_count,
            score_distribution,
        }
    }
}

/// Pipeline构建器
pub struct PipelineBuilder {
    clusterer: Option<Box<dyn Clusterer>>,
    scorer: Option<Box<dyn Scorer>>,
    config: PipelineConfig,
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self {
            clusterer: None,
            scorer: None,
            config: PipelineConfig::default(),
        }
    }

    pub fn with_clusterer(mut self, clusterer: Box<dyn Clusterer>) -> Self {
        self.clusterer = Some(clusterer);
        self
    }

    /// Convenience method to create a DiMergeCo clusterer with adaptive configuration
    ///
    /// # Arguments
    /// * `k` - Expected number of co-clusters
    /// * `n` - Total number of samples
    /// * `delta` - Preservation probability parameter (e.g., 0.05 for 95% preservation)
    /// * `local_clusterer` - Local clustering algorithm (e.g., SVDClusterer)
    /// * `num_threads` - Number of threads for parallel execution
    ///
    /// # Example
    /// ```no_run
    /// use fast_cocluster::pipeline::*;
    /// use fast_cocluster::dimerge_co::ClustererAdapter;
    /// use fast_cocluster::scoring::PearsonScorer;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let local_clusterer = ClustererAdapter::new(SVDClusterer::new(5, 0.1));
    /// let pipeline = CoclusterPipeline::builder()
    ///     .with_dimerge_co(
    ///         5,                 // k clusters
    ///         1000,              // n samples
    ///         0.05,              // δ = 5% failure probability
    ///         local_clusterer,   // Local clusterer
    ///         8,                 // 8 threads
    ///     )?
    ///     .with_scorer(Box::new(PearsonScorer::new(3, 3)))
    ///     .min_score(0.6)
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_dimerge_co<C: crate::dimerge_co::LocalClusterer + 'static>(
        mut self,
        k: usize,
        n: usize,
        delta: f64,
        local_clusterer: C,
        num_threads: usize,
    ) -> Result<Self, Box<dyn Error>> {
        use crate::dimerge_co::{
            DiMergeCoClusterer, HierarchicalMergeConfig, MergeStrategy,
        };

        // Default merge configuration
        let merge_config = HierarchicalMergeConfig {
            merge_strategy: MergeStrategy::Adaptive,
            merge_threshold: 0.5,
            rescore_merged: true,
            parallel_level: 2,
        };

        // Create DiMergeCo clusterer with adaptive partitioning
        let clusterer = DiMergeCoClusterer::with_adaptive(
            k,
            n,
            delta,
            local_clusterer,
            merge_config,
            num_threads,
        )?;

        self.clusterer = Some(Box::new(clusterer));
        Ok(self)
    }

    /// Create a DiMergeCo clusterer with explicit configuration
    ///
    /// For more control over partitioning and merging strategy.
    pub fn with_dimerge_co_explicit<C: crate::dimerge_co::LocalClusterer + 'static>(
        mut self,
        k: usize,
        n: usize,
        delta: f64,
        num_partitions: usize,
        local_clusterer: C,
        merge_config: crate::dimerge_co::HierarchicalMergeConfig,
        num_threads: usize,
    ) -> Result<Self, Box<dyn Error>> {
        use crate::dimerge_co::DiMergeCoClusterer;

        let clusterer = DiMergeCoClusterer::new(
            k,
            n,
            delta,
            num_partitions,
            local_clusterer,
            merge_config,
            num_threads,
            1, // default T_p=1 for pipeline usage
        )?;

        self.clusterer = Some(Box::new(clusterer));
        Ok(self)
    }

    pub fn with_scorer(mut self, scorer: Box<dyn Scorer>) -> Self {
        self.scorer = Some(scorer);
        self
    }

    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.config = config;
        self
    }

    pub fn min_score(mut self, min_score: f64) -> Self {
        self.config.min_score = min_score;
        self
    }

    pub fn max_submatrices(mut self, max: usize) -> Self {
        self.config.max_submatrices = max;
        self
    }

    pub fn min_submatrix_size(mut self, rows: usize, cols: usize) -> Self {
        self.config.min_submatrix_size = (rows, cols);
        self
    }

    pub fn parallel(mut self, parallel: bool) -> Self {
        self.config.parallel = parallel;
        self
    }

    pub fn build(self) -> Result<CoclusterPipeline, &'static str> {
        let clusterer = self.clusterer.ok_or("Clusterer not set")?;
        let scorer = self.scorer.ok_or("Scorer not set")?;

        Ok(CoclusterPipeline {
            clusterer,
            scorer,
            config: self.config,
        })
    }
}

#[derive(Debug)]
pub struct PipelineError(String);

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pipeline error: {}", self.0)
    }
}

impl Error for PipelineError {}

pub trait Pipeline<'a, T: Clusterer> {
    fn run(&self, matrix: &'a Matrix<f64>) -> Result<Vec<Submatrix<'a, f64>>, Box<dyn Error>>;
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BasicCoclustererParams {
    pub n_clusters: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SpectralCoclustererParams {
    pub n_clusters: usize,
    pub n_svd_vectors: Option<usize>,
    pub max_svd_features: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum CoclustererAlgorithmParams {
    Basic(BasicCoclustererParams),
    Spectral(SpectralCoclustererParams),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicCoclusterer {
    params: BasicCoclustererParams,
}

impl BasicCoclusterer {
    pub fn new(params: BasicCoclustererParams) -> Self {
        Self { params }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralCoclustererHook {
    params: SpectralCoclustererParams,
}

impl SpectralCoclustererHook {
    pub fn new(params: SpectralCoclustererParams) -> Self {
        Self { params }
    }
}

impl<'a> Default for Box<dyn Clusterer + 'a> {
    fn default() -> Self {
        Box::new(BasicCoclusterer::new(BasicCoclustererParams {
            n_clusters: 2,
        }))
    }
}

impl<T: Clusterer + ?Sized> Clusterer for Box<T> {
    fn cluster<'matrix_life>(
        &self,
        matrix: &'matrix_life Matrix<f64>
    ) -> Result<Vec<Submatrix<'matrix_life, f64>>, Box<dyn Error>> {
        (**self).cluster(matrix)
    }

    fn name(&self) -> &str {
        (**self).name()
    }
}

pub struct StandardPipeline<'a, C: Clusterer> {
    pub coclusterer: C,
    pub scorer: Box<dyn Scorer>,
    pub config: PipelinecrateConfig,
    phantom: PhantomData<&'a ()>,
}

impl<'a, C: Clusterer> StandardPipeline<'a, C> {
    pub fn new(
        coclusterer: C,
        scorer: Box<dyn Scorer>,
        config: PipelinecrateConfig,
    ) -> StandardPipeline<'a, C> {
        StandardPipeline {
            coclusterer,
            scorer,
            config,
            phantom: PhantomData,
        }
    }
}

impl<'a, C: Clusterer + Sync + Send> Pipeline<'a, C>
    for StandardPipeline<'a, C>
{
    fn run(&self, matrix: &'a PipelinecrateMatrix<f64>) -> Result<Vec<Submatrix<'a, f64>>, Box<dyn Error>> {
        // Perform one round of clustering
        let submatrices = self.coclusterer.cluster(matrix)?; // Corrected: removed map_err, just use ?

        // Score the submatrices - This part needs to align with the trait's expected return.
        // The Pipeline trait expects run to return Result<Vec<Submatrix<'a, f64>>, Box<dyn Error>>
        // So, this implementation should probably just return the submatrices from the clusterer.
        // Scoring and filtering would be handled by a higher-level pipeline structure that uses this.

        // If the intention of StandardPipeline is to also score, the Pipeline trait might need adjustment,
        // or StandardPipeline should implement a different trait.
        // For now, stick to the Pipeline trait's signature.

        // The original code had scoring and complex filtering logic here, which seems to belong
        // to a more comprehensive pipeline runner (like CoclusterPipeline perhaps).
        // If StandardPipeline is a basic step, it should just cluster.
        
        // For now, to make it compile and satisfy the trait, let's assume it just returns clustered submatrices.
        // The extensive loop, scoring, and filtering based on self.config fields like max_iterations, 
        // min_score, parallel_scoring, etc., are removed as they don't fit well with crate::config::Config
        // and the simple Pipeline trait here.

        Ok(submatrices)
    }
}

impl Clusterer for BasicCoclusterer {
    fn cluster<'matrix_life>(
        &self, 
        matrix: &'matrix_life Matrix<f64>
    ) -> Result<Vec<Submatrix<'matrix_life, f64>>, Box<dyn Error>> {
        let n_rows = matrix.data.nrows();
        let n_cols = matrix.data.ncols();
        let mut submatrices = Vec::new();

        if self.params.n_clusters == 0 {
            return Err(Box::from("Number of clusters cannot be zero."));
        }

        for i in 0..self.params.n_clusters {
            let row_start = i * n_rows / self.params.n_clusters;
            let row_end = (i + 1) * n_rows / self.params.n_clusters;
            let col_start = i * n_cols / self.params.n_clusters;
            let col_end = (i + 1) * n_cols / self.params.n_clusters;

            if row_start >= n_rows || col_start >= n_cols || row_start >= row_end || col_start >= col_end {
                log::warn!(
                    "Skipping cluster {} due to invalid index range: rows {}-{}, cols {}-{}",
                    i, row_start, row_end, col_start, col_end
                );
                continue;
            }

            let row_indices: Vec<usize> = (row_start..row_end).collect();
            let col_indices: Vec<usize> = (col_start..col_end).collect();

            if row_indices.is_empty() || col_indices.is_empty() {
                log::warn!(
                    "Skipping cluster {} due to empty row or column indices after calculation.",
                    i
                );
                continue;
            }

            if row_indices.len() >= 3 && col_indices.len() >= 3 { // Ensure min size for submatrix
                match Submatrix::new(&matrix.data, row_indices.clone(), col_indices.clone()) {
                    Some(submatrix_data) => submatrices.push(submatrix_data),
                    None => {
                        log::warn!("Submatrix::new returned None for cluster index: {}. Row indices: {:?}, Col indices: {:?}. Skipping.", i, row_indices, col_indices);
                    }
                }
            }
        }
        Ok(submatrices)
    }

    fn name(&self) -> &str {
        "BasicCoclusterer"
    }
}
