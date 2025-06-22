use crate::tensor3d::{Tensor3D, TensorSubspace};
use crate::tensor3d_scoring::{TensorScorer, TuckerScorer, TuckerScoringWeights};
use crate::tucker_decomposition::{TuckerDecomposer, TuckerDecomposition, TuckerRank, TuckerConfig};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
use std::time::{Duration, Instant};
use rayon::prelude::*;

/// 3D张量co-clustering配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor3DCoClusterConfig {
    /// Tucker分解的rank配置
    pub tucker_rank: TuckerRank,
    /// 最小分数阈值
    pub min_score: f64,
    /// 最大子空间数量
    pub max_subspaces: usize,
    /// 是否按分数排序
    pub sort_by_score: bool,
    /// 最小子空间大小 (mode1, mode2, mode3)
    pub min_subspace_size: (usize, usize, usize),
    /// 是否收集详细统计信息
    pub collect_stats: bool,
    /// 是否并行处理
    pub parallel: bool,
    /// 聚类数量建议
    pub num_clusters: Option<usize>,
}

impl Default for Tensor3DCoClusterConfig {
    fn default() -> Self {
        Self {
            tucker_rank: TuckerRank::uniform(3),
            min_score: 0.5,
            max_subspaces: 20,
            sort_by_score: true,
            min_subspace_size: (3, 3, 3),
            collect_stats: true,
            parallel: true,
            num_clusters: Some(5),
        }
    }
}

/// 3D张量co-clustering统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor3DCoClusterStats {
    pub total_duration: Duration,
    pub tucker_decomposition_duration: Duration,
    pub clustering_duration: Duration,
    pub scoring_duration: Duration,
    pub filtering_duration: Duration,
    pub initial_subspaces: usize,
    pub filtered_subspaces: usize,
    pub score_distribution: ScoreDistribution3D,
    pub tensor_properties: TensorProperties,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreDistribution3D {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub median: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorProperties {
    pub shape: [usize; 3],
    pub density: f64,
    pub sparsity: f64,
    pub frobenius_norm: f64,
    pub rank_estimates: [usize; 3],
}

/// 3D张量co-clustering结果
#[derive(Debug, Clone)]
pub struct Tensor3DCoClusterResult<'a> {
    pub subspaces: Vec<TensorSubspace<'a>>,
    pub scores: Vec<f64>,
    pub tucker_decomposition: TuckerDecomposition,
    pub stats: Option<Tensor3DCoClusterStats>,
}

impl<'a> Tensor3DCoClusterResult<'a> {
    pub fn summary(&self) -> String {
        let mut summary = format!("Found {} tensor subspaces", self.subspaces.len());
        
        if !self.scores.is_empty() {
            let min_score = self.scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_score = self.scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let avg_score = self.scores.iter().sum::<f64>() / self.scores.len() as f64;
            
            summary.push_str(&format!(
                "\nScore range: {:.4} - {:.4} (avg: {:.4})",
                min_score, max_score, avg_score
            ));
        }
        
        for (i, (subspace, score)) in self
            .subspaces
            .iter()
            .zip(&self.scores)
            .take(5)
            .enumerate()
        {
            let shape = subspace.shape();
            summary.push_str(&format!(
                "\n  #{}: {}×{}×{} subspace, score: {:.4}",
                i + 1,
                shape[0], shape[1], shape[2],
                score
            ));
        }
        
        if self.subspaces.len() > 5 {
            summary.push_str(&format!("\n  ... and {} more", self.subspaces.len() - 5));
        }
        
        summary
    }
}

/// 3D张量co-clustering算法的trait
pub trait Tensor3DClusterer: Send + Sync {
    fn cluster<'tensor_life>(
        &self,
        tensor: &'tensor_life Tensor3D<f64>,
    ) -> Result<Vec<TensorSubspace<'tensor_life>>, Box<dyn Error>>;
    
    fn name(&self) -> &str;
}

/// 基于Tucker分解的3D张量聚类器
#[derive(Debug, Clone)]
pub struct TuckerCoclusterer {
    config: TuckerConfig,
    num_clusters: usize,
}

impl TuckerCoclusterer {
    pub fn new(tucker_rank: TuckerRank, num_clusters: usize) -> Self {
        let mut config = TuckerConfig::default();
        config.ranks = tucker_rank;
        
        Self {
            config,
            num_clusters,
        }
    }
    
    pub fn with_config(config: TuckerConfig, num_clusters: usize) -> Self {
        Self {
            config,
            num_clusters,
        }
    }
    
    /// 基于Tucker分解的因子矩阵进行聚类
    fn cluster_factor_matrices(&self, decomposition: &TuckerDecomposition) -> Result<Vec<Vec<Vec<usize>>>, Box<dyn Error>> {
        // 对每个模式的因子矩阵进行K-means聚类
        let mut mode_clusters = Vec::new();
        
        // 模式1聚类
        let clusters1 = self.kmeans_clustering(&decomposition.factor_matrix1, self.num_clusters)?;
        mode_clusters.push(clusters1);
        
        // 模式2聚类
        let clusters2 = self.kmeans_clustering(&decomposition.factor_matrix2, self.num_clusters)?;
        mode_clusters.push(clusters2);
        
        // 模式3聚类
        let clusters3 = self.kmeans_clustering(&decomposition.factor_matrix3, self.num_clusters)?;
        mode_clusters.push(clusters3);
        
        Ok(mode_clusters)
    }
    
    /// 使用K-means对因子矩阵进行聚类
    fn kmeans_clustering(&self, factor_matrix: &nalgebra::DMatrix<f64>, k: usize) -> Result<Vec<Vec<usize>>, Box<dyn Error>> {
        use kmeans_smid::{KMeans, KMeansConfig};
        
        let data: Vec<f64> = factor_matrix.data.as_slice().to_vec();
        let kmeans: KMeans<f64, 8> = KMeans::new(data, factor_matrix.nrows(), factor_matrix.ncols());
        
        let result = kmeans.kmeans_lloyd(k.min(factor_matrix.nrows()), 100, KMeans::init_kmeanplusplus, &KMeansConfig::default());
        
        // 将分配结果转换为聚类组
        let mut clusters = vec![Vec::new(); k];
        for (idx, &cluster_id) in result.assignments.iter().enumerate() {
            if cluster_id < k {
                clusters[cluster_id].push(idx);
            }
        }
        
        // 过滤空聚类
        clusters.retain(|cluster| !cluster.is_empty());
        
        Ok(clusters)
    }
    
    /// 生成所有可能的子空间组合
    fn generate_subspace_combinations<'a>(
        &self,
        tensor: &'a Tensor3D<f64>,
        mode_clusters: &[Vec<Vec<usize>>],
    ) -> Vec<TensorSubspace<'a>> {
        let mut subspaces = Vec::new();
        
        // 生成所有模式组合
        for cluster1 in &mode_clusters[0] {
            for cluster2 in &mode_clusters[1] {
                for cluster3 in &mode_clusters[2] {
                    if let Some(subspace) = TensorSubspace::new(
                        tensor,
                        cluster1.clone(),
                        cluster2.clone(),
                        cluster3.clone(),
                    ) {
                        subspaces.push(subspace);
                    }
                }
            }
        }
        
        subspaces
    }
}

impl Tensor3DClusterer for TuckerCoclusterer {
    fn cluster<'tensor_life>(
        &self,
        tensor: &'tensor_life Tensor3D<f64>,
    ) -> Result<Vec<TensorSubspace<'tensor_life>>, Box<dyn Error>> {
        // 执行Tucker分解
        let decomposer = TuckerDecomposer::new(self.config.clone());
        let decomposition = decomposer.decompose(tensor)?;
        
        // 基于因子矩阵进行聚类
        let mode_clusters = self.cluster_factor_matrices(&decomposition)?;
        
        // 生成子空间组合
        let subspaces = self.generate_subspace_combinations(tensor, &mode_clusters);
        
        Ok(subspaces)
    }
    
    fn name(&self) -> &str {
        "TuckerCoclusterer"
    }
}

/// 3D张量co-clustering管道
pub struct Tensor3DCoClusterPipeline {
    clusterer: Box<dyn Tensor3DClusterer>,
    scorer: Box<dyn TensorScorer>,
    config: Tensor3DCoClusterConfig,
}

impl Tensor3DCoClusterPipeline {
    pub fn builder() -> Tensor3DPipelineBuilder {
        Tensor3DPipelineBuilder::new()
    }
    
    pub fn run<'a>(&self, tensor: &'a Tensor3D<f64>) -> Result<Tensor3DCoClusterResult<'a>, Box<dyn Error>> {
        let start_time = Instant::now();
        println!("Starting 3D tensor co-clustering with {} clusterer", self.clusterer.name());
        
        // Step 1: Tucker分解
        let tucker_start = Instant::now();
        let decomposer = TuckerDecomposer::new(TuckerConfig {
            ranks: self.config.tucker_rank.clone(),
            ..TuckerConfig::default()
        });
        let tucker_decomposition = decomposer.decompose(tensor)?;
        let tucker_duration = tucker_start.elapsed();
        println!("Tucker decomposition completed in {:?}", tucker_duration);
        
        // Step 2: 聚类
        let clustering_start = Instant::now();
        let subspaces_from_clusterer = self.clusterer.cluster(tensor)?;
        let clustering_duration = clustering_start.elapsed();
        let initial_subspaces_count = subspaces_from_clusterer.len();
        println!("Clustering completed in {:?}, found {} subspaces", 
                clustering_duration, initial_subspaces_count);
        
        // Step 3: 评分
        let scoring_start = Instant::now();
        let scores = if self.config.parallel {
            self.score_parallel(tensor, &subspaces_from_clusterer)
        } else {
            self.scorer.score_all(tensor, &subspaces_from_clusterer)
        };
        let scoring_duration = scoring_start.elapsed();
        println!("Scoring completed in {:?}", scoring_duration);
        
        // Step 4: 过滤和排序
        let filtering_start = Instant::now();
        let (filtered_subspaces, filtered_scores) = 
            self.filter_and_sort(subspaces_from_clusterer, scores, &self.config);
        let filtering_duration = filtering_start.elapsed();
        
        let total_duration = start_time.elapsed();
        println!("Pipeline completed in {:?}, {} subspaces retained", 
                total_duration, filtered_subspaces.len());
        
        // 收集统计信息
        let stats = if self.config.collect_stats {
            Some(self.collect_stats(
                total_duration,
                tucker_duration,
                clustering_duration,
                scoring_duration,
                filtering_duration,
                initial_subspaces_count,
                filtered_subspaces.len(),
                &filtered_scores,
                tensor,
            ))
        } else {
            None
        };
        
        Ok(Tensor3DCoClusterResult {
            subspaces: filtered_subspaces,
            scores: filtered_scores,
            tucker_decomposition,
            stats,
        })
    }
    
    fn score_parallel(&self, tensor: &Tensor3D<f64>, subspaces: &[TensorSubspace]) -> Vec<f64> {
        subspaces
            .par_iter()
            .map(|subspace| self.scorer.score(tensor, subspace))
            .collect()
    }
    
    fn filter_and_sort<'a>(
        &self,
        subspaces: Vec<TensorSubspace<'a>>,
        scores: Vec<f64>,
        config: &Tensor3DCoClusterConfig,
    ) -> (Vec<TensorSubspace<'a>>, Vec<f64>) {
        let mut combined: Vec<_> = subspaces.into_iter().zip(scores.into_iter()).collect();
        
        // 过滤
        combined.retain(|(subspace, score)| {
            let shape = subspace.shape();
            *score >= config.min_score
                && shape[0] >= config.min_subspace_size.0
                && shape[1] >= config.min_subspace_size.1
                && shape[2] >= config.min_subspace_size.2
        });
        
        // 排序
        if config.sort_by_score {
            combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        }
        
        // 限制数量
        combined.truncate(config.max_subspaces);
        
        // 分离结果
        combined.into_iter().unzip()
    }
    
    fn collect_stats(
        &self,
        total_duration: Duration,
        tucker_duration: Duration,
        clustering_duration: Duration,
        scoring_duration: Duration,
        filtering_duration: Duration,
        initial_count: usize,
        filtered_count: usize,
        scores: &[f64],
        tensor: &Tensor3D<f64>,
    ) -> Tensor3DCoClusterStats {
        let score_distribution = if scores.is_empty() {
            ScoreDistribution3D {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                std_dev: 0.0,
                median: 0.0,
            }
        } else {
            let mut sorted_scores = scores.to_vec();
            sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let min = *sorted_scores.first().unwrap();
            let max = *sorted_scores.last().unwrap();
            let mean = scores.iter().sum::<f64>() / scores.len() as f64;
            let variance = scores.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / scores.len() as f64;
            let std_dev = variance.sqrt();
            let median = if sorted_scores.len() % 2 == 0 {
                (sorted_scores[sorted_scores.len() / 2 - 1] + sorted_scores[sorted_scores.len() / 2]) / 2.0
            } else {
                sorted_scores[sorted_scores.len() / 2]
            };
            
            ScoreDistribution3D {
                min,
                max,
                mean,
                std_dev,
                median,
            }
        };
        
        // 计算张量属性
        let shape = tensor.shape();
        let total_elements = shape[0] * shape[1] * shape[2];
        let non_zero_count = tensor.data.iter().filter(|&&x| x.abs() > 1e-10).count();
        let density = non_zero_count as f64 / total_elements as f64;
        let sparsity = 1.0 - density;
        let frobenius_norm = tensor.frobenius_norm();
        
        // 估计每个模式的rank
        let rank_estimates = [
            (shape[0] as f64 * 0.1).ceil() as usize,
            (shape[1] as f64 * 0.1).ceil() as usize,
            (shape[2] as f64 * 0.1).ceil() as usize,
        ];
        
        let tensor_properties = TensorProperties {
            shape,
            density,
            sparsity,
            frobenius_norm,
            rank_estimates,
        };
        
        Tensor3DCoClusterStats {
            total_duration,
            tucker_decomposition_duration: tucker_duration,
            clustering_duration,
            scoring_duration,
            filtering_duration,
            initial_subspaces: initial_count,
            filtered_subspaces: filtered_count,
            score_distribution,
            tensor_properties,
        }
    }
}

/// 3D张量管道构建器
pub struct Tensor3DPipelineBuilder {
    clusterer: Option<Box<dyn Tensor3DClusterer>>,
    scorer: Option<Box<dyn TensorScorer>>,
    config: Tensor3DCoClusterConfig,
}

impl Tensor3DPipelineBuilder {
    pub fn new() -> Self {
        Self {
            clusterer: None,
            scorer: None,
            config: Tensor3DCoClusterConfig::default(),
        }
    }
    
    pub fn with_clusterer(mut self, clusterer: Box<dyn Tensor3DClusterer>) -> Self {
        self.clusterer = Some(clusterer);
        self
    }
    
    pub fn with_scorer(mut self, scorer: Box<dyn TensorScorer>) -> Self {
        self.scorer = Some(scorer);
        self
    }
    
    pub fn with_config(mut self, config: Tensor3DCoClusterConfig) -> Self {
        self.config = config;
        self
    }
    
    pub fn tucker_rank(mut self, rank1: usize, rank2: usize, rank3: usize) -> Self {
        self.config.tucker_rank = TuckerRank::new(rank1, rank2, rank3);
        self
    }
    
    pub fn min_score(mut self, min_score: f64) -> Self {
        self.config.min_score = min_score;
        self
    }
    
    pub fn max_subspaces(mut self, max: usize) -> Self {
        self.config.max_subspaces = max;
        self
    }
    
    pub fn min_subspace_size(mut self, size1: usize, size2: usize, size3: usize) -> Self {
        self.config.min_subspace_size = (size1, size2, size3);
        self
    }
    
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.config.parallel = parallel;
        self
    }
    
    pub fn num_clusters(mut self, num: usize) -> Self {
        self.config.num_clusters = Some(num);
        self
    }
    
    pub fn build(self) -> Result<Tensor3DCoClusterPipeline, &'static str> {
        let clusterer = self.clusterer.unwrap_or_else(|| {
            Box::new(TuckerCoclusterer::new(
                self.config.tucker_rank.clone(),
                self.config.num_clusters.unwrap_or(5),
            ))
        });
        
        let scorer = self.scorer.unwrap_or_else(|| {
            Box::new(TuckerScorer::new(self.config.tucker_rank.clone()))
        });
        
        Ok(Tensor3DCoClusterPipeline {
            clusterer,
            scorer,
            config: self.config,
        })
    }
}

/// 3D张量co-clustering错误
#[derive(Debug)]
pub enum Tensor3DCoClusterError {
    InvalidConfiguration(String),
    TuckerDecompositionFailed(String),
    ClusteringFailed(String),
    ScoringFailed(String),
}

impl fmt::Display for Tensor3DCoClusterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tensor3DCoClusterError::InvalidConfiguration(msg) => write!(f, "Invalid configuration: {}", msg),
            Tensor3DCoClusterError::TuckerDecompositionFailed(msg) => write!(f, "Tucker decomposition failed: {}", msg),
            Tensor3DCoClusterError::ClusteringFailed(msg) => write!(f, "Clustering failed: {}", msg),
            Tensor3DCoClusterError::ScoringFailed(msg) => write!(f, "Scoring failed: {}", msg),
        }
    }
}

impl Error for Tensor3DCoClusterError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor3d::Tensor3D;
    
    #[test]
    fn test_tucker_coclusterer() {
        let tensor = Tensor3D::random([12, 10, 8]);
        let clusterer = TuckerCoclusterer::new(TuckerRank::uniform(3), 3);
        
        let result = clusterer.cluster(&tensor);
        assert!(result.is_ok());
        
        let subspaces = result.unwrap();
        assert!(!subspaces.is_empty());
        println!("Found {} subspaces", subspaces.len());
    }
    
    #[test]
    fn test_tensor3d_pipeline() {
        let tensor = Tensor3D::random([15, 12, 10]);
        
        let pipeline = Tensor3DCoClusterPipeline::builder()
            .tucker_rank(3, 3, 3)
            .min_score(0.3)
            .max_subspaces(10)
            .min_subspace_size(2, 2, 2)
            .num_clusters(4)
            .build()
            .unwrap();
        
        let result = pipeline.run(&tensor);
        assert!(result.is_ok());
        
        let cluster_result = result.unwrap();
        println!("Pipeline result: {}", cluster_result.summary());
        
        if let Some(stats) = &cluster_result.stats {
            println!("Total time: {:?}", stats.total_duration);
            println!("Tucker decomposition time: {:?}", stats.tucker_decomposition_duration);
            println!("Tensor properties: {:?}", stats.tensor_properties);
        }
    }
}