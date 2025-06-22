# Fast Co-clustering Library

A high-performance Rust library for bi-clustering (co-clustering) large matrices using SVD-based algorithms and flexible scoring methods.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Input Data Format](#input-data-format)
- [Core Algorithms](#core-algorithms)
- [Scoring Methods](#scoring-methods)
- [Pipeline Configuration](#pipeline-configuration)
- [Usage Examples](#usage-examples)
- [Output Format](#output-format)
- [Performance](#performance)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

Fast Co-clustering finds coherent subgroups in data by simultaneously clustering rows and columns of a matrix. This is particularly useful for:

- **Gene expression analysis**: Finding co-expressed genes and sample groups
- **Recommendation systems**: Discovering user-item preference patterns
- **Market basket analysis**: Identifying product-customer segments
- **Document clustering**: Finding document-term associations
- **Social network analysis**: Detecting community structures

### Key Features

- **High Performance**: Parallel processing with Rayon
- **Flexible Algorithms**: SVD-based, spectral, and basic clustering
- **Multiple Scoring Methods**: Pearson correlation, exponential, compatibility scoring
- **Configurable Pipeline**: Easy-to-use builder pattern with sensible defaults
- **Memory Efficient**: Optimized for large matrices
- **Type Safe**: Full Rust type safety and error handling

## Installation

### Prerequisites

- Rust 1.70+ 
- BLAS/LAPACK libraries (for linear algebra operations)

### Add to Your Project

```toml
[dependencies]
fast_cocluster = { git = "https://github.com/wzh4464/fast_cocluster" }
nalgebra = "0.33"
ndarray = "0.15"
log = "0.4"
env_logger = "0.11"  # For logging (optional)
```

### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get install libblas-dev liblapack-dev
```

**macOS:**
```bash
brew install openblas lapack
```

**Windows:**
Install Intel MKL or OpenBLAS through vcpkg.

## Quick Start

```rust
use fast_cocluster::pipeline::*;
use fast_cocluster::scoring::*;
use fast_cocluster::Matrix;
use nalgebra::DMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging (optional)
    env_logger::init();
    
    // Create your data matrix (rows = samples, cols = features)
    let data = DMatrix::from_vec(100, 50, vec![/* your data */]);
    let matrix = Matrix::new(data.into());
    
    // Build and configure the pipeline
    let pipeline = CoclusterPipeline::builder()
        .with_clusterer(Box::new(SVDClusterer::new(5, 0.1)))
        .with_scorer(Box::new(PearsonScorer::new(3, 3)))
        .min_score(0.6)
        .max_submatrices(10)
        .build()?;
    
    // Run co-clustering
    let result = pipeline.run(&matrix)?;
    
    // Process results
    println!("Found {} co-clusters", result.submatrices.len());
    for (i, (submatrix, score)) in result.submatrices.iter()
        .zip(&result.scores).enumerate() {
        println!("Cluster {}: {}×{} (score: {:.3})", 
                 i+1, 
                 submatrix.row_indices.len(), 
                 submatrix.col_indices.len(), 
                 score);
    }
    
    Ok(())
}
```

## Input Data Format

### Matrix Structure

The input should be a 2D matrix where:
- **Rows**: Observations/samples (e.g., genes, users, documents)
- **Columns**: Features/variables (e.g., conditions, items, terms)
- **Values**: Numeric data (f64)

### Supported Input Formats

#### 1. From Vec<f64>
```rust
use ndarray::Array2;
use fast_cocluster::Matrix;

let data_vec = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let array = Array2::from_shape_vec((2, 3), data_vec)?;
let matrix = Matrix::new(array);
```

#### 2. From CSV File
```rust
use csv::Reader;
use std::fs::File;

fn load_from_csv(path: &str) -> Result<Matrix<f64>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = Reader::from_reader(file);
    
    let mut data = Vec::new();
    let mut rows = 0;
    let mut cols = 0;
    
    for result in reader.records() {
        let record = result?;
        if rows == 0 {
            cols = record.len();
        }
        
        for field in record.iter() {
            data.push(field.parse::<f64>()?);
        }
        rows += 1;
    }
    
    let array = Array2::from_shape_vec((rows, cols), data)?;
    Ok(Matrix::new(array))
}
```

#### 3. From NumPy Arrays (.npy)
```rust
use ndarray_npy::ReadNpyExt;
use std::fs::File;

fn load_from_npy(path: &str) -> Result<Matrix<f64>, Box<dyn std::error::Error>> {
    let reader = File::open(path)?;
    let array: Array2<f64> = Array2::read_npy(reader)?;
    Ok(Matrix::new(array))
}
```

### Data Preprocessing Recommendations

```rust
// 1. Normalization (z-score)
fn normalize_matrix(matrix: &mut Array2<f64>) {
    let mean = matrix.mean().unwrap();
    let std = matrix.std(1.0);
    matrix.mapv_inplace(|x| (x - mean) / std);
}

// 2. Log transformation (for expression data)
fn log_transform(matrix: &mut Array2<f64>) {
    matrix.mapv_inplace(|x| (x + 1.0).ln());
}

// 3. Missing value handling
fn handle_missing_values(matrix: &mut Array2<f64>, fill_value: f64) {
    matrix.mapv_inplace(|x| if x.is_nan() { fill_value } else { x });
}
```

## 🆕 3D 张量 Co-clustering (新功能)

### Tucker 分解驱动的3D张量分析

本项目现已支持3D张量的co-clustering分析，使用Tucker分解算法实现高效的多维数据聚类。

#### 快速开始 - 3D张量

```rust
use fast_cocluster::tensor3d::*;
use fast_cocluster::tensor3d_scoring::*;
use fast_cocluster::tucker_decomposition::*;

// 创建3D张量 (用户 × 物品 × 上下文)
let tensor = Tensor3D::random([100, 50, 20]);

// Tucker分解
let tucker_rank = TuckerRank::new(5, 4, 3); // 指定每个模式的rank
let decomposer = TuckerDecomposer::with_ranks(5, 4, 3);
let decomposition = decomposer.decompose(&tensor)?;

// 张量评分
let scorer = TuckerScorer::new(tucker_rank);
let subspace = TensorSubspace::new(&tensor, vec![0,1,2], vec![0,1,2], vec![0,1,2]).unwrap();
let score = scorer.score(&tensor, &subspace);

println!("Tucker分解重构误差: {:.4}", decomposition.reconstruction_error);
println!("子空间质量分数: {:.4}", score);
```

#### 支持的应用场景

- **基因表达分析**: 基因 × 条件 × 时间点
- **推荐系统**: 用户 × 物品 × 上下文  
- **时空数据**: 传感器 × 地点 × 时间
- **社交网络**: 用户 × 内容 × 社群
- **金融分析**: 资产 × 因子 × 时期

#### 核心特性

- **Tucker分解**: 高效的3D张量分解算法
- **多种评分器**: Tucker、密度、方差、组合评分
- **可配置Rank**: 灵活的Tucker rank配置
- **高性能**: 并行计算和内存优化
- **完整Pipeline**: 从数据加载到结果输出

#### 与2D co-clustering的对比

| 特性 | 2D矩阵 | 3D张量 |
|------|--------|--------|
| 算法 | SVD + K-means | Tucker分解 + 聚类 |
| 数据结构 | Matrix<T> | Tensor3D<T> |
| 分解方式 | 奇异值分解 | Tucker分解 |
| 应用场景 | 二维关联分析 | 多维关联分析 |
| 计算复杂度 | O(mn min(m,n)) | O(I₁I₂I₃R₁R₂R₃) |

#### 示例：完整3D分析流程

```bash
# 运行3D张量co-clustering演示
cargo run --example tensor3d_complete_demo

# 运行基础3D张量示例
cargo run --example tensor3d_cocluster_example
```

## 替换原子化 Cocluster 方法

### 从原子化到模块化的迁移

原始的 `Coclusterer::cocluster()` 方法是一个原子化实现，将所有算法步骤硬编码在一个函数中。新的模块化实现提供了更好的灵活性和可扩展性。

#### 原子化方法 (旧)
```rust
use fast_cocluster::cocluster::Coclusterer;

// 原子化 - 不可定制
let mut coclusterer = Coclusterer::new(matrix, 5, 0.1);
let result = coclusterer.cocluster()?; // 固定: SVD + K-means
```

#### 模块化方法 (新) - 等效替换
```rust
use fast_cocluster::modular_cocluster::*;

// 模块化 - 完全等效于原子化方法
let mut coclusterer = ModularCoclusterer::with_defaults(matrix, 5);
let result = coclusterer.cocluster()?;
```

#### 模块化方法 - 增强功能
```rust
// 使用改进的归一化
let mut coclusterer = ModularCoclusterer::with_zscore(matrix, 5);

// 使用加权特征组合
let mut coclusterer = ModularCoclusterer::with_weighted_features(matrix, 5, 0.8, 0.2);

// 完全自定义组件
let mut coclusterer = ModularCoclustererBuilder::new()
    .matrix(matrix)
    .k(5)
    .normalizer(Box::new(ZScoreNormalizer))
    .reducer(Box::new(SVDReducer))
    .combiner(Box::new(WeightedCombiner { row_weight: 0.7, col_weight: 0.3 }))
    .assigner(Box::new(KMeansAssigner))
    .build()?;
```

### 自定义组件示例

```rust
// 自定义归一化器
struct MinMaxNormalizer;
impl MatrixNormalizer for MinMaxNormalizer {
    fn normalize(&self, matrix: &DMatrix<f64>) -> DMatrix<f64> {
        let min_val = matrix.min();
        let max_val = matrix.max();
        let range = max_val - min_val;
        if range > 0.0 {
            matrix.map(|x| (x - min_val) / range)
        } else {
            matrix.clone()
        }
    }
}

// 使用自定义组件
let mut coclusterer = ModularCoclustererBuilder::new()
    .matrix(matrix)
    .k(5)
    .normalizer(Box::new(MinMaxNormalizer))
    .build()?;
```

### 性能对比

| 方法 | 执行时间 | 优势 | 限制 |
|------|----------|------|------|
| 原子化 | 105ms | 简单直接 | 不可定制、不可扩展 |
| 模块化默认 | 76ms | 等效功能 + 可扩展 | 无 |
| 模块化自定义 | 75ms | 完全可定制 | 需要理解组件接口 |

## Core Algorithms

### 1. SVD Clusterer (Recommended)

Uses Singular Value Decomposition for dimensionality reduction followed by k-means clustering.

```rust
let clusterer = SVDClusterer::new(
    5,    // Number of clusters
    0.1   // Convergence tolerance
);
```

**Best for**: General-purpose co-clustering, works well with most data types.

**Advantages**: 
- Fast and memory-efficient
- Handles noise well
- Good for large matrices

### 2. Spectral Co-clusterer

Advanced spectral clustering approach for non-linear patterns.

```rust
let clusterer = SpectralCoclustererHook::new(
    SpectralCoclustererParams {
        n_clusters: 5,
        n_svd_vectors: Some(10),
        max_svd_features: Some(100),
    }
);
```

**Best for**: Complex, non-linear patterns in data.

### 3. Basic Co-clusterer

Simple partitioning approach for quick results.

```rust
let clusterer = BasicCoclusterer::new(
    BasicCoclustererParams {
        n_clusters: 3,
    }
);
```

**Best for**: Quick prototyping, simple datasets.

## Scoring Methods

### 1. Pearson Correlation Scorer

Measures linear correlation within co-clusters.

```rust
let scorer = PearsonScorer::new(
    3,  // Minimum rows
    3   // Minimum columns
);
```

**Range**: [-1, 1] (higher is better)  
**Best for**: Linear relationships, gene expression data

### 2. Exponential Scorer

Emphasizes tight clustering with exponential decay.

```rust
let scorer = ExponentialScorer::new(1.5); // Decay parameter
```

**Range**: [0, ∞) (higher is better)  
**Best for**: Compact, well-defined clusters

### 3. Compatibility Scorer

Measures variance-based cluster quality.

```rust
let scorer = CompatibilityScorer::new(
    0.5,  // Row weight
    0.5   // Column weight
);
```

**Range**: [0, 1] (higher is better)  
**Best for**: Balanced row-column clustering

### 4. Composite Scorer

Combines multiple scoring methods with weights.

```rust
let scorer = CompositeScorer::new()
    .add_scorer(Box::new(PearsonScorer::new(3, 3)), 0.6)
    .add_scorer(Box::new(ExponentialScorer::new(1.0)), 0.3)
    .add_scorer(Box::new(CompatibilityScorer::new(0.5, 0.5)), 0.1);
```

## Pipeline Configuration

### Basic Configuration

```rust
let config = PipelineConfig {
    min_score: 0.5,              // Minimum score threshold
    max_submatrices: 50,         // Maximum number of results
    sort_by_score: true,         // Sort results by score
    min_submatrix_size: (3, 3),  // Minimum size (rows, cols)
    collect_stats: true,         // Collect performance statistics
    parallel: true,              // Enable parallel processing
};
```

### Advanced Configuration

```rust
let pipeline = CoclusterPipeline::builder()
    .with_clusterer(Box::new(SVDClusterer::new(8, 0.1)))
    .with_scorer(Box::new(composite_scorer))
    .min_score(0.7)
    .max_submatrices(20)
    .min_submatrix_size(5, 5)
    .parallel(true)
    .build()?;
```

## Usage Examples

### Example 1: Gene Expression Analysis

```rust
use fast_cocluster::*;
use nalgebra::DMatrix;

fn analyze_gene_expression() -> Result<(), Box<dyn std::error::Error>> {
    // Load gene expression matrix (genes × samples)
    let expression_data = load_expression_data("expression.csv")?;
    
    // Configure for biological data
    let pipeline = CoclusterPipeline::builder()
        .with_clusterer(Box::new(SVDClusterer::new(6, 0.1)))
        .with_scorer(Box::new(PearsonScorer::new(5, 3)))
        .min_score(0.7)
        .max_submatrices(15)
        .min_submatrix_size(10, 5)  // At least 10 genes, 5 samples
        .build()?;
    
    let result = pipeline.run(&expression_data)?;
    
    // Find co-expressed gene modules
    for (i, (submatrix, score)) in result.submatrices.iter()
        .zip(&result.scores).enumerate() {
        println!("Gene module {}: {} genes × {} samples (r={:.3})", 
                 i+1, 
                 submatrix.row_indices.len(), 
                 submatrix.col_indices.len(), 
                 score);
        
        // Get gene and sample indices
        println!("Genes: {:?}", &submatrix.row_indices[..5.min(submatrix.row_indices.len())]);
        println!("Samples: {:?}", &submatrix.col_indices);
    }
    
    Ok(())
}
```

### Example 2: Recommendation System

```rust
fn analyze_user_item_preferences() -> Result<(), Box<dyn std::error::Error>> {
    // Load user-item rating matrix
    let ratings = load_ratings_matrix("ratings.csv")?;
    
    // Configure for recommendation data
    let pipeline = CoclusterPipeline::builder()
        .with_clusterer(Box::new(SVDClusterer::new(10, 0.05)))
        .with_scorer(Box::new(CompatibilityScorer::new(0.6, 0.4)))
        .min_score(0.6)
        .max_submatrices(25)
        .min_submatrix_size(5, 3)  // At least 5 users, 3 items
        .build()?;
    
    let result = pipeline.run(&ratings)?;
    
    // Analyze user-item co-clusters
    for (i, (submatrix, score)) in result.submatrices.iter()
        .zip(&result.scores).enumerate() {
        println!("User-Item cluster {}: {} users × {} items (score={:.3})", 
                 i+1, 
                 submatrix.row_indices.len(), 
                 submatrix.col_indices.len(), 
                 score);
    }
    
    Ok(())
}
```

### Example 3: Time Series Co-clustering

```rust
fn analyze_time_series() -> Result<(), Box<dyn std::error::Error>> {
    // Load time series matrix (sensors × time points)
    let time_series = load_time_series("sensors.csv")?;
    
    // Use exponential scorer for tight temporal patterns
    let scorer = CompositeScorer::new()
        .add_scorer(Box::new(ExponentialScorer::new(2.0)), 0.7)
        .add_scorer(Box::new(PearsonScorer::new(3, 3)), 0.3);
    
    let pipeline = CoclusterPipeline::builder()
        .with_clusterer(Box::new(SVDClusterer::new(5, 0.1)))
        .with_scorer(Box::new(scorer))
        .min_score(0.8)
        .max_submatrices(10)
        .build()?;
    
    let result = pipeline.run(&time_series)?;
    
    // Analyze temporal patterns
    for (i, (submatrix, score)) in result.submatrices.iter()
        .zip(&result.scores).enumerate() {
        println!("Pattern {}: {} sensors × {} time points (score={:.3})", 
                 i+1, 
                 submatrix.row_indices.len(), 
                 submatrix.col_indices.len(), 
                 score);
    }
    
    Ok(())
}
```

## Output Format

### StepResult Structure

```rust
pub struct StepResult<'a> {
    pub submatrices: Vec<Submatrix<'a, f64>>,  // Found co-clusters
    pub scores: Vec<f64>,                       // Corresponding scores
    pub stats: Option<PipelineStats>,           // Performance statistics
}
```

### Submatrix Structure

```rust
pub struct Submatrix<'a, T> {
    pub row_indices: Vec<usize>,    // Row indices in original matrix
    pub col_indices: Vec<usize>,    // Column indices in original matrix
    // Internal data view...
}
```

### Accessing Results

```rust
let result = pipeline.run(&matrix)?;

// Iterate through co-clusters
for (i, (submatrix, score)) in result.submatrices.iter()
    .zip(&result.scores).enumerate() {
    
    // Get dimensions
    let n_rows = submatrix.row_indices.len();
    let n_cols = submatrix.col_indices.len();
    
    // Access specific elements
    let first_row = submatrix.row_indices[0];
    let first_col = submatrix.col_indices[0];
    
    // Get the actual data values
    let data_value = matrix.data[(first_row, first_col)];
    
    println!("Cluster {}: {}×{} (score: {:.3})", i+1, n_rows, n_cols, score);
}
```

### Exporting Results

```rust
use std::fs::File;
use std::io::Write;

fn export_results(result: &StepResult, filename: &str) -> std::io::Result<()> {
    let mut file = File::create(filename)?;
    
    writeln!(file, "cluster_id,score,n_rows,n_cols,row_indices,col_indices")?;
    
    for (i, (submatrix, score)) in result.submatrices.iter()
        .zip(&result.scores).enumerate() {
        
        let row_str = submatrix.row_indices.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(";");
            
        let col_str = submatrix.col_indices.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(";");
        
        writeln!(file, "{},{:.4},{},{},\"{}\",\"{}\"", 
                 i+1, score, 
                 submatrix.row_indices.len(), 
                 submatrix.col_indices.len(),
                 row_str, col_str)?;
    }
    
    Ok(())
}
```

## Performance

### Benchmarks

**Dataset**: 1000×500 random matrix  
**Hardware**: Intel i7-8750H, 16GB RAM

| Algorithm | Clusters | Time | Memory |
| --------- | -------- | ---- | ------ |
| SVD       | 5        | 1.2s | 45MB   |
| SVD       | 10       | 2.1s | 52MB   |
| Basic     | 5        | 0.3s | 25MB   |
| Spectral  | 5        | 3.8s | 85MB   |

### Optimization Guidelines

#### 1. Choose Appropriate Parameters

```rust
// For large matrices (>1000×1000)
let pipeline = CoclusterPipeline::builder()
    .with_clusterer(Box::new(SVDClusterer::new(5, 0.1)))  // Fewer clusters
    .max_submatrices(20)    // Limit results
    .min_submatrix_size(10, 10)  // Larger minimum size
    .parallel(true)         // Enable parallelism
    .build()?;
```

#### 2. Memory Management

```rust
// Process in batches for very large datasets
fn process_large_matrix(matrix: &Array2<f64>) -> Result<Vec<StepResult>, Box<dyn std::error::Error>> {
    let chunk_size = 1000;
    let mut results = Vec::new();
    
    for chunk in matrix.axis_chunks_iter(Axis(0), chunk_size) {
        let chunk_matrix = Matrix::new(chunk.to_owned());
        let result = pipeline.run(&chunk_matrix)?;
        results.push(result);
    }
    
    Ok(results)
}
```

#### 3. Parallel Processing

```rust
// Set number of threads
std::env::set_var("RAYON_NUM_THREADS", "8");

// Enable parallel scoring
let config = PipelineConfig {
    parallel: true,
    // ... other settings
};
```

## API Reference

### Core Types

- `Matrix<T>`: Wrapper around ndarray::Array2<T>
- `Submatrix<'a, T>`: View into a matrix with row/column indices
- `CoclusterPipeline`: Main pipeline for co-clustering
- `PipelineConfig`: Configuration structure

### Clusterer Trait

```rust
pub trait Clusterer: Send + Sync {
    fn cluster<'matrix_life>(
        &self,
        matrix: &'matrix_life Matrix<f64>
    ) -> Result<Vec<Submatrix<'matrix_life, f64>>, Box<dyn Error>>;
    
    fn name(&self) -> &str;
}
```

### Scorer Trait

```rust
pub trait Scorer: Send + Sync {
    fn score(&self, matrix: &Matrix<f64>, submatrix: &Submatrix<f64>) -> f64;
    fn score_all(&self, matrix: &Matrix<f64>, submatrices: &[Submatrix<f64>]) -> Vec<f64>;
}
```

### Builder Pattern

```rust
impl PipelineBuilder {
    pub fn new() -> Self;
    pub fn with_clusterer(self, clusterer: Box<dyn Clusterer>) -> Self;
    pub fn with_scorer(self, scorer: Box<dyn Scorer>) -> Self;
    pub fn min_score(self, min_score: f64) -> Self;
    pub fn max_submatrices(self, max: usize) -> Self;
    pub fn min_submatrix_size(self, rows: usize, cols: usize) -> Self;
    pub fn parallel(self, parallel: bool) -> Self;
    pub fn build(self) -> Result<CoclusterPipeline, &'static str>;
}
```

## Troubleshooting

### Common Issues

#### 1. "SVD did not converge"

**Cause**: Matrix has numerical issues or is rank-deficient.

**Solutions**:
- Increase tolerance: `SVDClusterer::new(k, 0.2)`
- Preprocess data: normalize or add small noise
- Check for NaN/infinite values

```rust
// Check for problematic values
fn validate_matrix(matrix: &Array2<f64>) -> Result<(), &'static str> {
    if matrix.iter().any(|&x| x.is_nan() || x.is_infinite()) {
        return Err("Matrix contains NaN or infinite values");
    }
    Ok(())
}
```

#### 2. "No co-clusters found"

**Cause**: Parameters too restrictive or data doesn't have clear structure.

**Solutions**:
- Lower `min_score` threshold
- Reduce `min_submatrix_size`
- Try different scoring methods
- Increase number of clusters

```rust
// More permissive configuration
let pipeline = CoclusterPipeline::builder()
    .min_score(0.3)              // Lower threshold
    .min_submatrix_size(2, 2)    // Smaller minimum size
    .max_submatrices(100)        // More results
    .build()?;
```

#### 3. "Out of memory"

**Cause**: Matrix too large for available memory.

**Solutions**:
- Process in chunks
- Use dimensionality reduction first
- Increase virtual memory
- Use streaming algorithms

```rust
// Chunked processing
fn process_in_chunks(matrix: &Array2<f64>, chunk_size: usize) 
    -> Result<Vec<StepResult>, Box<dyn std::error::Error>> {
    // Implementation above
}
```

#### 4. "Poor clustering quality"

**Cause**: Inappropriate algorithm or parameters for data type.

**Solutions**:
- Try different clustering algorithms
- Experiment with scoring methods
- Preprocess data (normalization, log transform)
- Adjust number of clusters

```rust
// Try multiple configurations
let algorithms = vec![
    SVDClusterer::new(5, 0.1),
    SVDClusterer::new(10, 0.05),
    // Add more variants
];

for clusterer in algorithms {
    let result = pipeline_with_clusterer(clusterer).run(&matrix)?;
    evaluate_quality(&result);
}
```

### Debug Tips

#### Enable Detailed Logging

```rust
env_logger::Builder::from_default_env()
    .filter_level(log::LevelFilter::Debug)
    .init();
```

#### Collect Statistics

```rust
let config = PipelineConfig {
    collect_stats: true,
    // ... other settings
};

// Access statistics
if let Some(stats) = &result.stats {
    println!("Total time: {:?}", stats.total_duration);
    println!("Clustering time: {:?}", stats.clustering_duration);
    println!("Scoring time: {:?}", stats.scoring_duration);
    println!("Score distribution: {:.3} ± {:.3}", 
             stats.score_distribution.mean,
             stats.score_distribution.std_dev);
}
```

#### Visualize Results

```rust
// Simple visualization function
fn print_cluster_matrix(matrix: &Array2<f64>, submatrix: &Submatrix<f64>) {
    println!("Cluster data (first 5×5):");
    for (i, &row_idx) in submatrix.row_indices.iter().take(5).enumerate() {
        for (j, &col_idx) in submatrix.col_indices.iter().take(5).enumerate() {
            print!("{:6.2} ", matrix[(row_idx, col_idx)]);
        }
        println!();
    }
}
```

### Performance Issues

#### Slow Performance

1. **Enable parallel processing**: Set `parallel: true`
2. **Reduce precision**: Increase tolerance values
3. **Limit results**: Reduce `max_submatrices`
4. **Use appropriate algorithm**: SVD for general use, Basic for quick results

#### High Memory Usage

1. **Limit cluster count**: Reduce number of clusters
2. **Disable statistics**: Set `collect_stats: false`
3. **Process in batches**: Split large matrices
4. **Use views instead of copies**: Ensure efficient memory usage

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/wzh4464/fast_cocluster
cd fast_cocluster
cargo build
cargo test
```

### Running Benchmarks

```bash
cargo bench
```

### Adding New Algorithms

Implement the `Clusterer` trait:

```rust
pub struct MyClusterer {
    // parameters
}

impl Clusterer for MyClusterer {
    fn cluster<'matrix_life>(
        &self,
        matrix: &'matrix_life Matrix<f64>
    ) -> Result<Vec<Submatrix<'matrix_life, f64>>, Box<dyn Error>> {
        // Implementation
    }
    
    fn name(&self) -> &str {
        "MyClusterer"
    }
}
```

### Adding New Scoring Methods

Implement the `Scorer` trait:

```rust
pub struct MyScorer {
    // parameters
}

impl Scorer for MyScorer {
    fn score(&self, matrix: &Matrix<f64>, submatrix: &Submatrix<f64>) -> f64 {
        // Implementation
    }
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@inproceedings{wu2024ScalableCoclusteringLargescale,
  title = {Scalable Co-Clustering for Large-Scale Data through Dynamic Partitioning and Hierarchical Merging},
  booktitle = {2024 {{IEEE International Conference}} on {{Systems}}, {{Man}}, and {{Cybernetics}} ({{SMC}})},
  author = {Wu, Zihan and Huang, Zhaoke and Yan, Hong},
  year = {2024},
  month = oct,
  pages = {4686--4691},
  publisher = {IEEE},
  address = {Kuching, Malaysia},
  doi = {10.1109/SMC54092.2024.10832071},
  copyright = {https://doi.org/10.15223/policy-029},
  isbn = {978-1-6654-1020-5},
}
```
