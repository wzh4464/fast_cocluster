# Fast Cocluster Pipeline 使用指南

## 快速开始

### 1. 添加依赖

```toml
[dependencies]
fast_cocluster = { path = "../fast_cocluster" }
nalgebra = "0.32"
log = "0.4"
env_logger = "0.10"
```

### 2. 基本使用

```rust
use fast_cocluster::pipeline::*;
use fast_cocluster::scoring::*;
use nalgebra::DMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 创建数据矩阵
    let data = DMatrix::from_vec(100, 80, vec![/* your data */]);
    let matrix = Matrix { data, row_labels: vec![], col_labels: vec![] };
    
    // 构建Pipeline
    let pipeline = CoclusterPipeline::builder()
        .with_clusterer(Box::new(SVDClusterer::new(5, 0.1)))
        .with_scorer(Box::new(PearsonScorer::new(3, 3)))
        .min_score(0.5)
        .max_submatrices(10)
        .build()?;
    
    // 运行分析
    let result = pipeline.run(&matrix)?;
    
    // 使用结果
    println!("Found {} biclusters", result.submatrices.len());
    for (sub, score) in result.submatrices.iter().zip(&result.scores) {
        println!("Bicluster: {}x{}, score: {:.3}", 
                 sub.rows.len(), sub.cols.len(), score);
    }
    
    Ok(())
}
```

## 高级配置

### 使用自定义配置

```rust
let config = PipelineConfig {
    min_score: 0.6,
    max_submatrices: 20,
    sort_by_score: true,
    min_submatrix_size: (5, 5),
    collect_stats: true,
    parallel: true,
};

let pipeline = CoclusterPipeline::builder()
    .with_config(config)
    .with_clusterer(Box::new(SVDClusterer::new(6, 0.1)))
    .with_scorer(Box::new(ExponentialScorer::new(1.0)))
    .build()?;
```

### 组合多个评分器

```rust
let scorer = CompositeScorer::new()
    .add_scorer(Box::new(PearsonScorer::new(3, 3)), 0.5)
    .add_scorer(Box::new(ExponentialScorer::new(1.0)), 0.3)
    .add_scorer(Box::new(CompatibilityScorer::new(0.5, 0.5)), 0.2);

let pipeline = CoclusterPipeline::builder()
    .with_scorer(Box::new(scorer))
    // ... 其他配置
    .build()?;
```

## 实现自定义组件

### 自定义聚类器

```rust
use fast_cocluster::pipeline::Clusterer;

struct MyCustomClusterer {
    // 自定义参数
}

impl Clusterer for MyCustomClusterer {
    fn cluster(&self, matrix: &Matrix) -> Result<Vec<Submatrix>, Box<dyn Error>> {
        // 实现自定义聚类算法
        todo!()
    }
    
    fn name(&self) -> &str {
        "MyCustom"
    }
}

// 使用
let pipeline = CoclusterPipeline::builder()
    .with_clusterer(Box::new(MyCustomClusterer { /* params */ }))
    // ...
    .build()?;
```

### 自定义评分器

```rust
use fast_cocluster::scoring::Scorer;

struct MyCustomScorer {
    threshold: f64,
}

impl Scorer for MyCustomScorer {
    fn score(&self, matrix: &Matrix, submatrix: &Submatrix) -> f64 {
        // 实现自定义评分逻辑
        // 返回值：分数越高表示质量越好
        todo!()
    }
}
```

## 性能优化

### 并行处理

```rust
// 启用并行评分（默认开启）
let pipeline = CoclusterPipeline::builder()
    .parallel(true)
    // ...
    .build()?;

// 使用rayon设置线程数
std::env::set_var("RAYON_NUM_THREADS", "4");
```

### 缓存和重用

```rust
// 重用聚类器和评分器
let clusterer = Box::new(SVDClusterer::new(5, 0.1));
let scorer = Box::new(PearsonScorer::new(3, 3));

// 对多个矩阵使用相同的配置
for matrix in matrices {
    let pipeline = CoclusterPipeline::builder()
        .with_clusterer(clusterer.clone())
        .with_scorer(scorer.clone())
        .build()?;
    
    let result = pipeline.run(&matrix)?;
    // 处理结果
}
```

## 结果分析

### 访问统计信息

```rust
let result = pipeline.run(&matrix)?;

if let Some(stats) = &result.stats {
    println!("Performance metrics:");
    println!("  Total time: {:?}", stats.total_duration);
    println!("  Clustering: {:?}", stats.clustering_duration);
    println!("  Scoring: {:?}", stats.scoring_duration);
    
    println!("\nQuality metrics:");
    println!("  Score range: {:.3} - {:.3}", 
             stats.score_distribution.min,
             stats.score_distribution.max);
    println!("  Average score: {:.3}", stats.score_distribution.mean);
}
```

### 导出结果

```rust
use std::fs::File;
use std::io::Write;

fn export_results(result: &StepResult, filename: &str) -> std::io::Result<()> {
    let mut file = File::create(filename)?;
    
    writeln!(file, "# Biclustering Results")?;
    writeln!(file, "# Total: {} biclusters\n", result.submatrices.len())?;
    
    for (i, (sub, score)) in result.submatrices.iter()
        .zip(&result.scores)
        .enumerate() {
        
        writeln!(file, "Bicluster {}", i + 1)?;
        writeln!(file, "Score: {:.4}", score)?;
        writeln!(file, "Rows: {:?}", sub.rows)?;
        writeln!(file, "Cols: {:?}", sub.cols)?;
        writeln!(file)?;
    }
    
    Ok(())
}
```

## 错误处理

```rust
use log::{error, warn};

match pipeline.run(&matrix) {
    Ok(result) => {
        // 处理成功结果
    }
    Err(e) => {
        error!("Pipeline failed: {}", e);
        
        // 特定错误处理
        if let Some(io_error) = e.downcast_ref::<std::io::Error>() {
            warn!("IO error occurred: {}", io_error);
        }
    }
}
```

## 测试建议

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pipeline_with_known_data() {
        // 创建已知结构的测试数据
        let mut data = DMatrix::zeros(20, 15);
        // 添加明确的双聚类结构
        for i in 0..10 {
            for j in 0..8 {
                data[(i, j)] = 5.0;
            }
        }
        
        let matrix = Matrix { 
            data, 
            row_labels: vec![], 
            col_labels: vec![] 
        };
        
        let pipeline = CoclusterPipeline::builder()
            .with_clusterer(Box::new(SVDClusterer::new(2, 0.1)))
            .with_scorer(Box::new(PearsonScorer::new(2, 2)))
            .min_score(0.7)
            .build()
            .unwrap();
        
        let result = pipeline.run(&matrix).unwrap();
        
        // 验证找到了预期的双聚类
        assert!(!result.submatrices.is_empty());
        assert!(result.scores[0] > 0.8);
    }
}
```

## 最佳实践

1. **选择合适的聚类器**
   - SVD：适合一般数据，性能好
   - Spectral：适合非线性结构
   - KMeans：适合球形聚类

2. **选择合适的评分器**
   - Pearson：适合线性相关性
   - Exponential：适合紧密聚类
   - Compatibility：适合方差较小的聚类

3. **参数调优**
   - 从宽松的参数开始，逐步调整
   - 使用统计信息指导参数选择
   - 考虑数据特点设置最小尺寸

4. **性能考虑**
   - 大数据集使用并行处理
   - 合理设置最大子矩阵数量
   - 考虑使用采样进行快速预览

## 常见问题

### Q: Pipeline找不到任何双聚类？

A: 检查以下几点：

- 降低`min_score`阈值
- 减小`min_submatrix_size`
- 增加聚类数量参数
- 检查数据是否已归一化

### Q: 运行时间太长？

A: 尝试：

- 启用并行处理
- 减少聚类数量
- 限制最大子矩阵数量
- 对大数据集进行采样

### Q: 内存使用过高？

A: 考虑：

- 分批处理数据
- 使用更小的聚类数量
- 限制收集的统计信息

## Example

```rust
// examples/gene_expression_analysis.rs
use nalgebra::DMatrix;
use fast_cocluster::pipeline::*;
use fast_cocluster::scoring::*;
use fast_cocluster::{Matrix, Submatrix};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::error::Error;
use std::path::Path;

/// 基因表达数据分析示例
/// 展示如何使用Pipeline分析基因表达矩阵，找出共表达的基因组和样本组
fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    
    println!("=== Gene Expression Biclustering Analysis ===\n");
    
    // 1. 加载或生成基因表达数据
    let (matrix, gene_names, sample_names) = load_gene_expression_data()?;
    println!("Loaded gene expression matrix: {} genes × {} samples", 
             gene_names.len(), sample_names.len());
    
    // 2. 数据预处理
    let normalized_matrix = normalize_expression_data(&matrix);
    println!("Data normalized using log2 transformation and standardization");
    
    // 3. 运行双聚类分析
    let biclusters = run_biclustering_analysis(&normalized_matrix)?;
    println!("\nFound {} significant biclusters", biclusters.len());
    
    // 4. 分析结果
    analyze_biclusters(&biclusters, &matrix, &gene_names, &sample_names)?;
    
    // 5. 导出结果
    export_results(&biclusters, &gene_names, &sample_names, "gene_biclusters.txt")?;
    println!("\nResults exported to gene_biclusters.txt");
    
    Ok(())
}

/// 加载基因表达数据（这里使用模拟数据）
fn load_gene_expression_data() -> Result<(Matrix, Vec<String>, Vec<String>), Box<dyn Error>> {
    // 在实际应用中，这里会从文件读取
    // 例如：从CSV或TSV文件加载表达矩阵
    
    // 模拟数据：200个基因，50个样本
    let n_genes = 200;
    let n_samples = 50;
    
    // 生成基因名称
    let gene_names: Vec<String> = (0..n_genes)
        .map(|i| format!("GENE_{:04}", i))
        .collect();
    
    // 生成样本名称
    let sample_names: Vec<String> = (0..n_samples)
        .map(|i| {
            if i < 25 {
                format!("TUMOR_{:02}", i)
            } else {
                format!("NORMAL_{:02}", i - 25)
            }
        })
        .collect();
    
    // 生成表达数据
    let mut data = DMatrix::zeros(n_genes, n_samples);
    
    // 创建几个共表达模块
    // 模块1：癌症相关基因在肿瘤样本中高表达
    for i in 0..30 {
        for j in 0..25 {
            data[(i, j)] = 12.0 + rand::random::<f64>() * 3.0;
        }
    }
    
    // 模块2：细胞周期基因在部分样本中共表达
    for i in 50..80 {
        for j in 10..30 {
            data[(i, j)] = 10.0 + rand::random::<f64>() * 2.5;
        }
    }
    
    // 模块3：代谢基因在正常样本中表达
    for i in 100..130 {
        for j in 25..50 {
            data[(i, j)] = 11.0 + rand::random::<f64>() * 2.0;
        }
    }
    
    // 添加背景表达
    for i in 0..n_genes {
        for j in 0..n_samples {
            if data[(i, j)] == 0.0 {
                data[(i, j)] = 3.0 + rand::random::<f64>() * 4.0;
            }
        }
    }
    
    let matrix = Matrix {
        data,
        row_labels: gene_names.clone(),
        col_labels: sample_names.clone(),
    };
    
    Ok((matrix, gene_names, sample_names))
}

/// 归一化表达数据
fn normalize_expression_data(matrix: &Matrix) -> Matrix {
    let mut normalized = matrix.data.clone();
    
    // 1. Log2转换
    normalized.apply(|x| {
        if *x > 0.0 {
            *x = x.log2();
        }
    });
    
    // 2. 标准化每个基因的表达（按行）
    for i in 0..normalized.nrows() {
        let row = normalized.row(i);
        let mean = row.mean();
        let std = row.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            .sqrt() / (row.len() as f64 - 1.0).sqrt();
        
        if std > 0.0 {
            for j in 0..normalized.ncols() {
                normalized[(i, j)] = (normalized[(i, j)] - mean) / std;
            }
        }
    }
    
    Matrix {
        data: normalized,
        row_labels: matrix.row_labels.clone(),
        col_labels: matrix.col_labels.clone(),
    }
}

/// 运行双聚类分析
fn run_biclustering_analysis(matrix: &Matrix) -> Result<Vec<BiclusterResult>, Box<dyn Error>> {
    // 配置Pipeline
    let config = PipelineConfig {
        min_score: 0.6,           // 较高的相关性阈值
        max_submatrices: 20,      // 最多找20个双聚类
        sort_by_score: true,      // 按分数排序
        min_submatrix_size: (10, 5), // 至少10个基因，5个样本
        collect_stats: true,
        parallel: true,
    };
    
    // 使用组合评分器
    let scorer = CompositeScorer::new()
        .add_scorer(Box::new(PearsonScorer::new(5, 3)), 0.6)     // 重视相关性
        .add_scorer(Box::new(CompatibilityScorer::new(0.5, 0.5)), 0.4); // 兼容性
    
    let pipeline = CoclusterPipeline::builder()
        .with_clusterer(Box::new(SVDClusterer::new(8, 0.1)))
        .with_scorer(Box::new(scorer))
        .with_config(config)
        .build()?;
    
    println!("\nRunning biclustering analysis...");
    let start = std::time::Instant::now();
    let result = pipeline.run(matrix)?;
    println!("Analysis completed in {:?}", start.elapsed());
    
    // 打印统计信息
    if let Some(stats) = &result.stats {
        println!("\nPipeline Statistics:");
        println!("  Initial biclusters: {}", stats.initial_submatrices);
        println!("  After filtering: {}", stats.filtered_submatrices);
        println!("  Score range: {:.3} - {:.3}",
                 stats.score_distribution.min,
                 stats.score_distribution.max);
    }
    
    // 转换为结果结构
    let biclusters: Vec<BiclusterResult> = result.submatrices.into_iter()
        .zip(result.scores)
        .enumerate()
        .map(|(idx, (submatrix, score))| BiclusterResult {
            id: idx + 1,
            submatrix,
            score,
            enrichment: None,
        })
        .collect();
    
    Ok(biclusters)
}

#[derive(Debug, Clone)]
struct BiclusterResult {
    id: usize,
    submatrix: Submatrix,
    score: f64,
    enrichment: Option<EnrichmentResult>,
}

#[derive(Debug, Clone)]
struct EnrichmentResult {
    go_terms: Vec<String>,
    p_value: f64,
}

/// 分析双聚类结果
fn analyze_biclusters(
    biclusters: &[BiclusterResult],
    original_matrix: &Matrix,
    gene_names: &[String],
    sample_names: &[String],
) -> Result<(), Box<dyn Error>> {
    
    println!("\n=== Bicluster Analysis Results ===");
    
    for (i, bc) in biclusters.iter().take(5).enumerate() {
        println!("\nBicluster {} (Score: {:.3}):", bc.id, bc.score);
        println!("  Size: {} genes × {} samples", 
                 bc.submatrix.rows.len(), 
                 bc.submatrix.cols.len());
        
        // 显示基因
        print!("  Genes: ");
        for (j, &gene_idx) in bc.submatrix.rows.iter().take(5).enumerate() {
            if j > 0 { print!(", "); }
            print!("{}", gene_names[gene_idx]);
        }
        if bc.submatrix.rows.len() > 5 {
            print!(" ... ({} more)", bc.submatrix.rows.len() - 5);
        }
        println!();
        
        // 显示样本
        print!("  Samples: ");
        for (j, &sample_idx) in bc.submatrix.cols.iter().take(5).enumerate() {
            if j > 0 { print!(", "); }
            print!("{}", sample_names[sample_idx]);
        }
        if bc.submatrix.cols.len() > 5 {
            print!(" ... ({} more)", bc.submatrix.cols.len() - 5);
        }
        println!();
        
        // 计算平均表达水平
        let mut sum = 0.0;
        let mut count = 0;
        for &row in &bc.submatrix.rows {
            for &col in &bc.submatrix.cols {
                sum += original_matrix.data[(row, col)];
                count += 1;
            }
        }
        let avg_expression = sum / count as f64;
        println!("  Average expression: {:.2}", avg_expression);
        
        // 检查样本类型分布
        let tumor_count = bc.submatrix.cols.iter()
            .filter(|&&idx| sample_names[idx].starts_with("TUMOR"))
            .count();
        let normal_count = bc.submatrix.cols.len() - tumor_count;
        
        println!("  Sample distribution: {} tumor, {} normal", 
                 tumor_count, normal_count);
        
        // 简单的富集分析（模拟）
        if tumor_count > normal_count * 2 {
            println!("  Enrichment: Potentially cancer-related genes");
        } else if normal_count > tumor_count * 2 {
            println!("  Enrichment: Potentially normal tissue genes");
        }
    }
    
    Ok(())
}

/// 导出结果到文件
fn export_results(
    biclusters: &[BiclusterResult],
    gene_names: &[String],
    sample_names: &[String],
    filename: &str,
) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(filename)?;
    
    writeln!(file, "# Gene Expression Biclustering Results")?;
    writeln!(file, "# Generated by fast_cocluster")?;
    writeln!(file, "# Total biclusters: {}", biclusters.len())?;
    writeln!(file)?;
    
    for bc in biclusters {
        writeln!(file, "Bicluster_{}", bc.id)?;
        writeln!(file, "Score: {:.4}", bc.score)?;
        writeln!(file, "Size: {} genes × {} samples", 
                 bc.submatrix.rows.len(), 
                 bc.submatrix.cols.len())?;
        
        // 写入基因列表
        writeln!(file, "Genes:")?;
        for &gene_idx in &bc.submatrix.rows {
            writeln!(file, "\t{}", gene_names[gene_idx])?;
        }
        
        // 写入样本列表
        writeln!(file, "Samples:")?;
        for &sample_idx in &bc.submatrix.cols {
            writeln!(file, "\t{}", sample_names[sample_idx])?;
        }
        
        writeln!(file)?;
    }
    
    Ok(())
}

// 也可以从实际文件加载数据
fn load_from_csv(filename: &str) -> Result<(Matrix, Vec<String>, Vec<String>), Box<dyn Error>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    
    // 读取样本名称（第一行）
    let header = lines.next().ok_or("Empty file")??;
    let sample_names: Vec<String> = header.split('\t').skip(1).map(String::from).collect();
    
    // 读取数据
    let mut gene_names = Vec::new();
    let mut data_vec = Vec::new();
    
    for line in lines {
        let line = line?;
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() > 1 {
            gene_names.push(parts[0].to_string());
            for value_str in parts.iter().skip(1) {
                let value: f64 = value_str.parse()?;
                data_vec.push(value);
            }
        }
    }
    
    let n_genes = gene_names.len();
    let n_samples = sample_names.len();
    let data = DMatrix::from_vec(n_genes, n_samples, data_vec);
    
    let matrix = Matrix {
        data,
        row_labels: gene_names.clone(),
        col_labels: sample_names.clone(),
    };
    
    Ok((matrix, gene_names, sample_names))
}

// 扩展trait实现
impl Matrix {
    pub fn from_csv(path: &Path) -> Result<Self, Box<dyn Error>> {
        // 实现CSV加载逻辑
        todo!()
    }
    
    pub fn to_csv(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        // 实现CSV保存逻辑
        todo!()
    }
}
```
