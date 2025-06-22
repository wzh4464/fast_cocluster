use fast_cocluster::cocluster::Coclusterer;
use fast_cocluster::modular_cocluster::*;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    println!("=== 原子化 vs 模块化 Co-clustering 比较 ===\n");
    
    // 创建测试数据
    let test_matrix: Array2<f64> = Array2::random((100, 60), Uniform::new(0.0, 10.0));
    println!("创建了 {}×{} 的测试矩阵\n", test_matrix.nrows(), test_matrix.ncols());
    
    // 方法1: 原子化方法 (原始实现)
    println!("🔴 原子化方法 (原始实现):");
    let start = Instant::now();
    let mut atomic_coclusterer = Coclusterer::new(test_matrix.clone(), 5, 0.1);
    let atomic_result = atomic_coclusterer.cocluster()?;
    let atomic_time = start.elapsed();
    
    println!("   ✅ 执行时间: {:?}", atomic_time);
    println!("   ✅ 结果长度: {}", atomic_result.len());
    println!("   ✅ 聚类分布: {:?}", count_clusters(&atomic_result, 5));
    println!("   ❌ 固定算法: SVD + K-means");
    println!("   ❌ 不可定制归一化");
    println!("   ❌ 不可替换组件");
    
    // 方法2: 模块化方法 - 默认配置 (等效于原子化)
    println!("\n🟢 模块化方法 - 默认配置 (等效算法):");
    let start = Instant::now();
    let mut modular_default = ModularCoclusterer::with_defaults(test_matrix.clone(), 5);
    let modular_default_result = modular_default.cocluster()?;
    let modular_default_time = start.elapsed();
    
    println!("   ✅ 执行时间: {:?}", modular_default_time);
    println!("   ✅ 结果长度: {}", modular_default_result.len());
    println!("   ✅ 聚类分布: {:?}", count_clusters(&modular_default_result, 5));
    println!("   ✅ 可替换组件");
    println!("   ✅ 可定制配置");
    
    // 方法3: 模块化方法 - Z-score归一化
    println!("\n🟡 模块化方法 - Z-score归一化:");
    let start = Instant::now();
    let mut modular_zscore = ModularCoclusterer::with_zscore(test_matrix.clone(), 5);
    let modular_zscore_result = modular_zscore.cocluster()?;
    let modular_zscore_time = start.elapsed();
    
    println!("   ✅ 执行时间: {:?}", modular_zscore_time);
    println!("   ✅ 结果长度: {}", modular_zscore_result.len());
    println!("   ✅ 聚类分布: {:?}", count_clusters(&modular_zscore_result, 5));
    println!("   ✅ 改进的归一化方法");
    
    // 方法4: 模块化方法 - 加权特征
    println!("\n🔵 模块化方法 - 加权特征组合:");
    let start = Instant::now();
    let mut modular_weighted = ModularCoclusterer::with_weighted_features(test_matrix.clone(), 5, 0.8, 0.2);
    let modular_weighted_result = modular_weighted.cocluster()?;
    let modular_weighted_time = start.elapsed();
    
    println!("   ✅ 执行时间: {:?}", modular_weighted_time);
    println!("   ✅ 结果长度: {}", modular_weighted_result.len());
    println!("   ✅ 聚类分布: {:?}", count_clusters(&modular_weighted_result, 5));
    println!("   ✅ 行列特征加权 (行:0.8, 列:0.2)");
    
    // 方法5: 完全自定义的模块化方法
    println!("\n🟣 模块化方法 - 完全自定义:");
    let start = Instant::now();
    let mut modular_custom = ModularCoclustererBuilder::new()
        .matrix(test_matrix.clone())
        .k(5)
        .normalizer(Box::new(ZScoreNormalizer))
        .reducer(Box::new(SVDReducer))
        .combiner(Box::new(WeightedCombiner { row_weight: 0.6, col_weight: 0.4 }))
        .assigner(Box::new(KMeansAssigner))
        .build()?;
    
    let modular_custom_result = modular_custom.cocluster()?;
    let modular_custom_time = start.elapsed();
    
    println!("   ✅ 执行时间: {:?}", modular_custom_time);
    println!("   ✅ 结果长度: {}", modular_custom_result.len());
    println!("   ✅ 聚类分布: {:?}", count_clusters(&modular_custom_result, 5));
    println!("   ✅ Z-score + 加权特征 + 完全可定制");
    
    // 性能比较
    println!("\n=== 性能比较 ===");
    println!("原子化方法:     {:>8.2?}", atomic_time);
    println!("模块化默认:     {:>8.2?} (开销: {:.1}%)", 
             modular_default_time, 
             (modular_default_time.as_nanos() as f64 / atomic_time.as_nanos() as f64 - 1.0) * 100.0);
    println!("模块化Z-score:  {:>8.2?}", modular_zscore_time);
    println!("模块化加权:     {:>8.2?}", modular_weighted_time);
    println!("模块化自定义:   {:>8.2?}", modular_custom_time);
    
    // 结果质量比较
    println!("\n=== 聚类质量比较 ===");
    println!("方法              | 聚类分布               | 最大聚类 | 最小聚类");
    println!("------------------|------------------------|----------|----------");
    
    let atomic_dist = count_clusters(&atomic_result, 5);
    let default_dist = count_clusters(&modular_default_result, 5);
    let zscore_dist = count_clusters(&modular_zscore_result, 5);
    let weighted_dist = count_clusters(&modular_weighted_result, 5);
    let custom_dist = count_clusters(&modular_custom_result, 5);
    
    println!("原子化           | {:?} | {:>8} | {:>8}", atomic_dist, atomic_dist.iter().max().unwrap(), atomic_dist.iter().min().unwrap());
    println!("模块化默认       | {:?} | {:>8} | {:>8}", default_dist, default_dist.iter().max().unwrap(), default_dist.iter().min().unwrap());
    println!("模块化Z-score    | {:?} | {:>8} | {:>8}", zscore_dist, zscore_dist.iter().max().unwrap(), zscore_dist.iter().min().unwrap());
    println!("模块化加权       | {:?} | {:>8} | {:>8}", weighted_dist, weighted_dist.iter().max().unwrap(), weighted_dist.iter().min().unwrap());
    println!("模块化自定义     | {:?} | {:>8} | {:>8}", custom_dist, custom_dist.iter().max().unwrap(), custom_dist.iter().min().unwrap());
    
    // 可扩展性展示
    println!("\n=== 可扩展性优势 ===");
    demonstrate_extensibility(test_matrix)?;
    
    Ok(())
}

/// 统计每个聚类的元素数量
fn count_clusters(assignments: &[usize], k: usize) -> Vec<usize> {
    let mut counts = vec![0; k];
    for &assignment in assignments {
        if assignment < k {
            counts[assignment] += 1;
        }
    }
    counts
}

/// 演示模块化方法的可扩展性
fn demonstrate_extensibility(test_matrix: Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    println!("✨ 创建自定义归一化器:");
    
    // 自定义归一化器 - Min-Max归一化
    struct MinMaxNormalizer;
    impl MatrixNormalizer for MinMaxNormalizer {
        fn normalize(&self, matrix: &nalgebra::DMatrix<f64>) -> nalgebra::DMatrix<f64> {
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
    
    // 自定义特征组合器 - 平均组合
    struct AverageCombiner;
    impl FeatureCombiner for AverageCombiner {
        fn combine(&self, u: &nalgebra::DMatrix<f64>, v: &nalgebra::DMatrix<f64>) -> nalgebra::DMatrix<f64> {
            let rows = u.nrows();
            let cols = u.ncols();
            
            // 将行列特征平均组合
            nalgebra::DMatrix::from_fn(rows + v.nrows(), cols, |r, c| {
                if r < rows {
                    u[(r, c)] * 0.5
                } else {
                    v[(r - rows, c)] * 0.5
                }
            })
        }
    }
    
    let start = Instant::now();
    let mut custom_coclusterer = ModularCoclustererBuilder::new()
        .matrix(test_matrix)
        .k(4)
        .normalizer(Box::new(MinMaxNormalizer))
        .combiner(Box::new(AverageCombiner))
        .build()?;
    
    let custom_result = custom_coclusterer.cocluster()?;
    let custom_time = start.elapsed();
    
    println!("   ✅ 自定义Min-Max归一化 + 平均特征组合");
    println!("   ✅ 执行时间: {:?}", custom_time);
    println!("   ✅ 聚类分布: {:?}", count_clusters(&custom_result, 4));
    println!("   🎯 这在原子化方法中是不可能的!");
    
    Ok(())
}