use fast_cocluster::tensor3d::*;
use fast_cocluster::tensor3d_scoring::*;
use fast_cocluster::tucker_decomposition::*;
use ndarray::Array3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    println!("🔥 Fast Cocluster - 3D张量Co-clustering完整演示 🔥\n");
    
    // 1. 创建3个不同类型的测试张量
    println!("📊 1. 创建测试数据集");
    let datasets = create_test_datasets();
    
    for (name, tensor) in &datasets {
        println!("   📈 {}: 形状{:?}, 范数{:.2}", 
                 name, tensor.shape(), tensor.frobenius_norm());
    }
    
    // 2. 测试Tucker分解
    println!("\n🔍 2. Tucker分解分析");
    test_tucker_decomposition(&datasets)?;
    
    // 3. 测试不同的评分器
    println!("\n⭐ 3. 3D张量评分器对比");
    test_scoring_methods(&datasets)?;
    
    // 4. 演示完整的co-clustering流程
    println!("\n🎯 4. 完整Co-clustering演示");
    demonstrate_full_cocluster(&datasets)?;
    
    // 5. 性能分析
    println!("\n⚡ 5. 性能分析");
    performance_analysis()?;
    
    println!("\n✅ 演示完成! 3D张量Co-clustering已成功实现!");
    
    Ok(())
}

/// 创建测试数据集
fn create_test_datasets() -> Vec<(&'static str, Tensor3D<f64>)> {
    vec![
        ("基因表达张量", create_gene_expression_tensor()),
        ("推荐系统张量", create_recommendation_tensor()),
        ("时空数据张量", create_spatiotemporal_tensor()),
    ]
}

/// 创建模拟基因表达张量 (基因 × 条件 × 时间点)
fn create_gene_expression_tensor() -> Tensor3D<f64> {
    let shape = [50, 20, 10]; // 50个基因, 20个条件, 10个时间点
    let mut data = Array3::zeros(shape);
    
    // 创建基因表达模块
    // 模块1: 应激反应基因组 (前10个基因, 前5个条件, 前5个时间点)
    for i in 0..10 {
        for j in 0..5 {
            for k in 0..5 {
                data[[i, j, k]] = 8.0 + rand::random::<f64>() * 2.0; // 高表达
            }
        }
    }
    
    // 模块2: 发育基因组 (基因20-30, 条件10-15, 时间点5-10)
    for i in 20..30 {
        for j in 10..15 {
            for k in 5..10 {
                data[[i, j, k]] = 6.0 + rand::random::<f64>() * 1.5;
            }
        }
    }
    
    // 模块3: 代谢基因组 (基因35-45, 条件15-20, 所有时间点)
    for i in 35..45 {
        for j in 15..20 {
            for k in 0..shape[2] {
                data[[i, j, k]] = 4.0 + rand::random::<f64>() * 1.0;
            }
        }
    }
    
    // 添加背景噪声
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                if data[[i, j, k]] == 0.0 {
                    data[[i, j, k]] = 1.0 + rand::random::<f64>() * 0.5;
                }
            }
        }
    }
    
    let gene_labels: Vec<String> = (0..shape[0]).map(|i| format!("Gene_{:03}", i)).collect();
    let condition_labels: Vec<String> = (0..shape[1]).map(|i| format!("Condition_{:02}", i)).collect();
    let time_labels: Vec<String> = (0..shape[2]).map(|i| format!("Time_{:02}h", i)).collect();
    
    Tensor3D::new(data, gene_labels, condition_labels, time_labels).unwrap()
}

/// 创建模拟推荐系统张量 (用户 × 物品 × 上下文)
fn create_recommendation_tensor() -> Tensor3D<f64> {
    let shape = [30, 25, 8]; // 30个用户, 25个物品, 8个上下文
    let mut data = Array3::zeros(shape);
    
    // 用户群体1: 年轻用户喜欢科技产品
    for i in 0..10 {
        for j in 0..8 { // 科技产品
            for k in 0..4 { // 工作日上下文
                data[[i, j, k]] = 4.0 + rand::random::<f64>() * 1.0;
            }
        }
    }
    
    // 用户群体2: 中年用户喜欢家居产品
    for i in 10..20 {
        for j in 8..16 { // 家居产品
            for k in 4..8 { // 周末/居家上下文
                data[[i, j, k]] = 4.5 + rand::random::<f64>() * 0.8;
            }
        }
    }
    
    // 用户群体3: 老年用户喜欢健康产品
    for i in 20..30 {
        for j in 16..25 { // 健康产品
            for k in 2..6 { // 日常上下文
                data[[i, j, k]] = 3.8 + rand::random::<f64>() * 1.2;
            }
        }
    }
    
    // 添加随机噪声评分
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                if data[[i, j, k]] == 0.0 {
                    data[[i, j, k]] = 1.0 + rand::random::<f64>() * 2.0;
                }
            }
        }
    }
    
    let user_labels: Vec<String> = (0..shape[0]).map(|i| format!("User_{:03}", i)).collect();
    let item_labels: Vec<String> = (0..shape[1]).map(|i| format!("Item_{:03}", i)).collect();
    let context_labels: Vec<String> = ["Work_Morning", "Work_Afternoon", "Commute", "Evening", 
                                      "Weekend_Morning", "Weekend_Afternoon", "Social", "Travel"]
                                      .iter().map(|s| s.to_string()).collect();
    
    Tensor3D::new(data, user_labels, item_labels, context_labels).unwrap()
}

/// 创建时空数据张量 (传感器 × 地点 × 时间)
fn create_spatiotemporal_tensor() -> Tensor3D<f64> {
    let shape = [15, 12, 24]; // 15个传感器, 12个地点, 24小时
    let mut data = Array3::zeros(shape);
    
    // 模拟日常模式: 交通传感器在早晚高峰期高活跃
    for i in 0..5 { // 交通传感器
        for j in 0..shape[1] {
            // 早高峰 (7-9点)
            for k in 7..9 {
                data[[i, j, k]] = 80.0 + rand::random::<f64>() * 20.0;
            }
            // 晚高峰 (17-19点)
            for k in 17..19 {
                data[[i, j, k]] = 85.0 + rand::random::<f64>() * 15.0;
            }
            // 其他时间
            for k in 0..shape[2] {
                if data[[i, j, k]] == 0.0 {
                    data[[i, j, k]] = 20.0 + rand::random::<f64>() * 10.0;
                }
            }
        }
    }
    
    // 环境传感器: 白天高活跃
    for i in 5..10 {
        for j in 0..shape[1] {
            for k in 6..18 { // 白天6-18点
                data[[i, j, k]] = 60.0 + rand::random::<f64>() * 20.0;
            }
            for k in 0..shape[2] {
                if data[[i, j, k]] == 0.0 {
                    data[[i, j, k]] = 30.0 + rand::random::<f64>() * 10.0;
                }
            }
        }
    }
    
    // 安全传感器: 夜间高活跃
    for i in 10..15 {
        for j in 0..shape[1] {
            for k in 22..24 { // 22-24点
                data[[i, j, k]] = 70.0 + rand::random::<f64>() * 15.0;
            }
            for k in 0..6 { // 0-6点
                data[[i, j, k]] = 75.0 + rand::random::<f64>() * 10.0;
            }
            for k in 0..shape[2] {
                if data[[i, j, k]] == 0.0 {
                    data[[i, j, k]] = 40.0 + rand::random::<f64>() * 15.0;
                }
            }
        }
    }
    
    let sensor_labels: Vec<String> = (0..shape[0]).map(|i| format!("Sensor_{:02}", i)).collect();
    let location_labels: Vec<String> = (0..shape[1]).map(|i| format!("Location_{:02}", i)).collect();
    let time_labels: Vec<String> = (0..shape[2]).map(|i| format!("{:02}:00", i)).collect();
    
    Tensor3D::new(data, sensor_labels, location_labels, time_labels).unwrap()
}

/// 测试Tucker分解
fn test_tucker_decomposition(datasets: &[(&str, Tensor3D<f64>)]) -> Result<(), Box<dyn std::error::Error>> {
    let ranks = [2, 3, 4];
    
    for (name, tensor) in datasets {
        println!("   🔍 分析{}", name);
        
        for &rank in &ranks {
            let tucker_rank = TuckerRank::uniform(rank);
            let mut config = TuckerConfig::default();
            config.ranks = tucker_rank;
            config.max_iterations = 15;
            
            let decomposer = TuckerDecomposer::new(config);
            match decomposer.decompose(tensor) {
                Ok(decomposition) => {
                    let relative_error = decomposition.reconstruction_error / tensor.frobenius_norm();
                    println!("     ✅ Rank={}: 相对误差={:.4}", rank, relative_error);
                },
                Err(e) => println!("     ❌ Rank={}: 失败 - {}", rank, e),
            }
        }
    }
    Ok(())
}

/// 测试评分方法
fn test_scoring_methods(datasets: &[(&str, Tensor3D<f64>)]) -> Result<(), Box<dyn std::error::Error>> {
    for (name, tensor) in datasets {
        println!("   📊 评分{}", name);
        
        // 创建不同大小的测试子空间
        let subspaces = create_test_subspaces(tensor);
        
        // 不同的评分器
        let scorers: Vec<(Box<dyn TensorScorer>, &str)> = vec![
            (Box::new(TuckerScorer::with_uniform_rank(2)), "Tucker(2)"),
            (Box::new(TuckerScorer::with_uniform_rank(3)), "Tucker(3)"),
            (Box::new(DensityScorer::new(1.0)), "密度"),
            (Box::new(VarianceScorer::new(5.0)), "方差"),
        ];
        
        for (i, subspace) in subspaces.iter().enumerate() {
            let shape = subspace.shape();
            println!("     子空间{}: {}×{}×{}", i+1, shape[0], shape[1], shape[2]);
            
            for (scorer, name) in &scorers {
                let score = scorer.score(tensor, subspace);
                println!("       {}: {:.4}", name, score);
            }
        }
    }
    Ok(())
}

/// 创建测试子空间
fn create_test_subspaces(tensor: &Tensor3D<f64>) -> Vec<TensorSubspace> {
    let shape = tensor.shape();
    let mut subspaces = Vec::new();
    
    // 小子空间
    if let Some(sub) = TensorSubspace::new(tensor, 
                                          vec![0, 1, 2], 
                                          vec![0, 1, 2], 
                                          vec![0, 1, 2]) {
        subspaces.push(sub);
    }
    
    // 中等子空间
    if let Some(sub) = TensorSubspace::new(tensor, 
                                          (0..5.min(shape[0])).collect(),
                                          (0..5.min(shape[1])).collect(),
                                          (0..3.min(shape[2])).collect()) {
        subspaces.push(sub);
    }
    
    // 大子空间
    if let Some(sub) = TensorSubspace::new(tensor, 
                                          (0..10.min(shape[0])).collect(),
                                          (0..8.min(shape[1])).collect(),
                                          (0..5.min(shape[2])).collect()) {
        subspaces.push(sub);
    }
    
    subspaces
}

/// 演示完整co-clustering
fn demonstrate_full_cocluster(datasets: &[(&str, Tensor3D<f64>)]) -> Result<(), Box<dyn std::error::Error>> {
    for (name, tensor) in datasets {
        println!("   🎯 处理{}", name);
        
        // 简化的co-clustering方法
        let result = simple_tensor_cocluster(tensor, 3.0)?;
        
        println!("     ✅ 发现{}个高质量子空间", result.len());
        for (i, (subspace, score)) in result.iter().take(3).enumerate() {
            let shape = subspace.shape();
            println!("       #{}: {}×{}×{} (质量分数: {:.3})", 
                     i+1, shape[0], shape[1], shape[2], score);
        }
    }
    Ok(())
}

/// 简化的张量co-clustering算法
fn simple_tensor_cocluster(tensor: &Tensor3D<f64>, threshold: f64) -> Result<Vec<(TensorSubspace, f64)>, Box<dyn std::error::Error>> {
    let shape = tensor.shape();
    let mut results = Vec::new();
    
    // 使用滑动窗口寻找高密度区域
    let window_size = (3, 3, 2);
    
    for i in 0..=(shape[0].saturating_sub(window_size.0)) {
        for j in 0..=(shape[1].saturating_sub(window_size.1)) {
            for k in 0..=(shape[2].saturating_sub(window_size.2)) {
                let rows: Vec<usize> = (i..i+window_size.0).collect();
                let cols: Vec<usize> = (j..j+window_size.1).collect();
                let depths: Vec<usize> = (k..k+window_size.2).collect();
                
                if let Some(subspace) = TensorSubspace::new(tensor, rows, cols, depths) {
                    // 计算平均值作为质量分数
                    let sub_data = subspace.extract_data();
                    let mean = sub_data.mean().unwrap_or(0.0);
                    
                    if mean > threshold {
                        results.push((subspace, mean));
                    }
                }
            }
        }
    }
    
    // 按分数排序
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    // 去重重叠度过高的子空间
    let mut filtered_results = Vec::new();
    for (subspace, score) in results {
        let mut should_add = true;
        
        for (existing_subspace, _) in &filtered_results {
            if subspaces_overlap(&subspace, existing_subspace, 0.5) {
                should_add = false;
                break;
            }
        }
        
        if should_add {
            filtered_results.push((subspace, score));
        }
        
        if filtered_results.len() >= 10 {
            break;
        }
    }
    
    Ok(filtered_results)
}

/// 检查两个子空间是否重叠
fn subspaces_overlap(sub1: &TensorSubspace, sub2: &TensorSubspace, threshold: f64) -> bool {
    let overlap1 = sub1.mode1_indices.iter().filter(|&x| sub2.mode1_indices.contains(x)).count();
    let overlap2 = sub1.mode2_indices.iter().filter(|&x| sub2.mode2_indices.contains(x)).count();
    let overlap3 = sub1.mode3_indices.iter().filter(|&x| sub2.mode3_indices.contains(x)).count();
    
    let max_overlap1 = sub1.mode1_indices.len().min(sub2.mode1_indices.len());
    let max_overlap2 = sub1.mode2_indices.len().min(sub2.mode2_indices.len());
    let max_overlap3 = sub1.mode3_indices.len().min(sub2.mode3_indices.len());
    
    if max_overlap1 == 0 || max_overlap2 == 0 || max_overlap3 == 0 {
        return false;
    }
    
    let overlap_ratio1 = overlap1 as f64 / max_overlap1 as f64;
    let overlap_ratio2 = overlap2 as f64 / max_overlap2 as f64;
    let overlap_ratio3 = overlap3 as f64 / max_overlap3 as f64;
    
    overlap_ratio1 > threshold && overlap_ratio2 > threshold && overlap_ratio3 > threshold
}

/// 性能分析
fn performance_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let sizes = [(10, 8, 6), (20, 15, 10), (30, 25, 15)];
    
    for &size in &sizes {
        println!("   📏 张量大小: {:?}", size);
        
        let tensor = Tensor3D::random([size.0, size.1, size.2]);
        
        // 测试Tucker分解性能
        let start = std::time::Instant::now();
        let tucker_rank = TuckerRank::uniform(3);
        let mut config = TuckerConfig::default();
        config.ranks = tucker_rank;
        config.max_iterations = 10;
        
        let decomposer = TuckerDecomposer::new(config);
        match decomposer.decompose(&tensor) {
            Ok(_) => {
                let duration = start.elapsed();
                println!("     ⏱️  Tucker分解: {:?}", duration);
            },
            Err(e) => println!("     ❌ Tucker分解失败: {}", e),
        }
        
        // 测试评分性能
        if let Some(subspace) = TensorSubspace::new(&tensor, 
                                                   vec![0, 1, 2], 
                                                   vec![0, 1, 2], 
                                                   vec![0, 1, 2]) {
            let start = std::time::Instant::now();
            let scorer = TuckerScorer::with_uniform_rank(2);
            let _score = scorer.score(&tensor, &subspace);
            let duration = start.elapsed();
            println!("     ⏱️  Tucker评分: {:?}", duration);
        }
    }
    
    Ok(())
}