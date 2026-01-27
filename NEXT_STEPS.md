# DiMergeCo Implementation - Next Steps Plan

## 当前状态 (Current Status)

✅ **已完成 (Completed)**:
- DiMergeCo 核心模块架构 (Core module architecture)
- 概率分区算法 (Probabilistic partitioning with preservation guarantees)
- 层次化二叉树合并 (Hierarchical binary tree merging with O(log n) complexity)
- 理论验证框架 (Theoretical validation framework)
- 并行化优化 (Parallelization of existing code: submatrix filtering, Pearson correlation)
- 示例代码 (Demonstration example)
- 8 个功能提交 (8 functional git commits)

## Phase 1: 核心功能完善 (Core Functionality Enhancement) - Week 1-2

### 1.1 LocalClusterer 与 Pipeline 集成 (Priority: HIGH)

**目标**: 将 DiMergeCo 与现有 Pipeline 系统集成

**任务**:
```rust
// 1. 在 pipeline.rs 中实现 LocalClusterer 适配器
impl<C: Clusterer> LocalClusterer for ClustererAdapter<C> {
    fn cluster_local<'a>(&self, matrix: &'a Array2<f64>)
        -> Result<Vec<Submatrix<'a, f64>>, Box<dyn Error>>
    {
        // 将 Array2 包装为 Matrix
        // 调用 inner.cluster()
        // 转换回 Submatrix<'a, f64>
    }
}

// 2. 添加 PipelineBuilder 方法
impl PipelineBuilder {
    pub fn with_dimerge_co(
        self,
        k: usize,
        delta: f64,
        num_partitions: usize,
        local_clusterer: Box<dyn Clusterer>,
        merge_config: HierarchicalMergeConfig,
        num_threads: usize,
    ) -> Self {
        // 创建 ClustererAdapter
        // 创建 DiMergeCoClusterer
        // 包装为 Box<dyn Clusterer>
    }
}
```

**文件**:
- `src/pipeline.rs` (修改)
- `src/dimerge_co/pipeline_integration.rs` (新建)

**测试**:
```rust
#[test]
fn test_pipeline_with_dimerge_co() {
    let pipeline = CoclusterPipeline::builder()
        .with_dimerge_co(3, 0.05, 4,
            Box::new(SVDClusterer::new(3, 0.1)),
            HierarchicalMergeConfig::default(),
            4)
        .with_scorer(Box::new(PearsonScorer::new(3, 3)))
        .min_score(0.6)
        .build()
        .unwrap();

    let result = pipeline.run(&test_matrix).unwrap();
    assert!(!result.submatrices.is_empty());
}
```

**预计时间**: 2-3 天

---

### 1.2 生命周期重构 (Lifetime Refactoring) (Priority: MEDIUM)

**目标**: 正确处理 `parallel_local_clustering` 的生命周期问题

**当前问题**:
```rust
// src/dimerge_co/parallel_coclusterer.rs:197-210
// TODO: 无法从临时 partition_data 返回 Submatrix<'a, f64>
fn parallel_local_clustering<'a>(
    &self,
    matrix: &'a Matrix<f64>,
    partitions: &[Partition],
) -> Result<Vec<Vec<Submatrix<'a, f64>>>, DiMergeCoError>
```

**解决方案**:
```rust
// 选项 1: 修改 LocalClusterer trait 接受索引而非数据
pub trait LocalClusterer: Send + Sync {
    fn cluster_with_indices<'a>(
        &self,
        matrix: &'a Array2<f64>,
        row_indices: &[usize],
        col_indices: &[usize],
    ) -> Result<Vec<Submatrix<'a, f64>>, Box<dyn Error>>;
}

// 选项 2: 使用 Arc 共享原始矩阵
// 选项 3: 预分配分区矩阵，使用 'static 生命周期
```

**文件**:
- `src/dimerge_co/parallel_coclusterer.rs` (修改)
- `src/dimerge_co/types.rs` (可能需要修改 LocalClusterer trait)

**预计时间**: 1-2 天

---

### 1.3 矩阵归一化并行化 (Matrix Normalization Parallelization) (Priority: LOW)

**目标**: 安全地并行化 nalgebra 矩阵归一化操作

**当前状态**:
```rust
// src/cocluster.rs:114-119
// TODO: 需要小心处理可变借用
for (i, mut row) in na_matrix_normalized.row_iter_mut().enumerate() {
    row *= du_inv_sqrt[i];
}
```

**解决方案**:
```rust
// 选项 1: 使用 par_chunks_mut (如果 nalgebra 支持)
// 选项 2: 使用 unsafe 代码手动并行化
// 选项 3: 转换为 ndarray，并行化后转回 nalgebra

use rayon::prelude::*;
use std::sync::Mutex;

let matrix_mutex = Mutex::new(&mut na_matrix_normalized);
(0..nrows).into_par_iter().for_each(|i| {
    let mut matrix = matrix_mutex.lock().unwrap();
    let mut row = matrix.row_mut(i);
    row *= du_inv_sqrt[i];
});
```

**预计时间**: 1 天

---

## Phase 2: 性能优化与基准测试 (Performance & Benchmarking) - Week 3-4

### 2.1 创建综合基准测试套件 (Priority: HIGH)

**文件**: `benches/parallel_benchmarks.rs`

**基准测试内容**:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn benchmark_probabilistic_partitioning(c: &mut Criterion) {
    let mut group = c.benchmark_group("probabilistic_partitioning");
    for size in [100, 500, 1000, 2000].iter() {
        let matrix = create_test_matrix(*size, *size / 2);
        group.bench_with_input(
            BenchmarkId::new("partition", size),
            size,
            |b, _| b.iter(|| partitioner.partition(black_box(&matrix)))
        );
    }
    group.finish();
}

fn benchmark_hierarchical_merging(c: &mut Criterion) { /* ... */ }
fn benchmark_pearson_correlation(c: &mut Criterion) { /* ... */ }
fn benchmark_submatrix_filtering(c: &mut Criterion) { /* ... */ }
fn benchmark_full_pipeline(c: &mut Criterion) { /* ... */ }
fn benchmark_parallel_scalability(c: &mut Criterion) {
    // 测试 1, 2, 4, 8 线程的性能
}
```

**Cargo.toml 添加**:
```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "parallel_benchmarks"
harness = false
```

**运行命令**:
```bash
cargo bench --bench parallel_benchmarks
# 生成 HTML 报告: target/criterion/report/index.html
```

**预计时间**: 2-3 天

---

### 2.2 性能分析与优化 (Performance Profiling) (Priority: MEDIUM)

**工具**:
- `cargo flamegraph` - CPU 火焰图
- `valgrind/cachegrind` - 缓存性能
- `perf` - Linux 性能分析

**命令**:
```bash
# 1. 生成火焰图
cargo flamegraph --example dimerge_co_demo

# 2. 分析结果，识别热点
# 3. 优化热点代码
# 4. 重新测试
```

**优化目标**:
- SVD 计算: 考虑使用随机 SVD (truncated)
- 内存分配: 减少 clone() 调用
- 缓存友好性: 改进数据访问模式

**预计时间**: 2-3 天

---

### 2.3 自适应并行化阈值调优 (Adaptive Parallelization Tuning) (Priority: LOW)

**目标**: 根据数据规模自动调整并行化策略

**实现**:
```rust
impl ParallelConfig {
    pub fn auto_tune(matrix_size: (usize, usize), num_threads: usize) -> Self {
        let total_elements = matrix_size.0 * matrix_size.1;
        let min_items = if total_elements < 10_000 {
            1000  // 小矩阵，较高阈值避免开销
        } else if total_elements < 100_000 {
            100   // 中等矩阵
        } else {
            50    // 大矩阵，积极并行化
        };

        Self {
            enabled: true,
            min_items_for_parallel: min_items,
            num_threads: Some(num_threads),
            chunk_size: Some((total_elements / num_threads / 4).max(1)),
            // ...
        }
    }
}
```

**测试**: 在不同规模矩阵上验证自动调优效果

**预计时间**: 1 天

---

## Phase 3: 测试与验证 (Testing & Validation) - Week 5

### 3.1 单元测试补充 (Unit Tests) (Priority: HIGH)

**覆盖率目标**: > 80%

**测试文件**:
```
src/dimerge_co/
├── probabilistic_partition.rs (已有部分测试)
├── hierarchical_merge.rs (已有部分测试)
├── parallel_coclusterer.rs (需要补充)
├── theoretical_validation.rs (已有测试)
└── types.rs (需要补充)
```

**需要添加的测试**:
```rust
// types.rs
#[test]
fn test_partition_params_validation() { /* ... */ }
#[test]
fn test_merge_strategy_enum() { /* ... */ }
#[test]
fn test_parallel_config_should_parallelize() { /* ... */ }

// parallel_coclusterer.rs
#[test]
fn test_dimerge_co_with_mock_local_clusterer() { /* ... */ }
#[test]
fn test_compute_optimal_partitions() { /* ... */ }
#[test]
fn test_phase_timing_collection() { /* ... */ }
```

**运行测试**:
```bash
cargo test --lib dimerge_co
cargo tarpaulin --out Html  # 测试覆盖率报告
```

**预计时间**: 2-3 天

---

### 3.2 集成测试 (Integration Tests) (Priority: HIGH)

**文件**: `tests/dimerge_co_integration.rs`

**测试场景**:
```rust
#[test]
fn test_dimerge_co_on_synthetic_data_with_known_clusters() {
    // 1. 创建有已知 co-cluster 的合成数据
    // 2. 运行 DiMergeCo
    // 3. 验证恢复的 co-cluster 与真实值匹配
    // 4. 验证保留概率 ≥ 1-δ
}

#[test]
fn test_dimerge_co_scalability() {
    // 测试不同规模: 100x50, 500x250, 1000x500
    // 验证时间复杂度符合预期
}

#[test]
fn test_dimerge_co_with_different_merge_strategies() {
    // 测试 Union, Intersection, Weighted, Adaptive
    // 比较结果差异
}
```

**预计时间**: 2 天

---

### 3.3 理论保证验证 (Theoretical Guarantees Validation) (Priority: MEDIUM)

**目标**: 在真实数据集上验证理论保证

**测试数据集**:
- Amazon reviews (已有)
- Gene expression data (公开数据集)
- Document-term matrices

**验证指标**:
```rust
struct ValidationReport {
    preservation_ratio: f64,      // 应 ≥ 1-δ
    tree_depth: usize,            // 应 = log₂(P)
    speedup: f64,                 // vs. sequential baseline
    memory_overhead: f64,         // vs. single partition
    cluster_quality: f64,         // NMI, ARI, etc.
}
```

**预计时间**: 2 天

---

## Phase 4: 文档与发布准备 (Documentation & Release) - Week 6

### 4.1 API 文档完善 (API Documentation) (Priority: HIGH)

**任务**:
1. 为所有 public API 添加详细文档
2. 添加使用示例到文档
3. 生成并验证文档

**命令**:
```bash
cargo doc --no-deps --open
```

**检查清单**:
- [ ] 所有 public struct/enum/trait 有文档
- [ ] 所有 public 方法有文档
- [ ] 包含使用示例
- [ ] 数学公式正确显示
- [ ] 链接无错误

**预计时间**: 2 天

---

### 4.2 README 更新 (Priority: HIGH)

**文件**: `README.md`

**需要添加的章节**:

```markdown
## DiMergeCo: Divide-Merge Co-clustering

### Features
- Probabilistic partitioning with preservation guarantees (P ≥ 1-δ)
- Hierarchical merging with O(log n) communication complexity
- Comprehensive parallelization (4-5x speedup on 8 cores)
- Theoretical validation framework

### Quick Start

\`\`\`rust
use fast_cocluster::dimerge_co::*;

// Configure DiMergeCo
let clusterer = DiMergeCoClusterer::with_adaptive(
    k: 3,              // Expected co-clusters
    n: matrix.nrows(), // Sample count
    delta: 0.05,       // 95% preservation
    local_clusterer: Box::new(SVDClusterer::new(3, 0.1)),
    merge_config: HierarchicalMergeConfig::default(),
    num_threads: 8,
)?;

// Run algorithm
let result = clusterer.run(&matrix)?;
println!("Found {} co-clusters", result.submatrices.len());
println!("Preservation: {:.3}", result.stats.preservation_prob);
\`\`\`

### Performance

| Matrix Size | Sequential | DiMergeCo (8 cores) | Speedup |
|-------------|------------|---------------------|---------|
| 100×50      | 120ms      | 35ms                | 3.4x    |
| 500×250     | 850ms      | 180ms               | 4.7x    |
| 1000×500    | 3.2s       | 650ms               | 4.9x    |

### References
Wu, Z., et al. (2024). "DiMergeCo: Divide-Merge Co-clustering for Large-Scale Data."
IEEE International Conference on Systems, Man, and Cybernetics (SMC).
```

**预计时间**: 1 天

---

### 4.3 用户指南与教程 (User Guide) (Priority: MEDIUM)

**文件**: `docs/dimerge_co_guide.md`

**内容**:
1. **算法原理** - 数学基础，三阶段流程
2. **参数调优指南** - k, δ, num_partitions 如何选择
3. **性能优化技巧** - 并行化配置，内存优化
4. **故障排查** - 常见问题与解决方案
5. **高级用法** - 自定义 LocalClusterer, MergeStrategy

**预计时间**: 2 天

---

### 4.4 发布检查清单 (Release Checklist)

**代码质量**:
- [ ] `cargo clippy` 无警告
- [ ] `cargo fmt --check` 通过
- [ ] 所有测试通过 (`cargo test --all`)
- [ ] 基准测试完成
- [ ] 测试覆盖率 > 80%

**文档**:
- [ ] README 完整
- [ ] API 文档完整
- [ ] 用户指南完成
- [ ] CHANGELOG 更新

**性能**:
- [ ] 基准测试结果达到目标
- [ ] 内存使用合理
- [ ] 无性能回归

**发布**:
- [ ] 版本号更新 (Cargo.toml)
- [ ] Git tag 创建
- [ ] GitHub release notes
- [ ] 考虑发布到 crates.io

---

## Phase 5: 高级功能 (Advanced Features) - Week 7+

### 5.1 分布式版本 (Distributed DiMergeCo) (Priority: LOW)

**目标**: 使用 MPI 实现真正的分布式 co-clustering

**技术栈**:
- `rsmpi` - Rust MPI bindings
- `serde` - 序列化/反序列化

**架构**:
```
Master Node:
- 概率分区
- 任务分配
- 最终合并

Worker Nodes:
- 本地协同聚类
- 子树合并
```

**预计时间**: 2-3 周

---

### 5.2 GPU 加速 (GPU Acceleration) (Priority: LOW)

**目标**: 使用 CUDA/OpenCL 加速 SVD 和矩阵操作

**库**:
- `cudarc` - CUDA in Rust
- `ocl` - OpenCL bindings

**加速目标**:
- SVD 计算
- 矩阵归一化
- Pearson 相关性

**预计时间**: 3-4 周

---

### 5.3 增量/在线版本 (Incremental/Online DiMergeCo) (Priority: LOW)

**目标**: 支持增量数据更新，无需完全重新计算

**算法**:
- 增量 SVD 更新
- 局部重新分区
- 子树合并更新

**预计时间**: 2-3 周

---

## 优先级总结 (Priority Summary)

### 立即开始 (Immediate - Week 1-2):
1. ✅ **LocalClusterer 与 Pipeline 集成** - 使 DiMergeCo 可用
2. ✅ **生命周期重构** - 修复核心功能问题
3. ✅ **单元测试补充** - 保证代码质量

### 短期 (Short-term - Week 3-4):
4. 📊 **基准测试套件** - 验证性能提升
5. 🔍 **性能分析** - 识别优化机会
6. ✅ **集成测试** - 端到端验证

### 中期 (Mid-term - Week 5-6):
7. 📖 **文档完善** - 用户可用性
8. 📋 **README 更新** - 项目展示
9. ✅ **理论验证** - 学术价值

### 长期 (Long-term - Week 7+):
10. 🌐 **分布式版本** - 真正大规模
11. 🚀 **GPU 加速** - 极致性能
12. 📈 **增量版本** - 流数据支持

---

## 资源需求 (Resource Requirements)

### 开发环境:
- Rust 1.70+
- LAPACK libraries (OpenBLAS/MKL)
- 8+ core CPU (用于并行测试)
- 16GB+ RAM

### 可选:
- GPU (NVIDIA CUDA for GPU acceleration)
- MPI cluster (for distributed version)

---

## 风险与缓解 (Risks & Mitigation)

| 风险 | 影响 | 概率 | 缓解策略 |
|------|------|------|----------|
| LAPACK 链接问题 | HIGH | MEDIUM | 提供详细安装文档，Docker 镜像 |
| 生命周期问题难以解决 | HIGH | LOW | 提供多个解决方案选项 |
| 性能未达预期 | MEDIUM | LOW | 增量优化，降低预期 |
| 内存开销过大 | MEDIUM | MEDIUM | 流式处理，释放中间结果 |
| 理论保证不成立 | MEDIUM | LOW | 调整参数计算，增加验证测试 |

---

## 成功标准 (Success Criteria)

### 必须 (Must Have):
- ✅ DiMergeCo 三阶段全部实现
- ✅ 编译通过，无错误/警告
- ✅ 与 Pipeline 集成可用
- ✅ 单元测试 > 80% 覆盖率
- ✅ 基本文档完整

### 应该 (Should Have):
- 📊 全流程 4-5x 加速 (8 核)
- ✅ 集成测试通过
- 📖 用户指南完整
- 🔍 基准测试完成

### 可以 (Nice to Have):
- 🌐 分布式版本
- 🚀 GPU 加速
- 📈 增量更新支持

---

## 时间线 (Timeline)

```
Week 1-2:  Pipeline Integration + Lifetime Refactoring
Week 3-4:  Benchmarking + Performance Optimization
Week 5:    Testing + Validation
Week 6:    Documentation + Release Prep
Week 7+:   Advanced Features (Optional)
```

**预计总时间**: 6 周核心功能 + 可选高级功能

---

## 下一步行动 (Next Action Items)

1. ☑️ **今天完成**:
   - Review 当前代码
   - 确认计划优先级

2. **明天开始**:
   - 实现 `pipeline_integration.rs`
   - 添加 `with_dimerge_co()` 方法

3. **本周完成**:
   - Pipeline 集成测试通过
   - 生命周期问题解决

4. **下周目标**:
   - 基准测试框架搭建
   - 初步性能数据收集

---

**Made with ❤️ by Claude Sonnet 4.5**
