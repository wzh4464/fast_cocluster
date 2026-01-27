# DiMergeCo Paper Implementation Guide

## Paper Location
**Project**: big-cocluster-paper  
**Path**: `/Volumes/Mac_Ext/link_cache/codes/latex/big-cocluster-paper`  
**Main File**: `root.tex`  
**Status**: Registered Serena project

## Paper Overview: DiMergeCo

**Full Title**: "DiMergeCo: A Scalable Framework for Large-Scale Co-Clustering with Theoretical Guarantees"

**Authors**: Zihan Wu, Zhaoke Huang, Hong Yan (IEEE Fellow)

**Key Innovations**:
1. **Probabilistic Partitioning Algorithm**: Preserves co-clusters during matrix division with theoretical guarantees
2. **Hierarchical Merging Strategy**: Binary tree-based merging reducing communication complexity from O(n) to O(log n)
3. **Distributed MPI Implementation**: Scales to datasets with 685K samples

**Performance Gains**:
- 83% reduction in computation time for dense matrices
- Successfully processes million-dimensional matrices
- Outperforms existing methods in speed and accuracy

## Core Algorithms from Paper

### 1. Probabilistic Partitioning
**Purpose**: Divide large matrix into smaller submatrices while preserving co-clustering structures

**Theoretical Basis**: Properties of low-rank submatrices

**Key Properties**:
- Preserves co-clusters during division
- Transforms global co-clustering into independent local problems
- Has provable guarantees

**Implementation Status in Code**:
- ✅ Basic partitioning exists in `union_dc.rs` (UnionDC divide-and-conquer)
- ⚠️ Probabilistic guarantees need theoretical validation
- ⚠️ Threshold computation needs main node coordination

### 2. Hierarchical Merging Strategy
**Purpose**: Aggregate local co-clustering results efficiently

**Key Features**:
- Binary tree-based merging
- O(log n) communication complexity (vs O(n) traditional)
- Eliminates central coordinator bottleneck

**Implementation Status in Code**:
- ✅ Basic merging framework in `union_dc.rs`
- ❌ Binary tree structure not explicitly implemented
- ❌ Distributed communication layer missing (no MPI)
- ⚠️ Current implementation is sequential, not distributed

### 3. Distributed Architecture (MPI-based)
**Purpose**: Enable parallel processing across multiple nodes

**Design**:
- Main node: Computes initial partitioning thresholds only
- Worker nodes: Perform local co-clustering independently
- Communication: Hierarchical merging via MPI

**Implementation Status in Code**:
- ✅ Parallel processing via Rayon (shared memory)
- ❌ MPI-based distributed processing not implemented
- ❌ Main/worker node architecture not implemented
- ⚠️ Current parallelism is thread-based, not process-based

## Mapping Paper to Code

### Paper Algorithm → Current Code Modules

| Paper Component | Current Module | Implementation Status |
|----------------|----------------|----------------------|
| Probabilistic Partitioning | `union_dc.rs` | Partial (basic D&C) |
| Hierarchical Merging | `union_dc.rs` | Partial (sequential) |
| Co-clustering Core | `cocluster.rs` | ✅ Complete (SVD-based) |
| Spectral Method | `spectral_cocluster.rs` | ✅ Complete |
| Scoring Methods | `scoring.rs` | ✅ Complete (Pearson, etc.) |
| Pipeline Framework | `pipeline.rs` | ✅ Complete |
| Distributed System | ❌ Missing | Not implemented |

### Key Code Components Aligned with Paper

#### 1. SVD-Based Clustering (`cocluster.rs`)
**Paper Reference**: Foundation for local co-clustering in partitioned submatrices

**Current Implementation**:
```rust
pub struct Coclusterer {
    // Performs SVD-based co-clustering
    // Used as the local co-clustering algorithm in DiMergeCo
}
```

#### 2. UnionDC Framework (`union_dc.rs`)
**Paper Reference**: Implements divide-and-conquer strategy

**Current Implementation**:
```rust
pub struct UnionDC;
// Generic divide-and-conquer framework
// Can be specialized for DiMergeCo partitioning/merging
```

#### 3. Pipeline Architecture (`pipeline.rs`)
**Paper Reference**: Orchestrates the overall co-clustering workflow

**Current Implementation**:
```rust
pub struct CoclusterPipeline {
    clusterer: Box<dyn Clusterer>,
    scorer: Box<dyn Scorer>,
    config: PipelineConfig,
}
```

## Implementation Gaps

### Critical Missing Features for DiMergeCo

1. **Probabilistic Partitioning with Guarantees**
   - Current: Basic divide-and-conquer
   - Needed: Threshold-based partitioning preserving co-clusters
   - Theory: Formalize preservation guarantees

2. **Binary Tree Merging**
   - Current: Sequential merging
   - Needed: Hierarchical binary tree structure
   - Theory: O(log n) complexity guarantee

3. **Distributed Communication Layer**
   - Current: Rayon (shared memory parallelism)
   - Needed: MPI or similar (distributed memory)
   - Challenge: Rust + MPI integration

4. **Main/Worker Architecture**
   - Current: Single-process design
   - Needed: Coordinator node + worker nodes
   - Design: Main node only for threshold computation

### Enhancement Opportunities

1. **Scalability Improvements**
   - Add support for sparse matrices
   - Implement streaming algorithms
   - Add memory-mapped file support

2. **Theoretical Validation**
   - Add convergence bound checks
   - Implement preservation guarantee tests
   - Add statistical validation of results

3. **Performance Optimization**
   - GPU acceleration for SVD
   - Better cache locality
   - Optimized matrix operations

## How to Use Paper as Implementation Guide

### Development Workflow

1. **Read Paper Section** (in big-cocluster-paper project)
   ```bash
   # View LaTeX source
   cd /Volumes/Mac_Ext/link_cache/codes/latex/big-cocluster-paper
   cat root.tex | grep -A 20 "section{Algorithm}"
   ```

2. **Identify Algorithm** (mathematical description)
   - Note: Algorithms usually in `\begin{algorithm}...\end{algorithm}` blocks
   - Extract: Input, Output, Steps, Complexity

3. **Map to Code Structure** (in fast_cocluster project)
   - Determine which module should implement it
   - Identify required data structures
   - Plan trait implementations

4. **Implement with Tests**
   - Write unit tests first (TDD approach)
   - Implement algorithm
   - Validate against paper's theoretical guarantees

5. **Validate Performance**
   - Compare against paper's benchmarks
   - Verify complexity bounds
   - Test on similar datasets

### Cross-Project Commands

#### Switch Between Projects
```bash
# In Serena, switch to paper project
activate_project("big-cocluster-paper")

# Read paper content
read_file("root.tex")

# Switch back to code project
activate_project("fast_cocluster")

# Implement feature
```

#### Extract Algorithm from Paper
When in paper project:
```bash
# Find algorithm blocks
grep -n "\\begin{algorithm}" root.tex

# Extract specific sections
sed -n '/\\section{Method}/,/\\section{Experiments}/p' root.tex
```

## Theoretical Guarantees to Implement

### From Paper Abstract:
1. **Co-cluster Preservation**: Probabilistic partitioning preserves co-clusters
2. **Bounded Convergence**: Hierarchical merging has convergence bounds
3. **Communication Complexity**: O(log n) instead of O(n)

### Implementation Checklist:
- [ ] Add preservation probability calculation
- [ ] Implement convergence bound monitoring
- [ ] Track communication rounds in merging
- [ ] Add theoretical validation tests
- [ ] Document assumptions and conditions

## Experimental Validation

### Datasets from Paper:
- Synthetic datasets (controlled experiments)
- Real-world datasets (685K samples)
- Dense matrices (83% speedup demonstrated)

### Metrics to Track:
- Computation time
- Memory usage
- Accuracy (compared to ground truth)
- Scalability (samples vs time)
- Communication overhead

### Benchmark Goals:
- Match or exceed paper's 83% speedup
- Scale to 500K+ samples
- Validate theoretical complexity bounds

## References

### Key Paper Citations (from root.tex)
- Spectral Co-clustering: `\SCCcite` → dhillon2001CoclusteringDocumentsWords
- PNMTF: `\PNMTFcite` → chen2023ParallelNonNegativeMatrix
- ONMTF: `\ONMTFcite` → ding2006OrthogonalNonnegativeMatrix
- FNMTF: `\FNMTFcite` → kim2011FastNonnegativeMatrix

### Related Code References
- Bibliography: `references.bib` in paper project
- Supplementary Material: `supplement.tex` in paper project

## Development Priorities

### High Priority (Core DiMergeCo Features)
1. ⚠️ Implement probabilistic partitioning with thresholds
2. ⚠️ Add binary tree hierarchical merging
3. ⚠️ Add theoretical guarantee validation

### Medium Priority (Distributed Computing)
4. ❌ Evaluate MPI alternatives for Rust
5. ❌ Design main/worker architecture
6. ❌ Implement distributed communication

### Low Priority (Enhancements)
7. Add sparse matrix support
8. GPU acceleration
9. Additional scoring methods from paper

## Next Steps

When implementing new features from the paper:

1. **Always check paper first**: Read relevant sections in big-cocluster-paper
2. **Document mapping**: Note which paper section → which code module
3. **Preserve theory**: Implement with theoretical guarantees in mind
4. **Validate experimentally**: Match paper's experimental setup
5. **Update this guide**: Add new mappings as features are implemented

---

**Last Updated**: 2026-01-27  
**Paper Version**: DiMergeCo (IEEE Transactions submission)  
**Code Version**: fast_cocluster v0.1.0