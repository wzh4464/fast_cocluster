# Cross-Project Workflow: Paper ↔ Code

## Quick Reference

### Two Projects Overview
1. **big-cocluster-paper** (LaTeX/Python) - Theory & Algorithms
   - Path: `/Volumes/Mac_Ext/link_cache/codes/latex/big-cocluster-paper`
   - Purpose: DiMergeCo paper with mathematical foundations
   - Main file: `root.tex`

2. **fast_cocluster** (Rust) - Implementation
   - Path: `/Volumes/Mac_Ext/link_cache/codes/fast_cocluster`
   - Purpose: High-performance co-clustering library
   - Main file: `src/lib.rs`

## Typical Development Flow

### Scenario 1: Implementing a New Algorithm from Paper

1. **Read paper** (switch to paper project)
   ```
   activate_project("big-cocluster-paper")
   read_file("root.tex") 
   # Or search for specific algorithm
   search_for_pattern("Algorithm.*Partitioning")
   ```

2. **Extract algorithm details**
   - Mathematical formulation
   - Input/output specifications
   - Complexity analysis
   - Theoretical guarantees

3. **Design implementation** (switch to code project)
   ```
   activate_project("fast_cocluster")
   ```

4. **Identify target module**
   - Check `paper_implementation_guide.md` memory
   - Decide which module (`pipeline.rs`, `cocluster.rs`, etc.)

5. **Write tests first**
   ```rust
   #[test]
   fn test_probabilistic_partitioning() {
       // Based on paper's examples
   }
   ```

6. **Implement algorithm**
   - Follow paper's mathematical description
   - Add doc comments referencing paper sections
   - Include theoretical guarantees as assertions

7. **Validate**
   - Compare with paper's experimental results
   - Check complexity bounds
   - Run benchmarks

### Scenario 2: Understanding Existing Code via Paper

1. **Start with code** (in fast_cocluster)
   ```
   # Find implementation
   find_symbol("SVDClusterer")
   ```

2. **Look up corresponding paper section**
   - Check `paper_implementation_guide.md` mapping table
   - Note which paper section describes this

3. **Read paper theory**
   ```
   activate_project("big-cocluster-paper")
   # Read relevant section
   ```

4. **Validate implementation matches theory**
   - Check mathematical equivalence
   - Verify complexity guarantees
   - Add missing theoretical checks if needed

### Scenario 3: Debugging with Paper Reference

1. **Encounter bug in code**
   ```
   activate_project("fast_cocluster")
   # Debug the issue
   ```

2. **Check paper for constraints/assumptions**
   ```
   activate_project("big-cocluster-paper")
   # Search for assumptions
   search_for_pattern("Assumption.*rank")
   ```

3. **Verify code follows paper's conditions**
   - Add validation checks
   - Document assumptions in code comments
   - Add test cases for boundary conditions

4. **Fix and document**
   ```
   activate_project("fast_cocluster")
   # Fix code
   # Add reference to paper section in comments
   ```

## Code Documentation Standards

### Referencing Paper in Code

```rust
/// Probabilistic partitioning algorithm from DiMergeCo paper.
/// 
/// # Paper Reference
/// Section 3.2: "Probabilistic Partitioning Algorithm"
/// Equation (7): Threshold computation
/// 
/// # Theoretical Guarantee
/// Preserves co-clusters with probability >= 1 - δ
/// where δ is the error tolerance.
/// 
/// # Complexity
/// O(n log n) as proven in Theorem 1
pub fn partition_with_guarantee(
    matrix: &Matrix<f64>,
    threshold: f64,
    delta: f64,
) -> Result<Vec<Submatrix>, Error> {
    // Implementation
}
```

### Linking Paper Citations to Code

```rust
// Based on Spectral Co-clustering (Dhillon et al., 2001)
// See paper reference: \SCCcite in root.tex
pub struct SpectralCoclusterer {
    // ...
}
```

## Maintaining Consistency

### When Paper is Updated

1. **Check for algorithm changes**
   ```
   cd /Volumes/Mac_Ext/link_cache/codes/latex/big-cocluster-paper
   git log --oneline root.tex
   git diff <prev-commit> root.tex
   ```

2. **Update code if needed**
   ```
   activate_project("fast_cocluster")
   # Implement changes
   ```

3. **Update memories**
   - Update `paper_implementation_guide.md`
   - Note version/date changes

### When Code is Enhanced

1. **Check if paper needs update**
   - New optimizations → supplementary material?
   - Bug fixes → errata?
   - New experiments → update results?

2. **Document divergence**
   - If code differs from paper, document why
   - Add to `paper_implementation_guide.md` under "Implementation Notes"

## Quick Commands

### Switch Projects
```bash
# In Serena/Claude Code interface
activate_project("big-cocluster-paper")   # Read paper
activate_project("fast_cocluster")         # Write code
```

### Search Across Projects

#### Find Algorithm in Paper
```bash
# When in big-cocluster-paper
search_for_pattern("\\\\begin{algorithm}")
search_for_pattern("Theorem.*preservation")
```

#### Find Implementation in Code
```bash
# When in fast_cocluster
find_symbol("partition")
grep -r "probabilistic" src/
```

## Memory Files Reference

### In fast_cocluster project:
- `paper_implementation_guide.md` ⭐ - This file maps paper → code
- `architecture_patterns.md` - Code design patterns
- `codebase_structure.md` - Code organization

### In big-cocluster-paper project:
- `project_overview.md` - Paper overview
- `code_structure.md` - Python/LaTeX structure

## Best Practices

### ✅ Do:
- Always check paper before implementing new features
- Reference paper sections in code comments
- Validate theoretical guarantees in tests
- Keep `paper_implementation_guide.md` updated
- Document when code intentionally differs from paper

### ❌ Don't:
- Implement algorithms without reading paper first
- Ignore theoretical guarantees in code
- Forget to update cross-references when either changes
- Assume paper and code are always in sync

## Example: Full Feature Implementation

**Goal**: Implement hierarchical merging from paper

**Step-by-step**:

1. Read paper (15 min)
   ```
   activate_project("big-cocluster-paper")
   read_file("root.tex", start=300, end=400)  # Merging section
   ```

2. Extract algorithm (5 min)
   - Note: Binary tree structure
   - Note: O(log n) complexity requirement
   - Note: Merge operation definition

3. Design in code (10 min)
   ```
   activate_project("fast_cocluster")
   // Sketch data structures
   // Plan module location
   ```

4. Write tests (20 min)
   ```rust
   #[test]
   fn test_hierarchical_merge_complexity() {
       // Verify O(log n) complexity
   }
   ```

5. Implement (60 min)
   ```rust
   /// Hierarchical merging from DiMergeCo (Section 3.3)
   pub fn hierarchical_merge(...) { ... }
   ```

6. Validate (30 min)
   - Run tests
   - Check complexity
   - Compare with paper results

7. Document (15 min)
   - Update `paper_implementation_guide.md`
   - Add code comments with paper references

**Total**: ~2.5 hours for a well-grounded implementation

## Troubleshooting

### "Can't find algorithm in paper"
- Check supplement: `supplement.tex`
- Search by keywords: `search_for_pattern("divide.*conquer")`
- Check references.bib for cited papers

### "Code behavior differs from paper"
- Verify implementation matches mathematical notation
- Check for typos in equations
- Review assumptions (paper may have different constraints)
- Consider numerical precision issues

### "Performance doesn't match paper"
- Check dataset size/characteristics
- Verify same algorithms are being compared
- Review hardware differences (paper may use cluster)
- Check if distributed features are implemented (many aren't yet)

---

**Key Insight**: Treat the paper as the "specification" and the code as the "implementation". Always sync bidirectionally when either changes.