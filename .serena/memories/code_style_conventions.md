# Fast Co-clustering Library - Code Style & Conventions

## Language Features

### Rust Edition & Compiler
- **Edition**: 2021
- **Compiler**: Nightly (required for latest features)
- **Allowed Lints**: `#![allow(unused)]` at crate level

## Naming Conventions

### Types
- **Structs**: PascalCase (e.g., `PipelineConfig`, `SVDClusterer`)
- **Traits**: PascalCase (e.g., `Clusterer`, `Scorer`)
- **Enums**: PascalCase (e.g., `CoclustererAlgorithmParams`)

### Functions & Methods
- **Functions**: snake_case (e.g., `new_config`, `timestamp`)
- **Methods**: snake_case (e.g., `cluster`, `score_parallel`)

### Variables & Fields
- **Variables**: snake_case (e.g., `test_matrix`, `end_time`)
- **Fields**: snake_case (e.g., `min_score`, `max_submatrices`)

### Constants
- SCREAMING_SNAKE_CASE (standard Rust convention)

## Documentation

### Comment Style
- **Mixed Language**: Chinese comments with English code
- **File Headers**: Include file path, creation date, author, modification history
- **Inline Comments**: Brief explanations in Chinese

### Example File Header:
```rust
/**
 * File: /src/lib.rs
 * Created Date: Monday, January 22nd 2024
 * Author: Zihan
 * -----
 * Last Modified: Monday, 26th May 2025 11:31:24 am
 * Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
 */
```

### Doc Comments
- Chinese descriptions for struct fields:
```rust
/// Pipeline配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// 最小分数阈值
    pub min_score: f64,
    /// 最大子矩阵数量
    pub max_submatrices: usize,
}
```

## Code Organization

### Imports
- Group by: std → external crates → internal modules
- Use explicit imports where possible
- Example:
```rust
use std::collections::HashMap;
use std::error::Error;

use log::{debug, info, warn};
use nalgebra::DMatrix;

use crate::scoring::Scorer;
use crate::matrix::Matrix;
```

### Module Structure
- Public API at top
- Trait definitions before implementations
- Helper functions at bottom

## Type Safety

### Generics & Lifetimes
- Explicit lifetime annotations for borrowed data
- Generic parameters with trait bounds:
```rust
pub trait Clusterer: Send + Sync {
    fn cluster<'matrix_life>(
        &self,
        matrix: &'matrix_life Matrix<f64>
    ) -> Result<Vec<Submatrix<'matrix_life, f64>>, Box<dyn Error>>;
}
```

### Error Handling
- Prefer `Result<T, E>` over panicking
- Use `?` operator for error propagation
- Custom error types when needed

## Parallelism

### Rayon Integration
- Use Rayon for data parallelism
- Traits require `Send + Sync` bounds
- Example:
```rust
use rayon::prelude::*;

scores.par_iter()
    .map(|score| process(score))
    .collect()
```

## Testing

### Debug Assertions
- Use `#[cfg(debug_assertions)]` for development-only code:
```rust
#[cfg(debug_assertions)]
{
    let end_time = Instant::now();
    println!("Time cost: {:?}", end_time.duration_since(start_time));
}
```

### Test Organization
- Tests in separate `tests` module
- Unit tests near implementation
- Integration tests in `tests/` directory

## Serialization

### Derive Macros
- Use serde for serializable types:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    // ...
}
```

## Performance Considerations

### Memory Efficiency
- Use references and borrowing where possible
- Avoid unnecessary cloning
- Use `Arc<Mutex<T>>` for shared state in parallel contexts

### Logging
- Use `log` crate with appropriate levels:
  - `debug!`: Detailed information
  - `info!`: General information
  - `warn!`: Warnings
- Initialize with `env_logger` in tests