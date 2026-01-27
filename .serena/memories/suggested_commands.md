# Fast Co-clustering Library - Suggested Commands

## Development Commands

### Build
```bash
# Standard build
cargo build

# Verbose build
cargo build --verbose

# Release build (optimized)
cargo build --release
```

### Testing
```bash
# Run all tests
cargo test

# Run tests with output
cargo test --verbose

# Run specific test
cargo test <test_name>

# Run tests without capturing output
cargo test -- --nocapture

# Run tests in single thread
cargo test -- --test-threads=1
```

### Formatting
```bash
# Format all code
cargo fmt

# Check formatting without modifying
cargo fmt -- --check
```

### Linting (if clippy installed)
```bash
# Install clippy
rustup component add clippy

# Run clippy
cargo clippy

# Run clippy with all warnings
cargo clippy -- -W clippy::all
```

### Documentation
```bash
# Generate documentation
cargo doc

# Generate and open documentation
cargo doc --open

# Generate documentation for dependencies
cargo doc --no-deps --open
```

### Cleaning
```bash
# Remove build artifacts
cargo clean

# Remove target directory and Cargo.lock
cargo clean && rm -f Cargo.lock
```

## Code Coverage (tarpaulin)

The project has coverage reporting via tarpaulin (as evidenced by tarpaulin-report.html):

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Run coverage
cargo tarpaulin --out Html

# Run coverage with all features
cargo tarpaulin --out Html --all-features
```

## Git Commands (Darwin/macOS specific)

### Basic Git Operations
```bash
# Check status
git status

# Stage changes
git add <file>

# Commit changes
git commit -m "message"

# Push to remote
git push origin main

# Pull from remote
git pull origin main

# View commit history
git log --oneline --graph
```

### Branch Operations
```bash
# Create new branch
git checkout -b <branch-name>

# Switch branches
git checkout <branch-name>

# List branches
git branch -a

# Delete branch
git branch -d <branch-name>
```

## System Utilities (Darwin/macOS)

### File Operations
```bash
# List files (macOS)
ls -la

# Find files
find . -name "*.rs"

# Search in files
grep -r "pattern" src/

# Count lines of code
find src -name "*.rs" -exec wc -l {} + | tail -1
```

### Process Management
```bash
# Check running processes
ps aux | grep cargo

# Kill process
kill <PID>

# Monitor system resources
top
```

## CI/CD

### GitHub Actions
The project uses GitHub Actions for CI (.github/workflows/rust.yml):

**Triggers:**
- Push to `main` branch
- Pull requests to `main` branch

**Jobs:**
- Set up Rust nightly
- Build with `cargo build --verbose`
- Test with `cargo test --verbose`

## Running the Project

### As Library
```rust
// In your Cargo.toml
[dependencies]
fast_cocluster = { path = "../fast_cocluster" }
// or
fast_cocluster = { git = "https://github.com/wzh4464/fast_cocluster" }
```

### As Binary (if main.rs is configured)
```bash
cargo run

# With arguments
cargo run -- <args>

# Release mode
cargo run --release
```

## Benchmarking

```bash
# Run benchmarks (if configured)
cargo bench

# Run specific benchmark
cargo bench <benchmark_name>
```

## Dependency Management

```bash
# Update dependencies
cargo update

# Check for outdated dependencies
cargo outdated  # Requires: cargo install cargo-outdated

# Audit dependencies for security issues
cargo audit  # Requires: cargo install cargo-audit

# Show dependency tree
cargo tree
```

## Useful Aliases (Optional)

Add to ~/.zshrc or ~/.bashrc (macOS uses zsh by default):

```bash
alias cb='cargo build'
alias ct='cargo test'
alias cr='cargo run'
alias cf='cargo fmt'
alias cc='cargo clippy'
alias cw='cargo watch -x test'  # Requires: cargo install cargo-watch
```