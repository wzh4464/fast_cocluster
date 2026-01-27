# Fast Co-clustering Library - Task Completion Checklist

## When a Task is Completed

Follow this checklist before marking any task as done:

### 1. Code Quality

#### Formatting
```bash
# Format code
cargo fmt

# Verify formatting
cargo fmt -- --check
```

#### Linting (if available)
```bash
# Run clippy
cargo clippy

# Fix clippy suggestions
cargo clippy --fix
```

#### Code Review
- [ ] Remove unused imports and variables
- [ ] Check for proper error handling
- [ ] Verify lifetime annotations are correct
- [ ] Ensure thread safety (Send + Sync) for concurrent code
- [ ] Add or update documentation comments

### 2. Testing

#### Run Tests
```bash
# Run all tests
cargo test --verbose

# Run specific module tests
cargo test <module_name>

# Run with output
cargo test -- --nocapture
```

#### Test Coverage
```bash
# Generate coverage report (if tarpaulin is installed)
cargo tarpaulin --out Html
```

#### Test Checklist
- [ ] Unit tests pass
- [ ] Integration tests pass (if applicable)
- [ ] Edge cases covered
- [ ] Error cases tested
- [ ] Performance-sensitive code benchmarked (if applicable)

### 3. Build

#### Development Build
```bash
# Build in debug mode
cargo build --verbose
```

#### Release Build
```bash
# Build in release mode (optimized)
cargo build --release
```

#### Build Checklist
- [ ] No compiler warnings
- [ ] No clippy warnings (if clippy is available)
- [ ] All dependencies resolve correctly
- [ ] Release build succeeds

### 4. Documentation

#### Generate Docs
```bash
# Generate and review documentation
cargo doc --open
```

#### Documentation Checklist
- [ ] Public APIs documented with doc comments
- [ ] Examples provided for complex features
- [ ] README.md updated (if needed)
- [ ] File headers updated with modification info

### 5. Git Operations

#### Before Committing
```bash
# Check status
git status

# Review changes
git diff

# Stage changes
git add <files>
```

#### Commit Message Format
Follow conventional commits:
- `feat: add new scoring method`
- `fix: correct SVD convergence issue`
- `docs: update README examples`
- `refactor: optimize parallel processing`
- `test: add integration tests for pipeline`
- `chore: update dependencies`

#### Git Checklist
- [ ] Meaningful commit messages
- [ ] No sensitive information in commits
- [ ] `.gitignore` is up to date
- [ ] Commit contains only related changes

### 6. Performance

#### Performance Checklist
- [ ] No unnecessary cloning
- [ ] Proper use of references
- [ ] Parallel processing enabled where appropriate
- [ ] Memory usage is reasonable
- [ ] No obvious performance bottlenecks

### 7. Code Style Compliance

#### Style Checklist
- [ ] Follows Rust naming conventions
- [ ] Uses Chinese comments where appropriate (project convention)
- [ ] Includes file header with modification info
- [ ] Proper error handling (Result types)
- [ ] Trait bounds specified correctly (Send + Sync where needed)

### 8. CI/CD

#### GitHub Actions
The CI pipeline will automatically:
- Build the project
- Run tests
- Report failures

#### CI Checklist
- [ ] Local tests pass (should match CI)
- [ ] No warnings in CI output
- [ ] Branch is up to date with main

### 9. Dependencies

#### If Dependencies Changed
```bash
# Update Cargo.lock
cargo build

# Check for security issues (if cargo-audit installed)
cargo audit

# Review dependency tree
cargo tree
```

#### Dependency Checklist
- [ ] Cargo.toml updated correctly
- [ ] Version constraints are appropriate
- [ ] No duplicate dependencies
- [ ] Security audit passes

## Quick Pre-Commit Checklist

For quick iterations, at minimum ensure:

1. [ ] `cargo fmt` - Code is formatted
2. [ ] `cargo test` - All tests pass
3. [ ] `cargo build` - Project builds without warnings
4. [ ] Git commit message is descriptive

## Full Pre-Release Checklist

Before releasing or merging to main:

1. [ ] All tests pass
2. [ ] Code coverage is acceptable
3. [ ] Documentation is complete
4. [ ] Examples work correctly
5. [ ] Release build succeeds
6. [ ] CHANGELOG.md updated (if exists)
7. [ ] Version number bumped (if applicable)
8. [ ] Git tags created for release
9. [ ] CI/CD pipeline passes