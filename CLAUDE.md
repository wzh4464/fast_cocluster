# Fast Co-clustering (DiMergeCo)

## Development Workflow

- **Local machine**: Edit code, `cargo check` only (no OpenBLAS, linking fails)
- **Server**: Run full benchmarks and evaluations via `cargo run --release`
- **Sync**: Server pulls code from local via `git pull`
- Local cannot run `cargo test` or `cargo run` (需要 OpenBLAS 链接)
- Local 验证用 `cargo check --lib --tests --examples`

## Quick Reference

```bash
# Build
cargo build --release

# Run evaluation
cargo run --release --example evaluate_classic4

# Run tests (local-safe)
cargo test

# Run with logging enabled
RUST_LOG=info cargo run --release --example evaluate_classic4
```
