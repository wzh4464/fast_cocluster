# Fast Co-clustering (DiMergeCo)

## Development Workflow

- **Local machine**: Edit code, run small tests only (`cargo test` with small data)
- **Server**: Run full benchmarks and evaluations via `cargo run --release`
- **Sync**: Server pulls code from local via `git pull`
- Local cannot run large datasets (limited resources)

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
