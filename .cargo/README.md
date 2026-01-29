# Cargo Configuration for LAPACK/OpenBLAS

This project requires LAPACK/OpenBLAS for linear algebra operations. The linker configuration varies by platform.

## Quick Setup

Choose the appropriate configuration file for your platform and copy it to `.cargo/config.toml`:

### macOS (Homebrew)
```bash
cp .cargo/config.toml.macos-example .cargo/config.toml
```

### Linux (System OpenBLAS)
```bash
cp .cargo/config.toml.linux-system-example .cargo/config.toml
```

### Linux (Conda Environment)
```bash
cp .cargo/config.toml.linux-conda-example .cargo/config.toml
# Edit the file to replace $CONDA_PREFIX with your actual conda path
```

## Installation Instructions

### macOS
```bash
brew install openblas
```

### Ubuntu/Debian
```bash
sudo apt-get install libopenblas-dev
```

### CentOS/RHEL/Fedora
```bash
sudo yum install openblas-devel
```

### Conda (Any Platform)
```bash
conda install -c conda-forge openblas
```

## Troubleshooting

### Linking Errors
If you get linking errors like:
- `cannot find -lopenblas`
- `undefined reference to LAPACK symbols`

Try these steps:

1. **Find your OpenBLAS library location:**
   ```bash
   # macOS
   brew --prefix openblas

   # Linux
   ldconfig -p | grep openblas

   # Conda
   echo $CONDA_PREFIX/lib
   ```

2. **Update `.cargo/config.toml`** with the correct path:
   ```toml
   [build]
   rustflags = [
       "-C", "link-arg=-L/path/to/your/openblas/lib",
       "-C", "link-arg=-lopenblas",
   ]
   ```

3. **Add rpath (Linux/macOS only, for non-system paths):**

   Linux:
   ```toml
   "-C", "link-arg=-Wl,-rpath,/path/to/your/openblas/lib",
   ```

   macOS:
   ```toml
   "-C", "link-arg=-rpath",
   "-C", "link-arg=/path/to/your/openblas/lib",
   ```

### No Configuration Needed (Default)

If OpenBLAS is installed in system default locations and your system finds it automatically, you may not need a `.cargo/config.toml` file at all. Try building without it first:

```bash
cargo build
```

If it works, you're all set!

## Notes

- `.cargo/config.toml` is in `.gitignore` and should not be committed
- Each developer can have their own platform-specific configuration
- The example files are for reference only and committed to the repository
