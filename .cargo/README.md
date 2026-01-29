# Cargo Configuration for LAPACK/OpenBLAS

**Note:** As of the latest version, OpenBLAS detection is **automatic** via `build.rs`. You typically don't need to manually configure anything!

## Automatic Detection (Recommended)

The `build.rs` script automatically detects OpenBLAS in these locations:

1. **Environment variable** `OPENBLAS_LIB` (highest priority)
2. **Conda environment** (`$CONDA_PREFIX/lib`)
3. **Platform-specific paths:**
   - macOS: Homebrew (`/opt/homebrew/opt/openblas/lib`, `/usr/local/opt/openblas/lib`)
   - Linux: System packages (`/usr/lib/x86_64-linux-gnu`, `/usr/lib64`, `/usr/lib`)
   - Also checks `pkg-config --libs openblas`

Just install OpenBLAS and build:

```bash
# macOS
brew install openblas
cargo build

# Ubuntu/Debian
sudo apt-get install libopenblas-dev
cargo build

# CentOS/RHEL/Fedora
sudo yum install openblas-devel
cargo build

# Conda (any platform)
conda install -c conda-forge openblas
cargo build
```

## Manual Configuration (Fallback)

If automatic detection doesn't work, you can manually specify the path:

### Option 1: Environment Variable (Recommended)
```bash
export OPENBLAS_LIB=/path/to/your/openblas/lib
cargo build
```

### Option 2: Use Example Config Files
Copy the appropriate example file for your platform:

```bash
# macOS (Homebrew)
cp .cargo/config.toml.macos-example .cargo/config.toml

# Linux (System OpenBLAS)
cp .cargo/config.toml.linux-system-example .cargo/config.toml

# Linux (Conda Environment)
cp .cargo/config.toml.linux-conda-example .cargo/config.toml
# Edit the file to replace $CONDA_PREFIX with your actual conda path
```

Note: `.cargo/config.toml` is in `.gitignore` and is optional.

## Troubleshooting

### Build fails with "cannot find -lopenblas"

1. **Verify OpenBLAS is installed:**
   ```bash
   # macOS
   brew list openblas

   # Linux
   dpkg -l | grep openblas  # Debian/Ubuntu
   rpm -qa | grep openblas  # RHEL/CentOS

   # Find library location
   find /usr -name "libopenblas.*" 2>/dev/null
   ```

2. **Set OPENBLAS_LIB environment variable:**
   ```bash
   export OPENBLAS_LIB=/usr/lib/x86_64-linux-gnu  # Example path
   cargo build
   ```

3. **Check build.rs output:**
   ```bash
   cargo clean
   cargo build -vv 2>&1 | grep -i openblas
   ```

   Look for the line: `Found OpenBLAS at: /path/to/lib`

### Still not working?

Open an issue with:
- Your OS and version
- OpenBLAS installation method
- Output of `cargo build -vv`
- Output of `find /usr /opt /home -name "libopenblas*" 2>/dev/null | head -10`

## For Developers

The automatic detection is implemented in `build.rs`. The example config files are kept for reference and edge cases where manual configuration is needed.
