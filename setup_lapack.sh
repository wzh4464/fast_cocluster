#!/bin/bash
# LAPACK Environment Setup for Fast CoCluster
# Cross-platform: macOS, Linux (Ubuntu/Debian/RHEL/Arch/openSUSE)
# Supports: OpenBLAS, Intel MKL, BLIS, Accelerate

set -e

echo "🔧 Setting up LAPACK environment for Fast CoCluster..."
echo

# Detect OS
OS="$(uname -s)"
case "$OS" in
    Linux*)
        PLATFORM="Linux"
        LIB_PATH_VAR="LD_LIBRARY_PATH"
        ;;
    Darwin*)
        PLATFORM="macOS"
        LIB_PATH_VAR="DYLD_LIBRARY_PATH"
        ;;
    CYGWIN*|MINGW*|MSYS*)
        PLATFORM="Windows"
        echo "⚠️  Windows is not fully supported. Use WSL2 for best experience."
        exit 1
        ;;
    *)
        PLATFORM="Unknown"
        echo "❌ Unsupported operating system: $OS"
        exit 1
        ;;
esac

echo "✓ Detected platform: $PLATFORM"
echo

# Function to detect Linux distribution
detect_linux_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "$ID"
    elif [ -f /etc/lsb-release ]; then
        . /etc/lsb-release
        echo "$DISTRIB_ID" | tr '[:upper:]' '[:lower:]'
    else
        echo "unknown"
    fi
}

# Function to find OpenBLAS on macOS
find_openblas_macos() {
    if command -v brew &> /dev/null; then
        if brew list openblas &> /dev/null 2>&1; then
            brew --prefix openblas
            return 0
        else
            echo ""
            return 1
        fi
    else
        echo ""
        return 1
    fi
}

# Function to find OpenBLAS on Linux
find_openblas_linux() {
    # Common OpenBLAS installation paths
    local COMMON_PATHS=(
        "/usr/lib/x86_64-linux-gnu/openblas-pthread"
        "/usr/lib/openblas-base"
        "/usr/lib64/openblas"
        "/usr/lib/x86_64-linux-gnu"
        "/usr/lib"
        "/usr/local/lib"
        "/opt/OpenBLAS/lib"
    )

    for path in "${COMMON_PATHS[@]}"; do
        if [ -f "$path/libopenblas.so" ] || [ -f "$path/libopenblas.a" ]; then
            echo "$path"
            return 0
        fi
    done

    # Try pkg-config
    if command -v pkg-config &> /dev/null; then
        if pkg-config --exists openblas; then
            pkg-config --variable=libdir openblas
            return 0
        fi
    fi

    echo ""
    return 1
}

# Function to find Intel MKL
find_mkl() {
    # Common MKL installation paths
    local MKL_PATHS=(
        "/opt/intel/oneapi/mkl/latest"
        "/opt/intel/mkl"
        "$HOME/intel/oneapi/mkl/latest"
        "$HOME/intel/mkl"
    )

    for path in "${MKL_PATHS[@]}"; do
        if [ -d "$path" ]; then
            echo "$path"
            return 0
        fi
    done

    echo ""
    return 1
}

# Main BLAS detection logic
BLAS_FOUND=0
BLAS_TYPE=""
BLAS_PATH=""

echo "🔍 Searching for BLAS libraries..."
echo

if [ "$PLATFORM" = "macOS" ]; then
    # macOS: Try OpenBLAS via Homebrew first, then fall back to Accelerate
    OPENBLAS_PATH=$(find_openblas_macos)

    if [ -n "$OPENBLAS_PATH" ]; then
        BLAS_FOUND=1
        BLAS_TYPE="OpenBLAS"
        BLAS_PATH="$OPENBLAS_PATH"
        echo "✓ Found OpenBLAS (Homebrew): $BLAS_PATH"
    else
        # Use macOS Accelerate framework (always available)
        BLAS_FOUND=1
        BLAS_TYPE="Accelerate"
        BLAS_PATH="/System/Library/Frameworks/Accelerate.framework"
        echo "✓ Using macOS Accelerate framework (system default)"
        echo "⚠️  Note: OpenBLAS recommended for better performance"
        echo "   Install with: brew install openblas"
    fi

elif [ "$PLATFORM" = "Linux" ]; then
    DISTRO=$(detect_linux_distro)
    echo "✓ Detected Linux distribution: $DISTRO"
    echo

    # Try to find Intel MKL first (best performance)
    MKL_PATH=$(find_mkl)
    if [ -n "$MKL_PATH" ]; then
        BLAS_FOUND=1
        BLAS_TYPE="Intel MKL"
        BLAS_PATH="$MKL_PATH"
        echo "✓ Found Intel MKL: $BLAS_PATH"
    else
        # Try to find OpenBLAS
        OPENBLAS_PATH=$(find_openblas_linux)
        if [ -n "$OPENBLAS_PATH" ]; then
            BLAS_FOUND=1
            BLAS_TYPE="OpenBLAS"
            BLAS_PATH="$OPENBLAS_PATH"
            echo "✓ Found OpenBLAS: $BLAS_PATH"
        fi
    fi

    # If no BLAS found, provide installation instructions
    if [ $BLAS_FOUND -eq 0 ]; then
        echo "❌ No BLAS library found!"
        echo
        echo "📦 Installation instructions:"
        echo

        case "$DISTRO" in
            ubuntu|debian)
                echo "  sudo apt-get update"
                echo "  sudo apt-get install libopenblas-dev liblapack-dev"
                ;;
            fedora|rhel|centos)
                echo "  sudo dnf install openblas-devel lapack-devel"
                ;;
            arch|manjaro)
                echo "  sudo pacman -S openblas lapack"
                ;;
            opensuse*)
                echo "  sudo zypper install openblas-devel lapack-devel"
                ;;
            *)
                echo "  # For your distribution, install:"
                echo "  # - openblas or libopenblas-dev"
                echo "  # - lapack or liblapack-dev"
                ;;
        esac

        echo
        echo "🌟 For Intel CPUs, consider Intel MKL for best performance:"
        echo "   https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html"
        echo
        exit 1
    fi
fi

echo
echo "✅ BLAS library configured: $BLAS_TYPE"
echo

# Set environment variables based on BLAS type
if [ "$BLAS_TYPE" = "OpenBLAS" ]; then
    # Detect library directory
    if [ -f "$BLAS_PATH/libopenblas.so" ] || [ -f "$BLAS_PATH/libopenblas.a" ]; then
        LIB_DIR="$BLAS_PATH"
    elif [ -f "$BLAS_PATH/lib/libopenblas.so" ] || [ -f "$BLAS_PATH/lib/libopenblas.a" ]; then
        LIB_DIR="$BLAS_PATH/lib"
    else
        LIB_DIR="$BLAS_PATH"
    fi

    # Detect include directory
    if [ -d "$BLAS_PATH/include" ]; then
        INCLUDE_DIR="$BLAS_PATH/include"
    else
        INCLUDE_DIR="/usr/include"
    fi

    export OPENBLAS_DIR="$BLAS_PATH"
    export OPENBLAS_LIB="$LIB_DIR"
    export OPENBLAS_INCLUDE="$INCLUDE_DIR"

    # Runtime library path
    if [ "$PLATFORM" = "macOS" ]; then
        export DYLD_LIBRARY_PATH="$LIB_DIR:${DYLD_LIBRARY_PATH:-}"
    else
        export LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}"
    fi

    # For ndarray-linalg
    export CARGO_FEATURE_OPENBLAS=1

    echo "✓ Environment variables set:"
    echo "  OPENBLAS_DIR=$OPENBLAS_DIR"
    echo "  OPENBLAS_LIB=$OPENBLAS_LIB"
    echo "  OPENBLAS_INCLUDE=$OPENBLAS_INCLUDE"
    echo "  $LIB_PATH_VAR=$LIB_DIR:..."

elif [ "$BLAS_TYPE" = "Intel MKL" ]; then
    export MKLROOT="$BLAS_PATH"
    export MKL_LIB="$BLAS_PATH/lib/intel64"
    export MKL_INCLUDE="$BLAS_PATH/include"

    if [ "$PLATFORM" = "macOS" ]; then
        export DYLD_LIBRARY_PATH="$MKL_LIB:${DYLD_LIBRARY_PATH:-}"
    else
        export LD_LIBRARY_PATH="$MKL_LIB:${LD_LIBRARY_PATH:-}"
    fi

    export CARGO_FEATURE_INTEL_MKL=1

    echo "✓ Environment variables set:"
    echo "  MKLROOT=$MKLROOT"
    echo "  MKL_LIB=$MKL_LIB"
    echo "  MKL_INCLUDE=$MKL_INCLUDE"
    echo "  $LIB_PATH_VAR=$MKL_LIB:..."

elif [ "$BLAS_TYPE" = "Accelerate" ]; then
    # macOS Accelerate framework requires no environment setup
    # ndarray-linalg will use it automatically
    echo "✓ Using macOS Accelerate (no environment setup needed)"
fi

echo

# Test compilation
echo "🔨 Testing compilation..."
if cargo check --all-targets 2>&1 | grep -q "error"; then
    echo
    echo "❌ Compilation test failed!"
    echo "   Check the error messages above."
    exit 1
else
    echo "✅ Compilation test passed!"
fi

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ LAPACK environment configured successfully!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "📋 Configuration Summary:"
echo "   Platform: $PLATFORM"
echo "   BLAS Library: $BLAS_TYPE"
echo "   Library Path: $BLAS_PATH"
echo
echo "🚀 Next Steps:"
echo
echo "   1. Activate environment (in your shell):"
echo "      source setup_lapack.sh"
echo
echo "   2. Run tests:"
echo "      cargo test"
echo
echo "   3. Run benchmarks:"
echo "      cargo bench --bench dimerge_co_benchmarks"
echo
echo "   4. Build release binary:"
echo "      cargo build --release"
echo

if [ "$BLAS_TYPE" = "Accelerate" ]; then
    echo "💡 Performance Tip:"
    echo "   For better performance on macOS, install OpenBLAS:"
    echo "   brew install openblas"
    echo "   Then re-run this script."
    echo
fi

if [ "$BLAS_TYPE" = "OpenBLAS" ] && [ "$PLATFORM" = "Linux" ]; then
    echo "💡 Performance Tip:"
    echo "   For Intel CPUs, consider Intel MKL for 2-3x better performance:"
    echo "   https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html"
    echo
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
