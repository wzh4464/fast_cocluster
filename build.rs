/**
 * File: /src/build.rs
 * Created Date: Sunday, January 28th 2024
 * Author: Zihan
 * -----
 * Last Modified: Sunday, 28th January 2024 5:16:54 pm
 * Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
**/

// Example custom build script.
fn main() {
    // Tell Cargo to link lapack
    
    // # switch to your own directory
    // CC = mpicc -O3 -pedantic -Wall -Wextra -Wconversion -I/home/zihan/miniconda3/envs/pnmtf/include/python3.8 -I/home/zihan/miniconda3/envs/pnmtf/include

    // # switch to your own directory
    // LDLIBS=-L/home/zihan/miniconda3/envs/pnmtf/lib -lpython3.8 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_lapack95_lp64 -liomp5 -lpthread -lm -lmkl_intel_lp64

    // cargo:rerun-if-changed=PATH — Tells Cargo when to re-run the script.
    // cargo:rerun-if-env-changed=VAR — Tells Cargo when to re-run the script.
    // cargo:rustc-link-arg=FLAG — Passes custom flags to a linker for benchmarks, binaries, cdylib crates, examples, and tests.
    // cargo:rustc-link-arg-bin=BIN=FLAG — Passes custom flags to a linker for the binary BIN.
    // cargo:rustc-link-arg-bins=FLAG — Passes custom flags to a linker for binaries.
    // cargo:rustc-link-arg-tests=FLAG — Passes custom flags to a linker for tests.
    // cargo:rustc-link-arg-examples=FLAG — Passes custom flags to a linker for examples.
    // cargo:rustc-link-arg-benches=FLAG — Passes custom flags to a linker for benchmarks.
    // cargo:rustc-link-lib=LIB — Adds a library to link.
    // cargo:rustc-link-search=[KIND=]PATH — Adds to the library search path.
    // cargo:rustc-flags=FLAGS — Passes certain flags to the compiler.
    // cargo:rustc-cfg=KEY[="VALUE"] — Enables compile-time cfg settings.
    // cargo:rustc-env=VAR=VALUE — Sets an environment variable.
    // cargo:rustc-cdylib-link-arg=FLAG — Passes custom flags to a linker for cdylib crates.
    // cargo:warning=MESSAGE — Displays a warning on the terminal.
    // cargo:KEY=VALUE — Metadata, used by links scripts.

    println!("cargo:rustc-link-lib=xblas");
}