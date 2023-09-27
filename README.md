# SCENIC+ core

Collection of common algorithms and functions for all SCENIC+ related tools.

## Build

```bash
# Install maturin and zig.
pip install maturin[zig]

# Build SCENIC+ core wheel
maturin build --release --compatibility manylinux2014 --zig
```

## Install

```bash
pip install --force-reinstall target/wheels/scenicplus_core-0.1.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```
