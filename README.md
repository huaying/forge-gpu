<div align="center">

# 🔥 Forge

**Rust-native GPU compute framework — faster than Warp, safer than CUDA.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.78+-orange.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

</div>

---

## Why Forge?

**GPU computing shouldn't require Python.** Forge lets you write GPU kernels in Rust with compile-time safety, zero-overhead abstractions, and a single binary deployment.

```rust
#[kernel]
fn saxpy(x: &Array<f32>, y: &mut Array<f32>, a: f32, n: i32) {
    let i = thread_id();
    if i < n {
        y[i] = a * x[i] + y[i];
    }
}

// If it compiles, it's correct.
saxpy::launch(&x, &mut y, 3.0, n, dim, 0)?;
```

## Benchmarks (vs NVIDIA Warp 1.12, L40 GPU)

| Benchmark | Warp (Python) | Forge (Rust) | |
|-----------|---------------|--------------|---|
| Kernel launch | 11 µs | **2.4 µs** | **4.6× faster** |
| SAXPY 100M | 1.577 ms | **1.573 ms** | tied |
| H2D copy 100M | 7.0 GB/s | **9.0 GB/s** | **29% faster** |
| D2H copy 1M | 12.4 GB/s | 11.8 GB/s | tied |
| First JIT | 318 ms | **< 1 ms** | **300× faster** |

Same GPU, same operations, no tricks. [Full benchmark script](benchmarks/warp_comparison.py).

## Quick Start

```bash
git clone https://github.com/huaying/forge-gpu.git
cd forge-gpu
cargo test --release --features cuda    # 76 tests
cargo run --release --example particles  # particle demo
```

Or run a simulation from TOML — no code needed:

```bash
forge run examples/dam-break.toml --serve 8080
# Open http://localhost:8080 for live WebGL viewer
```

## Features

### Core — `#[kernel]` + `#[func]`

Write GPU kernels in Rust. Forge compiles them to CUDA at build time.

```rust
#[kernel]
fn integrate(pos: &mut Array<Vec3f>, vel: &mut Array<Vec3f>, dt: f32, n: i32) {
    let tid = thread_id();
    if tid < n {
        vel[tid] = vel[tid] + Vec3f::new(0.0, -9.81 * dt, 0.0);
        pos[tid] = pos[tid] + vel[tid] * dt;
    }
}
```

**Supported in kernels:**
- All scalar types: `f16`/`f32`/`f64`, `i8`–`i64`, `u8`–`u64`, `bool`
- Vector types: `Vec2f`/`Vec3f`/`Vec4f` (f32 & f64 variants)
- Custom structs via `#[forge_struct]`
- Atomics: `atomic_add`/`min`/`max`/`cas`/`exch`/`sub`/`and`/`or`/`xor`
- Warp intrinsics: `shfl_down_sync`, `ballot_sync`, `all_sync`, `any_sync`
- 35+ math builtins: trig, exp/log, erf, cbrt, rsqrt, etc.
- PRNG: `rand_init`, `randf`, `randi`
- Thread indices: `thread_id()`, `tid_x()`/`tid_y()`/`tid_z()`
- Control flow: `if`/`else`, `for`, `while`

### Autodiff — `#[kernel(autodiff)]`

Reverse-mode automatic differentiation. Adjoint kernels generated at compile time.

```rust
#[kernel(autodiff)]
fn energy(pos: &Array<f32>, k: f32, rest: f32, out: &mut Array<f32>, n: i32) {
    let i = thread_id();
    if i < n {
        let dx = pos[i] - rest;
        out[i] = 0.5 * k * dx * dx;
    }
}
// Generates: energy::launch() + energy::launch_adjoint()
```

### CUDA Graphs — automatic capture + replay

```
Without graph: CPU→GPU→sync→CPU→GPU→sync→... (overhead per launch)
With graph:    CPU→[record]→GPU→GPU→GPU→...→[replay] (one-shot)
```

Forge auto-captures the simulation loop into a CUDA Graph. **37% speedup** on SPH.

### Multi-dim Arrays

```rust
let arr = Array::<f32>::zeros_nd(Shape::new_3d(64, 64, 64), Device::Cuda(0));
arr.reshape(Shape::new_2d(64, 4096));  // zero-copy
```

### Tile Primitives

Cooperative block-level operations using shared memory + warp shuffles:
- `tile_sum` / `tile_max` / `tile_min` — cooperative reductions
- `tile_matmul` — shared memory matrix multiply
- Tensor Core via inline PTX (`mma.sync.m16n8k16`, SM 8.0+)

### Spatial Queries

| Structure | Operations |
|-----------|-----------|
| `HashGrid` | Neighbor search (3×3×3 cell query) |
| `BVH` | Sphere query, closest point, ray cast |
| `Mesh` | Closest point (Voronoi), ray cast (Möller–Trumbore) |

### Streams & Events

```rust
let stream = cuda::new_stream(0);
let child = cuda::fork_stream(&stream);
// ... launch kernels on child ...
cuda::join_stream(&stream, &child);  // event-based sync
```

### PyTorch Interop

Zero-copy via `__cuda_array_interface__`:

```python
from forge_interop import ForgeArray
fa = ForgeArray.from_torch(tensor)       # torch → forge (zero-copy)
tensor = fa.to_torch()                    # forge → torch (zero-copy)
```

### Declarative Simulations

Describe physics in TOML, Forge compiles and runs it:

```toml
[simulation]
name = "dam-break"
dt = 0.0001
substeps = 10
duration = 2.0
count = 50000

[[forces]]
type = "sph_density"
smoothing_radius = 0.025

[[forces]]
type = "gravity"
value = [0.0, -9.81, 0.0]
```

```bash
$ forge run dam-break.toml
✅ 50K particles, 20K steps, 3.04e8 particle-steps/s
```

**Auto-optimizations** (transparent):
- Kernel fusion: 3 SPH passes → 2
- Shared memory tiling for neighbor queries
- CUDA Graph capture for the simulation loop
- GPU hash grid with prefix sum

## Architecture

```
forge-core/      Math types (Vec3f, Mat44f, Quat)
forge-macros/    #[kernel], #[func], #[forge_struct], #[kernel(autodiff)]
forge-runtime/   Array<T>, CUDA context, HashGrid, BVH, Mesh, Tape, Tiles
forge-manifest/  TOML simulation engine, 12 modules, forge CLI
forge-interop/   PyTorch zero-copy bridge
```

## What's Next

- [ ] NanoVDB volume queries
- [ ] Multi-backend (ROCm, Metal via wgpu)
- [ ] CI/CD + crates.io publish
- [ ] More demos (differentiable physics, cloth, robotics)

## License

Apache 2.0

---

<div align="center">
<i>Built for a world where AI writes the code and compilers catch the mistakes.</i>
</div>
