<div align="center">

# 🔥 Forge

**A Rust-native GPU compute framework built for the age of AI.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.78+-orange.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

*Write GPU kernels in Rust. Let the compiler catch your mistakes.*

</div>

---

## What is Forge?

Forge is a GPU compute framework that brings Rust's compile-time safety guarantees to GPU programming. Inspired by [NVIDIA Warp](https://github.com/NVIDIA/warp), Forge reimagines GPU computing for a world where **AI agents write the code**.

```rust
use forge_macros::kernel;
use forge_core::Vec3f;
use forge_runtime::{Array, Device, cuda};

#[kernel]
fn integrate(
    pos: &mut Array<Vec3f>,
    vel: &mut Array<Vec3f>,
    dt: f32,
    gravity: f32,
    n: i32,
) {
    let tid = thread_id();
    if tid < n {
        vel[tid] = vel[tid] + Vec3f::new(0.0, gravity * dt, 0.0);
        pos[tid] = pos[tid] + vel[tid] * dt;
    }
}

fn main() {
    cuda::init();
    let n = 100_000;
    let mut pos = Array::from_vec(vec![Vec3f::new(0.0, 10.0, 0.0); n], Device::Cuda(0));
    let mut vel = Array::from_vec(vec![Vec3f::zero(); n], Device::Cuda(0));

    integrate::launch(&mut pos, &mut vel, 1.0/60.0, -9.81, n as i32, n, 0).unwrap();

    let result = pos.to_vec();
    println!("pos[0] = {:?}", result[0]); // particles falling under gravity
}
```

If this compiles, the kernel is type-safe. No runtime surprises.

## Why Forge?

### The Problem

GPU computing today is either:
- **Unsafe** — raw CUDA C++ with manual memory management, silent type mismatches, and segfaults on the GPU
- **Dynamic** — Python frameworks (Warp, Taichi, Numba) that catch errors at runtime, not compile time

### The Insight

AI writes most code now. AI doesn't care about Python's gentle learning curve. AI *benefits* from Rust's strict compiler — it's a free correctness checker that catches mistakes before anything runs on the GPU.

### What You Get

| Feature | Details |
|---------|---------|
| 🛡️ **Compile-time safety** | Wrong types, wrong mutability → compiler error, not GPU crash |
| 🔧 **`#[kernel]` proc macro** | Write Rust, runs as CUDA — JIT compiled via nvrtc |
| 🔧 **`#[func]` proc macro** | Device-callable helper functions |
| ⚡ **Typed GPU arrays** | `Array<T>` with ownership semantics and automatic memory management |
| 🚀 **Flexible launch** | `launch()`, `launch_async()`, `launch_with_config()`, `launch_with_funcs()` |
| 🤖 **AI-first design** | Opinionated API with one right way to do things — perfect for code generation |

## Quick Start

```bash
# Requirements: Rust 1.78+, CUDA Toolkit 12.0+, NVIDIA GPU
git clone https://github.com/huaying/forge-gpu.git
cd forge-gpu
cargo test --workspace
cargo run --example particles --release
```

## Architecture

```
forge-gpu/
├── forge/             — Top-level re-exports and prelude
├── forge-core/        — Vec2/3/4, Mat22/33/44, Quat, scalar types
├── forge-codegen/     — Type mapping utilities (CUDA type catalog)
├── forge-macros/      — #[kernel], #[func], #[kernel(autodiff)] proc macros
├── forge-runtime/     — CUDA context, Array<T>, HashGrid, BVH, Mesh, CsrMatrix
├── forge-manifest/    — Declarative TOML manifests, module system, forge CLI
└── forge-interop/     — PyTorch zero-copy interop (DLPack)
```

| Crate | What it does |
|-------|-------------|
| **forge-core** | Math types: `Vec2f`/`Vec3f`/`Vec4f`, `Mat33f`/`Mat44f`, `Quatf`, scalar ops. All with operator overloading. |
| **forge-macros** | `#[kernel]` transforms Rust fn → CUDA C++ `__global__` kernel + host launch wrapper. `#[func]` generates `__device__` functions. `#[kernel(autodiff)]` generates forward + adjoint kernels. |
| **forge-runtime** | GPU runtime: CUDA context management (via cudarc 0.19), `Array<T>` with GPU memory, nvrtc JIT compilation, kernel launch dispatch, `HashGrid`, `Bvh`, `Mesh`, `CsrMatrix`, `Tape`. |
| **forge-manifest** | Declarative TOML simulation engine: schema parsing, validation, 12 builtin modules, pipeline executor, automatic kernel fusion, `forge run/check` CLI. |
| **forge-codegen** | CUDA type mapping utilities. (Actual codegen lives in forge-macros.) |
| **forge-interop** | PyTorch tensor ↔ Forge array zero-copy via DLPack capsule protocol. |
| **forge** | Thin wrapper: `forge::prelude::*` re-exports common types. |

## What Works Today

### `#[kernel]` — Rust to CUDA

```rust
use forge_macros::kernel;
use forge_runtime::{Array, Device, cuda};

#[kernel]
fn scale(data: &mut Array<f32>, factor: f32, n: i32) {
    let i = thread_id();
    if i < n {
        data[i] = data[i] * factor;
    }
}

fn main() {
    cuda::init();
    let n = 1024;
    let mut data = Array::from_vec(vec![2.0f32; n], Device::Cuda(0));
    scale::launch(&mut data, 3.0, n as i32, n, 0).unwrap();
    // data is now all 6.0
}
```

**Supported in kernels:**
- Scalar types: `f32`, `f64`, `i32`, `u32`, `i64`, `u64`, `bool`
- **Vector types: `Vec2f`, `Vec3f`, `Vec4f`, `Vec2d`, `Vec3d`, `Vec4d`** — as array elements and in expressions
- Array params: `&Array<T>` (read-only), `&mut Array<T>` (read-write) — T can be scalar or Vec type
- `thread_id()` → CUDA thread index
- Vec constructors: `Vec3f::new(x, y, z)`, `Vec3f::zero()`, `Vec3f::splat(v)`
- Vec field access: `v.x`, `v.y`, `v.z`
- Vec arithmetic: `+`, `-`, `*` (scalar), `/` (scalar), negation
- Math ops: `+`, `-`, `*`, `/`, `%`, comparisons, compound assignment (`+=`, etc.)
- Control flow: `if`/`else`, `for` (range-based), `while`
- Builtins: `sin`, `cos`, `sqrt`, `abs`, `min`, `max`, `floor`, `ceil`, `round`, `exp`, `log`, `pow`, `atan2`
- Type casts: `x as f32`
- Method-style math: `x.abs()`, `x.sqrt()`, `x.sin()`, `x.min(y)`

### `#[func]` — Device Functions

```rust
use forge_macros::{func, kernel};
use forge_runtime::Array;

#[func]
fn clamp_val(x: f32, lo: f32, hi: f32) -> f32 {
    if x < lo { return lo; }
    if x > hi { return hi; }
    return x;
}

#[kernel]
fn apply_clamp(data: &mut Array<f32>, lo: f32, hi: f32, n: i32) {
    let i = thread_id();
    if i < n {
        data[i] = clamp_val(data[i], lo, hi);
    }
}

// Launch with device function sources:
apply_clamp::launch_with_funcs(
    &mut data, 0.0, 1.0, n as i32, n, 0,
    &[clamp_val::CUDA_SOURCE],
).unwrap();
```

### Launch Variants

```rust
// Standard launch (auto grid/block, sync after)
kernel::launch(&mut data, n as i32, n, 0)?;

// Async launch (no sync — caller must sync manually)
kernel::launch_async(&mut data, n as i32, n, 0)?;
cuda::synchronize(0);

// Custom launch config
use forge_runtime::cuda::LaunchConfig;
let config = LaunchConfig {
    grid_dim: (128, 1, 1),
    block_dim: (256, 1, 1),
    shared_mem_bytes: 0,
};
kernel::launch_with_config(&mut data, n as i32, 0, config)?;

// With device functions
kernel::launch_with_funcs(&mut data, n as i32, n, 0, &[my_func::CUDA_SOURCE])?;
```

### `Array<T>` — Typed GPU Arrays

```rust
use forge_runtime::{Array, Device, Forge};

// Initialize runtime (auto-detects best device)
let ctx = Forge::init();

// Create on GPU
let a = Array::<f32>::zeros(1000, Device::Cuda(0));
let b = Array::from_vec(vec![1.0f32; 1000], Device::Cuda(0));
let c = Array::fill(1000, 3.14f32, Device::Cuda(0));

// Transfer
let cpu_data = a.to_vec();          // GPU → CPU
let gpu_copy = a.to(Device::Cpu);   // copy to another device

// Metadata
a.len();      // element count
a.device();   // Device::Cuda(0)

// Sync all pending GPU operations
ctx.synchronize();
```

### Performance

GPU particle simulation (100K particles, 300 timesteps, L40 GPU):

| Mode | Throughput | Time |
|------|-----------|------|
| GPU (CUDA) | **1.09 × 10⁹** particle-steps/s | 0.028s |
| CPU (single-thread) | 7.47 × 10⁸ particle-steps/s | 0.040s |
| **Speedup** | **54×** | |

### Declarative TOML Simulations

No code required — describe your simulation, Forge compiles and runs it:

```toml
[simulation]
name = "dam-break"
dt = 0.0001
substeps = 10
duration = 2.0
count = 50000

[[fields]]
name = "position"
type = "vec3f"
count = 50000
init = { type = "random", min = [0.0, 0.0, 0.0], max = [1.0, 2.0, 0.5] }

[spatial]
type = "hashgrid"
cell_size = 0.05
grid_dims = [40, 60, 20]

[[forces]]
type = "sph_density"
smoothing_radius = 0.025

[[forces]]
type = "sph_pressure"
gas_constant = 2000.0
rest_density = 1000.0

[[forces]]
type = "sph_viscosity"
coefficient = 0.01

[[forces]]
type = "gravity"
value = [0.0, -9.81, 0.0]

[[constraints]]
type = "box"
min = [-0.1, 0.0, -0.1]
max = [2.0, 3.0, 0.6]
restitution = 0.3
```

```bash
$ forge run dam-break.toml
🔥 Forge Simulation Runner
  ⚡ SPH kernel fusion: density + pressure + viscosity → 2 passes (was 3)
  Pipeline: 4 modules
✅ Simulation complete!
  Steps: 20000
  Time: 4.509s
  Throughput: 2.22e8 particle-steps/s
```

**Automatic optimizations** (transparent to the user):
- **Kernel fusion**: SPH density + pressure + viscosity → 2 passes instead of 3
- **Shared memory tiling**: neighbor data loaded cooperatively into shared memory
- **GPU hash grid**: full GPU pipeline (cell index → count → prefix sum → scatter)

**12 builtin modules**: gravity, integrate, ground_plane, spring, sphere_collider, pin, drag, sph_density, sph_pressure, sph_viscosity, sph_fused, box_collider

## What's NOT Implemented Yet

These are in the [DESIGN.md](DESIGN.md) and [ROADMAP.md](ROADMAP.md) but **not yet in code**:

- ❌ Method-style launch (`kernel.launch(...)` — currently `kernel::launch(...)`)
- ❌ Multi-backend (ROCm, Metal, Vulkan)
- ❌ 2D/3D thread grids (`thread_id_2d()`)
- ❌ `f16` type
- ❌ Memory pool / arena allocator
- ❌ `#[derive(ForgeType)]` for custom structs
- ❌ CI/CD pipeline
- ❌ USD/OBJ export from `forge` CLI
- ❌ Expression language in TOML (inline `expr = "vel.y += ..."`)
- ❌ Custom kernel escape hatch from TOML (`.rs` file path)

## What IS Implemented

Beyond the core `#[kernel]`/`#[func]` system:

| Feature | Status |
|---------|--------|
| `#[kernel(autodiff)]` — reverse-mode AD | ✅ Scalar + Vec3f |
| `Tape` API — multi-step differentiation | ✅ |
| `HashGrid` — spatial hash for neighbor queries | ✅ CPU + GPU build |
| `BVH` — bounding volume hierarchy | ✅ Sphere/closest/ray queries |
| `Mesh` — triangle mesh with BVH | ✅ Closest point + ray cast |
| `CsrMatrix` — sparse matrix (CPU + GPU spmv) | ✅ |
| PyTorch interop (zero-copy DLPack) | ✅ |
| **Declarative TOML manifests** | ✅ |
| `forge run sim.toml` CLI | ✅ |
| 12 builtin simulation modules | ✅ |
| SPH fluid simulation (GPU) | ✅ |
| Automatic kernel fusion | ✅ |
| Shared memory optimization | ✅ |

## Comparison with Warp

| | NVIDIA Warp | Forge (today) |
|---|---|---|
| Language | Python | Rust |
| Error detection | Runtime | **Compile time** |
| Memory safety | GC | **Ownership (borrow checker)** |
| Mutability | Any array writable | **`&` vs `&mut` enforced** |
| Kernel types | Scalars + vec/mat + structs | **Scalars + Vec2/3/4** (custom structs planned) |
| Device functions | `@wp.func` | `#[func]` ✅ |
| Autodiff | ✅ Runtime tape | ✅ **Compile-time adjoint generation** |
| Spatial queries | ✅ BVH, HashGrid, Mesh | ✅ **BVH, HashGrid, Mesh** |
| Sparse linear algebra | ✅ | ✅ **CsrMatrix (CPU + GPU spmv)** |
| Declarative sim | ❌ | ✅ **TOML manifests + auto-optimization** |
| GPU backends | NVIDIA only | NVIDIA only (multi-backend planned) |
| Target | Human researchers | **AI agents + human review** |

## Roadmap

See [ROADMAP.md](ROADMAP.md) for detailed milestones.

| Milestone | Target | Status |
|-----------|--------|--------|
| M1: Core Runtime | Q2 2026 | ✅ **Complete** |
| M2: Autodiff + Spatial | Q3 2026 | ✅ **Complete** |
| M3: Multi-backend + FEM | Q4 2026 | 📋 Planned |
| M4: Declarative Layer | Q1 2027 | 🚧 **In Progress** (SPH fluid, kernel fusion, 12 modules) |

## Contributing

Forge is in early development. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

<div align="center">
<i>Built by humans and AI, for AI and humans.</i>
</div>
