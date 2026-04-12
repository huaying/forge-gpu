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
use forge_runtime::{Array, Device, cuda};

#[kernel]
fn add_one(data: &mut Array<f32>, n: i32) {
    let i = thread_id();
    if i < n {
        data[i] += 1.0;
    }
}

fn main() {
    cuda::init();
    let n = 100_000;
    let mut data = Array::<f32>::zeros(n, Device::Cuda(0));

    add_one::launch(&mut data, n as i32, n, 0).unwrap();

    let result = data.to_vec();
    assert!((result[0] - 1.0).abs() < 1e-6);
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
├── forge-macros/      — #[kernel] and #[func] proc macros, Rust→CUDA emitter
└── forge-runtime/     — CUDA context, Array<T>, kernel compilation, launch dispatch
```

| Crate | What it does |
|-------|-------------|
| **forge-core** | Math types: `Vec2f`/`Vec3f`/`Vec4f`, `Mat33f`/`Mat44f`, `Quatf`, scalar ops. All with operator overloading. |
| **forge-macros** | `#[kernel]` transforms Rust fn → CUDA C++ `__global__` kernel + host launch wrapper. `#[func]` generates `__device__` functions. |
| **forge-runtime** | GPU runtime: CUDA context management (via cudarc 0.19), `Array<T>` with GPU memory, nvrtc JIT compilation, kernel launch dispatch. |
| **forge-codegen** | CUDA type mapping utilities. (Actual codegen lives in forge-macros.) |
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
- Array params: `&Array<T>` (read-only), `&mut Array<T>` (read-write)
- `thread_id()` → CUDA thread index
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

## What's NOT Implemented Yet

These are in the [DESIGN.md](DESIGN.md) and [ROADMAP.md](ROADMAP.md) but **not yet in code**:

- ❌ `Array<Vec3f>` as kernel parameter (custom struct codegen)
- ❌ Method-style launch (`kernel.launch(...)` — currently `kernel::launch(...)`)
- ❌ Autodiff / `Tape` / adjoint generation
- ❌ Spatial queries (BVH, HashGrid, Mesh)
- ❌ Multi-backend (ROCm, Metal, Vulkan)
- ❌ 2D/3D thread grids (`thread_id_2d()`)
- ❌ Atomic operations
- ❌ `f16` type
- ❌ Memory pool / arena allocator
- ❌ `#[derive(ForgeType)]` for custom structs
- ❌ CI/CD pipeline
- ❌ Python interop (PyO3 / DLPack)
- ❌ Declarative TOML layer

## Comparison with Warp

| | NVIDIA Warp | Forge (today) |
|---|---|---|
| Language | Python | Rust |
| Error detection | Runtime | **Compile time** |
| Memory safety | GC | **Ownership (borrow checker)** |
| Mutability | Any array writable | **`&` vs `&mut` enforced** |
| Kernel types | Scalars + vec/mat + structs | Scalars only (vec/mat planned) |
| Device functions | `@wp.func` | `#[func]` ✅ |
| Autodiff | ✅ Runtime tape | ❌ Not yet |
| Spatial queries | ✅ BVH, HashGrid, Mesh | ❌ Not yet |
| GPU backends | NVIDIA only | NVIDIA only (multi-backend planned) |
| Target | Human researchers | **AI agents + human review** |

## Roadmap

See [ROADMAP.md](ROADMAP.md) for detailed milestones.

| Milestone | Target | Status |
|-----------|--------|--------|
| M1: Core Runtime | Q2 2026 | ✅ **Complete** |
| M2: Autodiff + Spatial | Q3 2026 | 📋 Planned |
| M3: Multi-backend + FEM | Q4 2026 | 📋 Planned |
| M4: Declarative Layer | Q1 2027 | 💡 Design Phase |

## Contributing

Forge is in early development. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

<div align="center">
<i>Built by humans and AI, for AI and humans.</i>
</div>
