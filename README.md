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
use forge::prelude::*;

#[kernel]
fn integrate(
    pos: &mut Array<Vec3f>,
    vel: &Array<Vec3f>,
    dt: f32,
) {
    let tid = thread_id();
    pos[tid] = pos[tid] + vel[tid] * dt;
}

fn main() {
    let ctx = Forge::init();
    let mut pos = Array::<Vec3f>::zeros(100_000, Device::Cuda(0));
    let vel = Array::<Vec3f>::fill(100_000, Vec3f::new(0.0, -9.81, 0.0), Device::Cuda(0));

    integrate.launch(100_000, &mut pos, &vel, 1.0 / 60.0);
    ctx.synchronize();
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
| 🛡️ **Compile-time safety** | Wrong types, wrong mutability, wrong dimensions → compiler error, not GPU crash |
| 🧠 **Automatic differentiation** | `#[kernel(autodiff)]` generates adjoint kernels at compile time |
| 🔀 **Multi-backend** | CUDA today, ROCm/Metal/Vulkan planned — write once, run anywhere |
| 🧱 **Spatial queries** | BVH, hash grid, mesh queries — built in |
| 🤖 **AI-first design** | Opinionated API with one right way to do things — perfect for code generation |
| ⚡ **Zero-cost abstractions** | Rust's types vanish at runtime. Vec3f on GPU = 3 floats, period |

## Quick Start

```bash
# Requirements: Rust 1.78+, CUDA Toolkit 12.0+
cargo add forge-gpu
```

```rust
use forge::prelude::*;

#[kernel]
fn saxpy(a: f32, x: &Array<f32>, y: &mut Array<f32>) {
    let tid = thread_id();
    y[tid] = a * x[tid] + y[tid];
}

fn main() {
    let _ctx = Forge::init();
    let n = 1_000_000;

    let x = Array::<f32>::from_vec(vec![1.0; n], Device::Cuda(0));
    let mut y = Array::<f32>::from_vec(vec![2.0; n], Device::Cuda(0));

    saxpy.launch(n, 3.0, &x, &mut y);

    let result = y.to_vec();
    assert_eq!(result[0], 5.0);  // 3.0 * 1.0 + 2.0
}
```

## Architecture

```
forge (top-level)          — #[kernel], #[func] proc macros, re-exports
├── forge-core             — Vec3f, Mat33f, Quatf, Array<T>, dtype system
├── forge-codegen          — Rust AST → CUDA PTX / SPIR-V, autodiff
├── forge-runtime          — Device management, memory, kernel dispatch
├── forge-spatial          — BVH, HashGrid, Mesh, Volume
└── forge-interop          — PyTorch/NumPy/JAX via DLPack & PyO3
```

See [DESIGN.md](DESIGN.md) for the full architecture document.

## Comparison with Warp

| | NVIDIA Warp | Forge |
|---|---|---|
| Language | Python | Rust |
| Error detection | Runtime | **Compile time** |
| Memory safety | GC | **Ownership (borrow checker)** |
| Mutability | Any array writable | **`&` vs `&mut` enforced** |
| GPU backends | NVIDIA only | NVIDIA + AMD + Apple (planned) |
| Target | Human researchers | **AI agents + human review** |
| Autodiff | Runtime tape + codegen | **Compile-time adjoint generation** |
| Thread safety | Python GIL | **Rust `Send`/`Sync`** |

Forge isn't "Warp but in Rust." It's a reimagining of GPU computing for a world where AI generates the code and compilers verify it.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for detailed milestones.

| Milestone | Target | Status |
|-----------|--------|--------|
| M1: Core Runtime | Q2 2026 | 🔨 In Progress |
| M2: Autodiff + Spatial | Q3 2026 | 📋 Planned |
| M3: Multi-backend + FEM | Q4 2026 | 📋 Planned |
| M4: Declarative Layer | Q1 2027 | 💡 Design Phase |

## Philosophy

1. **If it compiles, it works.** The type system is your safety net.
2. **One way to do it.** Opinionated APIs reduce the space of possible bugs.
3. **AI-first, human-friendly.** Designed for code generation, readable by humans.
4. **Correct before fast.** Then make it fast. (Rust makes "fast" the default anyway.)

## Contributing

Forge is in early development. We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

<div align="center">
<i>Built by humans and AI, for AI and humans.</i>
</div>
