# Forge — Design Document

> A Rust-native GPU compute framework designed for AI-generated code.

## 1. Vision

Forge is a ground-up rewrite of the GPU compute paradigm pioneered by NVIDIA Warp, rebuilt in Rust with a singular design principle:

> **Narrow the space of valid programs so that AI agents can generate correct GPU code on the first try.**

Where Warp chose Python for human ergonomics, Forge chooses Rust for **machine ergonomics** — leveraging the compiler as an automatic correctness checker. Wrong code doesn't crash at runtime; it doesn't compile.

### Why Now?

The world is shifting from "humans write code" to "AI writes code, humans review it." This changes the calculus:

| Concern | Human-first (Warp) | AI-first (Forge) |
|---|---|---|
| Learning curve | Must be gentle → Python | Irrelevant → Rust is fine |
| Iteration speed | Must be fast → dynamic typing | AI generates in seconds → static typing is free |
| Error messages | Must be readable | Must be *actionable* → compiler errors are structured |
| Strict types | Burden on the programmer | **Gift to the AI** — fewer wrong programs to generate |
| Boilerplate | Annoying for humans | AI doesn't care about boilerplate |

### The Kubernetes Analogy

Kubernetes succeeded because it's **declarative and opinionated**. You describe desired state; the system enforces it. AI excels at generating declarations, struggles with managing mutable state.

Forge applies this insight to GPU computing:
- **Declare** your simulation (types, fields, constraints)
- **Forge compiles** it into optimal GPU kernels
- **The compiler** catches mistakes before anything touches the GPU

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    User / AI Agent                   │
│              (Rust code or TOML manifests)           │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                 forge (top-level crate)              │
│            Re-exports, macros, #[kernel] proc macro  │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼──────┐ ┌─────▼─────┐ ┌─────▼──────┐
│  forge-core  │ │forge-codegen│ │forge-runtime│
│  Types, math │ │ AST→PTX/   │ │ Device mgmt │
│  vec/mat/quat│ │ SPIR-V gen │ │ Memory, sync│
│  array, dtype│ │ Autodiff   │ │ Kernel launch│
└──────────────┘ └───────────┘ └─────────────┘
                       │              │
                 ┌─────▼─────┐ ┌─────▼──────┐
                 │forge-spatial│ │forge-interop│
                 │ BVH, mesh  │ │ PyTorch,    │
                 │ hash grid  │ │ NumPy, JAX  │
                 │ SDF, volume│ │ DLPack      │
                 └───────────┘ └────────────┘
```

### Crate Responsibilities

| Crate | Warp Equivalent | Purpose |
|-------|----------------|---------|
| `forge` | `import warp as wp` | Top-level API, proc macros (`#[kernel]`, `#[func]`) |
| `forge-core` | `types.py`, `builtins.py` | Scalar/vector/matrix types, arrays, dtype system |
| `forge-codegen` | `codegen.py` | Rust AST → CUDA/PTX/SPIR-V code generation, autodiff |
| `forge-runtime` | `warp.cpp`, `warp.cu`, `context.py` | GPU device management, memory, kernel dispatch |
| `forge-spatial` | `native/bvh.*`, `native/mesh.*`, `native/hashgrid.*` | Spatial data structures (BVH, hash grid, mesh queries) |
| `forge-interop` | `torch.py`, `jax.py`, `dlpack.py` | Framework interop via DLPack, C FFI, PyO3 |

## 3. Core Design Decisions

### 3.1 Type System — Compile-Time Safety

Warp's type system lives in Python — errors surface at kernel launch time (or worse, silently produce wrong results). Forge pushes everything to compile time.

```rust
// Forge: wrong types = won't compile
use forge::prelude::*;

#[kernel]
fn integrate(
    pos: &mut Array<Vec3f>,     // typed, dimensioned
    vel: &Array<Vec3f>,         // immutable reference = read-only
    dt: f32,
) {
    let tid = thread_id();
    pos[tid] = pos[tid] + vel[tid] * dt;
}

// This won't compile:
// pos[tid] = vel[tid] * dt;  // Vec3f + Vec3f * f32 → type mismatch if wrong
// vel[tid] = ...;             // vel is &Array (immutable) → compile error
```

**Key difference from Warp:** Mutability is encoded in the type system. In Warp, any array can be accidentally written to. In Forge, `&Array<T>` vs `&mut Array<T>` is enforced by the compiler.

### 3.2 Codegen — Proc Macros, Not AST Walking

Warp walks Python ASTs at runtime and emits C++ strings. This is fragile and slow. Forge uses Rust proc macros:

```
Warp:    Python source → AST parse → string C++ → nvcc → PTX → cubin
Forge:   Rust source → proc macro → CUDA IR → nvcc/ptx-compiler → cubin
                         (compile time)                (build time or JIT)
```

The `#[kernel]` proc macro:
1. Parses the Rust function's token stream
2. Validates types against the Forge type system
3. Generates CUDA C++ (or directly PTX via `ptx-builder`)
4. Generates the host-side launch wrapper with correct type signatures
5. Optionally generates the adjoint (reverse-mode) kernel for autodiff

All validation happens at **compile time**. If `cargo build` succeeds, the kernel is type-safe.

### 3.3 Memory Model — Ownership, Not GC

Warp relies on Python's GC for array lifetime management, with manual `device` annotations. Forge uses Rust's ownership:

```rust
// Array lives as long as it's in scope
let pos = Array::<Vec3f>::zeros(n, Device::Cuda(0));  // allocated on GPU 0
let vel = Array::<Vec3f>::zeros(n, Device::Cuda(0));

// Transfer
let pos_cpu = pos.to(Device::Cpu);  // explicit copy, new owner
// pos is still valid on GPU

// pos is dropped here → GPU memory freed automatically
// No leaks. No double-frees. No GC pauses.
```

### 3.4 Autodiff — Generated Adjoint Kernels

Warp's tape-based autodiff records kernel launches and replays them in reverse, using generated adjoint code. Forge does the same, but with compile-time adjoint generation:

```rust
#[kernel(autodiff)]  // generates both forward and adjoint kernels
fn spring_energy(
    x: &Array<Vec3f>,
    indices: &Array<[u32; 2]>,
    rest_length: &Array<f32>,
    energy: &mut Array<f32>,
) {
    let tid = thread_id();
    let (i, j) = (indices[tid][0], indices[tid][1]);
    let d = length(x[i] - x[j]) - rest_length[tid];
    energy[tid] = 0.5 * d * d;
}

// Tape records launches, backward() replays adjoints
let tape = Tape::new();
tape.record(|| {
    spring_energy.launch(n, &x, &indices, &rest, &mut energy);
});
tape.backward(&energy);
let grad_x = tape.gradient(&x);  // ∂energy/∂x
```

### 3.5 Multi-Backend — Not Just NVIDIA

Warp is NVIDIA-only (CUDA). Forge targets multiple backends:

| Backend | Status | How |
|---------|--------|-----|
| CUDA (NVIDIA) | Primary | PTX generation via `ptx-compiler` or `nvcc` |
| ROCm (AMD) | Planned | HIP code generation (structurally similar to CUDA) |
| Metal (Apple) | Planned | Metal Shading Language generation |
| Vulkan Compute | Planned | SPIR-V generation |
| CPU (fallback) | Day 1 | Direct Rust with rayon for parallelism |

The codegen layer abstracts over backends. Kernels are written once; the proc macro generates backend-specific code.

## 4. Type Catalog

### 4.1 Scalars

| Forge Type | CUDA Type | Description |
|------------|-----------|-------------|
| `f16` | `__half` | Half precision |
| `f32` | `float` | Single precision |
| `f64` | `double` | Double precision |
| `i8/i16/i32/i64` | `int8_t`... | Signed integers |
| `u8/u16/u32/u64` | `uint8_t`... | Unsigned integers |
| `bool` | `bool` | Boolean |

### 4.2 Vectors & Matrices

```rust
// Vectors: generic over dimension and scalar type
Vec2<T>  Vec3<T>  Vec4<T>
Vec2f = Vec2<f32>  // type aliases for common cases
Vec3d = Vec3<f64>

// Matrices: generic over rows, cols, scalar type
Mat22<T>  Mat33<T>  Mat44<T>
Mat33f = Mat33<f32>

// Quaternions
Quat<T>
Quatf = Quat<f32>

// Spatial types (for robotics)
Transform<T>       // position + orientation
SpatialVector<T>   // 6D twist/wrench
SpatialMatrix<T>   // 6x6 spatial inertia
```

### 4.3 Arrays

```rust
// Arrays are typed, dimensioned, and device-bound
Array<T, D = Cuda>           // 1D array
Array2d<T, D = Cuda>         // 2D array
Array3d<T, D = Cuda>         // 3D array
ArrayNd<T, N, D = Cuda>      // N-dimensional

// Indexed arrays (Warp's fabricarray equivalent)
IndexedArray<T, D>
```

### 4.4 Spatial Structures

```rust
Bvh              // Bounding Volume Hierarchy (ray/overlap queries)
HashGrid          // Spatial hash grid (neighbor queries)
Mesh              // Triangle mesh (closest point, ray cast)
Volume            // Sparse voxel volume (NanoVDB-backed)
MarchingCubes     // Isosurface extraction
```

## 5. Kernel Language

### 5.1 Thread Model

```rust
#[kernel]
fn my_kernel(data: &mut Array<f32>) {
    let tid = thread_id();          // 1D thread index
    let (tx, ty) = thread_id_2d();  // 2D grid
}

// Launch
my_kernel.launch(1024, &mut data);                    // 1D
my_kernel.launch_2d((32, 32), &mut data);             // 2D
my_kernel.launch_on(stream, 1024, &mut data);         // explicit stream
```

### 5.2 Builtins

The kernel language provides ~200 built-in functions matching Warp's:

- **Math:** `sin`, `cos`, `sqrt`, `pow`, `abs`, `min`, `max`, `clamp`, `lerp`, `smoothstep`
- **Vector:** `dot`, `cross`, `length`, `normalize`, `outer`
- **Matrix:** `mul`, `transpose`, `inverse`, `determinant`, `svd3`, `qr3`, `eig3`
- **Quaternion:** `quat_from_axis_angle`, `quat_rotate`, `quat_slerp`
- **Spatial:** `transform_point`, `transform_vector`, `spatial_cross`
- **Random:** `rand_f32`, `rand_vec3`, `noise_f32`, `noise_vec3` (Perlin)
- **Atomic:** `atomic_add`, `atomic_min`, `atomic_max`, `atomic_cas`
- **Printing:** `print!()` (GPU-side debug printing)

### 5.3 Control Flow

```rust
#[kernel]
fn example(a: &Array<f32>, out: &mut Array<f32>) {
    let tid = thread_id();

    // Conditionals — standard Rust
    if a[tid] > 0.0 {
        out[tid] = sqrt(a[tid]);
    } else {
        out[tid] = 0.0;
    }

    // Loops — standard Rust
    let mut sum = 0.0f32;
    for i in 0..10 {
        sum += a[tid] * (i as f32);
    }
    out[tid] = sum;
}
```

### 5.4 User Functions

```rust
#[func]  // callable from kernels, inlined at codegen
fn apply_gravity(vel: Vec3f, dt: f32) -> Vec3f {
    vel + Vec3f::new(0.0, -9.81 * dt, 0.0)
}

#[kernel]
fn simulate(vel: &mut Array<Vec3f>, dt: f32) {
    let tid = thread_id();
    vel[tid] = apply_gravity(vel[tid], dt);
}
```

## 6. Declarative Layer (Future — Milestone 4+)

The long-term vision: a declarative simulation specification that AI agents can generate like Kubernetes manifests.

```toml
# simulation.forge.toml — AI generates this

[simulation]
name = "cloth-drape"
dt = 0.001
substeps = 4
duration = 5.0

[[fields]]
name = "position"
type = "vec3f"
count = 10000

[[fields]]
name = "velocity"
type = "vec3f"
count = 10000

[[forces]]
type = "gravity"
acceleration = [0.0, -9.81, 0.0]

[[forces]]
type = "spring"
stiffness = 1000.0
damping = 10.0
mesh = "cloth.obj"

[[constraints]]
type = "pin"
indices = [0, 1, 2, 3]  # corners

[[colliders]]
type = "sphere"
center = [0.0, -1.0, 0.0]
radius = 2.0

[output]
format = "usd"
fps = 60
```

```bash
forge run simulation.forge.toml   # compiles + runs on GPU
forge export simulation.forge.toml --format usd
```

The compiler validates the manifest at build time: field counts match, types are compatible, forces reference valid fields, constraints reference valid indices.

**AI can generate these manifests confidently** — the schema is fixed, validation is exhaustive, and there's exactly one way to specify each thing.

## 7. Warp Feature Parity Matrix

| Warp Feature | Forge Equivalent | Priority |
|---|---|---|
| `@wp.kernel` | `#[kernel]` proc macro | M1 |
| `@wp.func` | `#[func]` proc macro | M1 |
| `wp.array` | `Array<T>` with ownership | M1 |
| `wp.launch()` | `kernel.launch(dim, ...)` | M1 |
| `wp.vec3`, `wp.mat33`, etc. | `Vec3f`, `Mat33f` — generic types | M1 |
| `wp.Tape` / autodiff | `Tape` + adjoint codegen | M2 |
| `wp.Mesh` | `Mesh` (BVH-backed) | M2 |
| `wp.HashGrid` | `HashGrid` | M2 |
| `wp.Volume` (NanoVDB) | `Volume` (NanoVDB wrapper) | M3 |
| `wp.sparse` | `SparseMatrix` (CSR/BSR) | M3 |
| `wp.fem` | `forge-fem` crate | M3 |
| `wp.render` | `forge-render` crate | M3 |
| `wp.Tape` (graph capture) | CUDA graph support | M3 |
| PyTorch interop | PyO3 + DLPack | M2 |
| JAX interop | XLA custom calls | M3 |
| Declarative TOML layer | `forge-manifest` crate | M4 |

## 8. What's Different from Warp (Summary)

| Aspect | Warp | Forge |
|--------|------|-------|
| Language | Python | Rust |
| Type checking | Runtime | Compile time |
| Memory safety | GC + manual | Ownership (Rust borrow checker) |
| Codegen | Python AST → C++ strings | Proc macro → CUDA IR |
| Mutability | Any array writable | `&` vs `&mut` enforced |
| Thread safety | Python GIL + locks | Rust `Send`/`Sync` |
| GPU backends | NVIDIA only | NVIDIA + AMD + Apple + Vulkan (planned) |
| Autodiff | Runtime tape + generated C++ | Compile-time adjoint generation |
| Target user | Human researchers | AI agents (with human review) |
| Philosophy | Pythonic, flexible | Opinionated, correct by construction |

## 9. Open Questions

1. **JIT vs AOT?** Warp JIT-compiles at first launch. Forge could do pure AOT (compile everything at `cargo build`), pure JIT (compile at first launch like Warp), or hybrid. Current plan: AOT by default, JIT opt-in for dynamic kernels.

2. **Python bindings?** Should Forge expose a Python API via PyO3 for researchers who want Rust safety but Python workflow? Likely yes, as a secondary interface.

3. **Kernel fusion?** Warp launches one kernel at a time. Forge could analyze the tape and fuse adjacent kernels. Complex but high payoff for performance.

4. **WGSL/WebGPU?** For browser-based simulations. Low priority but architecturally clean to add if the codegen abstraction is right.

---

*Document version: 0.1.0*
*Last updated: 2026-04-12*
*Author: Loyo (AI) + Hua-Ying Tsai*
