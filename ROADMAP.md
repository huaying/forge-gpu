# Forge — Roadmap

## Overview

Forge development follows a milestone-based approach. Each milestone builds on the previous one, with a working demo at each stage.

---

## M1: Core Runtime & Type System (Q2 2026) ✅

**Goal:** Run a simple kernel on the GPU from Rust. Type-safe array operations.

**Status: COMPLETE** (April 12, 2026)

### Delivered

- [x] **`forge-core`**: Scalar types, `Vec2/3/4<T>`, `Mat22/33/44<T>`, `Quat<T>`
  - All math ops with operator overloading
  - 18 unit tests
- [x] **`forge-runtime`**: GPU device management
  - CUDA context via cudarc 0.19 (safe bindings)
  - `Array<T>` — GPU array with typed allocation, H2D/D2H copy
  - nvrtc JIT kernel compilation with OnceLock caching
  - `ForgeError` typed error handling
- [x] **`forge-macros`**: Kernel compilation
  - `#[kernel]` proc macro — parse Rust fn, emit CUDA C++, generate host launch wrapper
  - `#[func]` proc macro — device-callable `__device__` functions
  - Built-in function mapping: sin, cos, sqrt, abs, min, max, floor, ceil, round, exp, log, pow, atan2
  - Control flow: if/else, for loops (range-based), while loops
  - Launch variants: `launch()`, `launch_async()`, `launch_with_config()`, `launch_with_funcs()`
- [x] **GPU particle demo**: 100K particles, 300 steps, gravity + ground bounce
  - 1.09 billion particle-steps/s on L40
  - 54x faster than single-threaded CPU

### Not Delivered (deferred to M2+)

- [ ] `#[derive(ForgeType)]` for custom structs as kernel params
- [ ] `Array<Vec3f>` — custom types as array elements in kernels
- [ ] Memory pool (arena allocator)
- [ ] Stream/event synchronization primitives
- [ ] CI/CD pipeline
- [ ] Benchmarks (criterion)

### Demo (actual working code)

```rust
use forge_macros::kernel;
use forge_runtime::{Array, Device, cuda};

#[kernel]
fn integrate(
    pos_x: &mut Array<f32>, pos_y: &mut Array<f32>, pos_z: &mut Array<f32>,
    vel_x: &mut Array<f32>, vel_y: &mut Array<f32>, vel_z: &mut Array<f32>,
    dt: f32, gravity: f32, ground_y: f32, restitution: f32, n: i32,
) {
    let tid = thread_id();
    if tid < n {
        vel_y[tid] = vel_y[tid] + gravity * dt;
        pos_x[tid] = pos_x[tid] + vel_x[tid] * dt;
        pos_y[tid] = pos_y[tid] + vel_y[tid] * dt;
        pos_z[tid] = pos_z[tid] + vel_z[tid] * dt;
        if pos_y[tid] < ground_y {
            pos_y[tid] = ground_y;
            vel_y[tid] = vel_y[tid] * restitution * -1.0;
        }
    }
}

// Run: cargo run --example particles --release
```

### Metrics (measured)
- Kernel launch + 300 steps: 0.028s for 100K particles (L40)
- JIT compilation (first launch): ~200ms
- 48 tests, all passing

---

## M2: Autodiff + Spatial Queries (Q3 2026)

**Goal:** Differentiable simulation. BVH and hash grid for spatial queries.

### Deliverables

- [ ] **Autodiff**
  - `Tape` type for recording kernel launches
  - `#[kernel(autodiff)]` generates forward + adjoint kernels
  - `tape.backward()` — reverse-mode automatic differentiation
  - Gradient accumulation via atomic adds
  - Support for control flow in adjoint (select-based)
- [ ] **`forge-spatial`**: Spatial data structures
  - `Bvh` — bounding volume hierarchy (build, refit, query)
  - `HashGrid` — spatial hash grid for particle neighbor queries
  - `Mesh` — triangle mesh (point queries, ray cast, closest point)
  - All structures work in kernels via device-side query APIs
- [ ] **PyTorch interop** (`forge-interop`)
  - DLPack zero-copy tensor sharing
  - PyO3 bindings for calling Forge kernels from Python
  - `torch.autograd.Function` wrapper for end-to-end gradients
- [ ] **Sparse matrices**
  - CSR and BSR formats
  - SpMV, SpMM operations
  - Integration with autodiff

### Demo

```rust
// M2 demo: differentiable cloth simulation
#[kernel(autodiff)]
fn spring_energy(x: &Array<Vec3f>, springs: &Array<[u32; 2]>,
                 rest: &Array<f32>, energy: &mut Array<f32>) {
    let tid = thread_id();
    let (i, j) = (springs[tid][0], springs[tid][1]);
    let d = length(x[i] - x[j]) - rest[tid];
    energy[tid] = 0.5 * d * d;
}

// Backprop through the simulation to optimize rest lengths
let tape = Tape::new();
tape.record(|| spring_energy.launch(n, &x, &springs, &rest, &mut energy));
tape.backward(&energy);
let grad_x = tape.gradient(&x);
```

---

## M3: Multi-Backend + FEM + Advanced Features (Q4 2026)

**Goal:** Run on AMD and Apple GPUs. Finite element support.

### Deliverables

- [ ] **Multi-backend codegen**
  - Backend trait abstraction in `forge-codegen`
  - ROCm/HIP code generation
  - Metal Shading Language generation
  - CPU fallback via Rayon
- [ ] **`forge-fem`**: Finite Element Method
  - Mesh function spaces (Lagrange elements)
  - Integration on cells, faces, edges
  - Dirichlet boundary conditions
  - PDE solvers (Poisson, elasticity, Navier-Stokes)
- [ ] **CUDA Graphs**
  - Graph capture from tape
  - Graph replay for repeated launch patterns
- [ ] **Tile operations**
  - Shared memory tiles
  - Tile-level matrix multiply (WMMA/tensor core)
  - Tile reductions
- [ ] **Volume support**
  - NanoVDB integration for sparse volumes
  - Volume sample, lookup, transform

### Demo

```rust
// M3 demo: same kernel, multiple backends
let device = Device::best_available(); // picks CUDA, ROCm, or Metal

#[kernel]
fn heat_diffusion(temp: &mut Array<f32>, neighbors: &Array<[u32; 4]>,
                  alpha: f32, dt: f32) {
    let tid = thread_id();
    let n = neighbors[tid];
    let laplacian = temp[n[0]] + temp[n[1]] + temp[n[2]] + temp[n[3]] - 4.0 * temp[tid];
    temp[tid] = temp[tid] + alpha * dt * laplacian;
}

heat_diffusion.launch_on(device, grid_size, &mut temp, &neighbors, alpha, dt);
```

---

## M4: Declarative Layer (Q1 2027)

**Goal:** AI agents generate TOML simulation manifests instead of code.

### Deliverables

- [ ] **`forge-manifest`**: Declarative simulation spec
  - TOML-based simulation description
  - Schema validation at parse time
  - Compile manifest → optimized simulation binary
- [ ] **`forge` CLI**
  - `forge run sim.toml` — compile and run
  - `forge check sim.toml` — validate without running
  - `forge export sim.toml --format usd/obj/vdb`
- [ ] **Template library**
  - Pre-built simulation templates (particles, cloth, fluid, rigid body)
  - Composable building blocks

### Demo

```toml
# M4 demo: AI-generated simulation manifest
[simulation]
name = "particle-rain"
dt = 0.001
substeps = 4
duration = 10.0

[[fields]]
name = "position"
type = "vec3f"
init = { type = "random", min = [-5, 10, -5], max = [5, 20, 5] }
count = 100000

[[forces]]
type = "gravity"

[[constraints]]
type = "ground_plane"
y = 0.0
restitution = 0.7

[output]
format = "usd"
fps = 60
```

```bash
$ forge run particle-rain.toml
Compiling particle-rain... done (0.3s)
Running 10.0s simulation with 100,000 particles...
[████████████████████████████████████████] 100% (2.1s)
Exported: particle-rain.usd (600 frames)
```

---

## Beyond M4

Ideas for future development:

- **forge-render**: Real-time visualization (wgpu-based)
- **forge-cloud**: Distributed simulation across multiple GPUs/nodes
- **forge-studio**: Visual editor for simulation manifests
- **WebGPU/WGSL**: Browser-based simulations
- **LLM integration**: Natural language → forge manifest pipeline
- **USD/Omniverse**: Native USD stage integration

---

## How to Contribute

Each milestone has its own tracking issue. Pick a deliverable, open a PR, and let's build this.

We especially welcome contributions in:
- 🧪 Testing (property-based tests, fuzzing)
- 📖 Documentation & examples
- 🔧 Backend implementations (ROCm, Metal)
- 🧮 Math correctness validation against Warp
