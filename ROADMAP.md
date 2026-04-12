# Forge — Roadmap

## Overview

Forge development follows a milestone-based approach. Each milestone builds on the previous one, with a working demo at each stage.

---

## M1: Core Runtime & Type System (Q2 2026)

**Goal:** Run a simple kernel on the GPU from Rust. Type-safe array operations.

### Deliverables

- [ ] **`forge-core`**: Scalar types, `Vec2/3/4<T>`, `Mat22/33/44<T>`, `Quat<T>`
  - All math ops with operator overloading
  - `#[derive(ForgeType)]` for custom structs
  - Comprehensive tests against Warp's math output
- [ ] **`forge-runtime`**: GPU device management
  - CUDA driver API wrapper (device enum, context, streams)
  - `Array<T>` — GPU array with typed allocation, H2D/D2H/D2D copy
  - Memory pool (arena allocator for reducing `cuMemAlloc` overhead)
  - Synchronization primitives (streams, events)
- [ ] **`forge-codegen`**: Basic kernel compilation
  - `#[kernel]` proc macro — parse Rust fn, emit CUDA C++
  - `#[func]` proc macro — device-side callable functions
  - Built-in function mapping (sin, cos, sqrt, dot, cross, normalize, etc.)
  - Control flow: if/else, for loops, while loops
  - Kernel launch with type-checked arguments
- [ ] **`forge`**: Top-level crate
  - `forge::prelude::*` with all common types
  - `Forge::init()` context management

### Demo

```rust
// M1 demo: 100K particle simulation (matching our Warp demo)
#[kernel]
fn integrate(pos: &mut Array<Vec3f>, vel: &mut Array<Vec3f>, dt: f32) {
    let tid = thread_id();
    vel[tid] = vel[tid] + Vec3f::new(0.0, -9.81 * dt, 0.0);
    pos[tid] = pos[tid] + vel[tid] * dt;
    if pos[tid].y < 0.0 {
        pos[tid] = Vec3f::new(pos[tid].x, 0.0, pos[tid].z);
        vel[tid] = Vec3f::new(vel[tid].x, -vel[tid].y * 0.7, vel[tid].z);
    }
}
```

### Metrics
- Kernel launch overhead within 2x of raw CUDA
- Compilation time < 5s for simple kernels
- Zero unsafe in user-facing API

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
