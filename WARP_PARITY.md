# Warp 1.12 Complete Feature Analysis — Forge Parity Roadmap

## Warp Architecture Overview

Warp is structured in 3 layers:
1. **Core** — kernel JIT, types, arrays, autodiff, CUDA graphs, tiles
2. **Domain modules** — warp.sim (physics), warp.fem (FEM), warp.sparse (linear algebra)
3. **Interop** — PyTorch, JAX, NumPy, USD, URDF/MJCF

---

## Layer 1: Core (Kernel Language + Runtime)

### 1.1 Types
| Feature | Warp | Forge | Gap |
|---------|------|-------|-----|
| Scalars: f16/f32/f64, i8-64, u8-64, bool | ✅ all | ✅ f32 only | Need f16/f64/int |
| vec2/3/4 (any scalar) | ✅ generic | ✅ f32 only | Need generic |
| mat22/33/44 (any scalar) | ✅ generic | ✅ f32 only | Need generic |
| quat | ✅ | ✅ | OK |
| transform (pos + quat) | ✅ | ❌ | NEW |
| spatial_vector (6D) | ✅ | ❌ | NEW (robotics) |
| spatial_matrix (6×6) | ✅ | ❌ | NEW (robotics) |
| Custom structs in kernels | ✅ | ❌ | NEW |
| array (1D-4D, any dtype) | ✅ 1D-4D | ✅ 1D only | Need 2D-4D |
| indexed arrays | ✅ | ❌ | NEW |

### 1.2 Kernel Language
| Feature | Warp | Forge | Gap |
|---------|------|-------|-----|
| @wp.kernel → CUDA | ✅ Python→C++ | ✅ Rust→CUDA | OK (different approach) |
| @wp.func (device fn) | ✅ | ✅ #[func] | OK |
| Thread ID: wp.tid() (1-4D) | ✅ 1-4D | ✅ 1D only | Need multi-dim |
| Control flow (if/for/while) | ✅ | ✅ | OK |
| Atomic ops (add/min/max/cas) | ✅ all | ✅ atomicAdd only | Need min/max/CAS |
| printf in kernel | ✅ | ❌ | Nice-to-have |
| Dynamic indexing | ✅ | ❌ | NEW |
| Closures/generics | ✅ | ❌ | NEW |
| Overloaded functions | ✅ | ❌ (Rust handles this) | Different |

### 1.3 Math Builtins
| Category | Warp count | Forge count | Notes |
|----------|-----------|-------------|-------|
| Scalar math | 35+ (sin,cos,exp,log,erf,...) | ~10 | Need erf, cbrt, etc |
| Vector math | 30+ (cross,dot,normalize,svd,eig,...) | ~8 | Need SVD, eigen, outer, trace |
| Quaternion | 15+ | ~4 | Need slerp, euler, axis-angle |
| Transform | 12+ | 0 | Full transform algebra |
| Spatial (6D) | 8+ | 0 | Robotics-specific |
| Noise (Perlin) | 4 (noise, curlnoise, pnoise, ...) | 0 | Procedural generation |
| Random | 6 (rand_init, randf, randi, ...) | 0 | GPU random |

### 1.4 Runtime
| Feature | Warp | Forge | Gap |
|---------|------|-------|-----|
| CUDA Graphs | ✅ wp.ScopedCapture | ✅ Just added! | ✅ |
| Streams | ✅ multi-stream | ✅ single sim stream | Partial |
| Events | ✅ | ❌ | NEW |
| Launch objects (cached) | ✅ | ❌ (OnceLock similar) | Similar |
| CPU fallback | ✅ same kernel on CPU | ❌ GPU only | NEW |
| Multi-GPU | ✅ | ❌ | NEW |
| Peer access | ✅ | ❌ | NEW |

### 1.5 Autodiff
| Feature | Warp | Forge | Gap |
|---------|------|-------|-----|
| Reverse mode (adjoint) | ✅ | ✅ | OK |
| Forward mode | ✅ | ✅ | OK |
| wp.Tape (record/replay) | ✅ | ✅ Tape API | OK |
| Custom adjoint functions | ✅ | ❌ | NEW |
| Jacobian computation | ✅ | ❌ | NEW |
| Differentiable through copy/clone | ✅ | ❌ | NEW |
| Overwrite detection | ✅ | ❌ | NEW |

### 1.6 Tiles (Tensor Core access)
| Feature | Warp | Forge | Gap |
|---------|------|-------|-----|
| tile_load/store | ✅ | ❌ | NEW — big feature |
| tile_matmul (cuBLASDx) | ✅ | ❌ | NEW |
| tile_fft (cuFFTDx) | ✅ | ❌ | NEW |
| tile_cholesky | ✅ | ❌ | NEW |
| tile_map/reduce/sum | ✅ | ❌ | NEW |
| tile_sort/scan | ✅ | ❌ | NEW |
| Register + shared mem tiles | ✅ | ❌ (manual shared mem) | NEW |
| launch_tiled | ✅ | ❌ | NEW |

### 1.7 Spatial Queries
| Feature | Warp | Forge | Gap |
|---------|------|-------|-----|
| HashGrid | ✅ radix sort | ✅ counting sort | Need radix sort |
| BVH | ✅ | ✅ | OK |
| Mesh (ray/closest) | ✅ | ✅ | OK |
| NanoVDB volumes | ✅ | ❌ | NEW |
| Marching Cubes | ✅ | ❌ | NEW |

---

## Layer 2: Domain Modules

### 2.1 warp.sim (Physics Simulation)
| Feature | Warp | Forge | Gap |
|---------|------|-------|-----|
| Particles (position-based) | ✅ | ✅ SPH, springs, cloth | OK |
| Rigid bodies | ✅ (articulated) | ❌ | BIG gap |
| Soft bodies (FEM) | ✅ | ❌ | BIG gap |
| Cloth | ✅ | ✅ (spring-based) | Partial |
| Contacts/collisions | ✅ (ground, mesh, body) | ✅ (box, sphere, ground) | Partial |
| Joint types | ✅ (revolute, prismatic, D6, ...) | ❌ | NEW |
| Integrators | ✅ (Euler, XPBD, VBD, Featherstone) | ✅ (symplectic Euler) | Need XPBD |
| URDF/MJCF import | ✅ | ❌ | NEW |
| USD Physics import | ✅ | ❌ | NEW |
| Model builder | ✅ | ❌ (TOML manifest) | Different approach |

### 2.2 warp.fem (Finite Elements)
| Feature | Warp | Forge |
|---------|------|-------|
| Galerkin method | ✅ | ❌ |
| Geometries: Grid, Mesh, NanoVDB | ✅ | ❌ |
| Function spaces: Lagrange, Serendipity, B-spline, Nédélec, Raviart-Thomas | ✅ | ❌ |
| Bilinear/linear form integration | ✅ | ❌ |
| Sparse matrix assembly | ✅ | ❌ |
| Boundary conditions | ✅ | ❌ |
| Mixed FEM | ✅ | ❌ |

### 2.3 warp.sparse (Linear Algebra)
| Feature | Warp | Forge |
|---------|------|-------|
| BSR sparse matrix | ✅ | ❌ |
| SpMV, SpMM | ✅ | ❌ |
| Iterative solvers (CG, BiCGSTAB, GMRES) | ✅ | ❌ |
| Preconditioners | ✅ | ❌ |

---

## Layer 3: Interop

| Feature | Warp | Forge |
|---------|------|-------|
| NumPy (zero-copy) | ✅ | N/A (Rust) |
| PyTorch (zero-copy, autograd bridge) | ✅ | ❌ — Critical |
| JAX (DLPack, custom call) | ✅ | ❌ — Critical |
| __cuda_array_interface__ | ✅ | ❌ |
| Python bindings | ✅ (native) | ✅ (PyO3 basic) | Need expansion |

---

## Where Forge Can Beat Warp

### Already Better
1. **Declarative TOML manifests** — describe sim, don't code it
2. **Auto kernel fusion** — Warp doesn't fuse SPH passes
3. **Compile-time safety** — Rust catches errors at compile, Warp at runtime
4. **Zero-dep binary** — `forge run sim.toml`, no Python/pip/conda
5. **Built-in web viewer** — WebGL streaming out of the box
6. **CUDA Graph by default** — we auto-capture, Warp requires manual wp.ScopedCapture

### Potential Advantages
1. **Shared memory tile loading** — we already do manual shared mem in SPH; Warp tiles is newer/more abstract
2. **Faster JIT** — Rust proc-macro compile is instant vs Warp's Python→C++→nvcc pipeline
3. **Memory efficiency** — Rust ownership = no GC pauses, deterministic dealloc
4. **Embeddable** — Forge can be a library in any Rust/C++ app; Warp requires Python runtime

### Where Warp Will Be Hard to Beat
1. **Python ecosystem** — PyTorch/JAX/NumPy interop is table stakes for ML
2. **NVIDIA resources** — Warp has a team, MathDx access, internal CUDA expertise
3. **FEM toolkit** — years of math/physics implementation
4. **warp.sim** — complete articulated rigid body solver (Featherstone)
5. **Community + docs** — extensive examples, tutorials

---

## Recommended Priority (Forge Roadmap)

### Phase 1: Core Parity (Weeks 1-4)
1. ✅ CUDA Graphs
2. Multi-dim arrays (2D/3D)
3. More scalar types (f64, integers)
4. Atomic ops (min, max, CAS)
5. Random number generation in kernels
6. Radix sort for HashGrid (Warp parity)

### Phase 2: ML Bridge (Weeks 5-8)
7. PyTorch interop (DLPack / __cuda_array_interface__)
8. Python bindings (PyO3 expansion)
9. JAX custom call
10. Custom structs in kernels

### Phase 3: Tile Primitives (Weeks 9-12)
11. tile_load/store (register + shared mem)
12. tile_matmul (Tensor Core via cuBLASDx or inline PTX)
13. tile_fft
14. tile_reduce/scan

### Phase 4: Simulation (Weeks 13-20)
15. XPBD integrator
16. Rigid bodies (basic)
17. Joint system
18. Sparse matrix (BSR)
19. Iterative solvers (CG, BiCGSTAB)
20. FEM basics (Galerkin on triangles)
