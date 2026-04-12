# Forge — Project Plan

> Living document tracking progress across all milestones.

## M1 Status: ✅ COMPLETE (April 12, 2026)

### Sprint History

| Sprint | Goal | Status | Key Deliverables |
|--------|------|--------|-----------------|
| Sprint 1 | CUDA wrapper + GPU memory | ✅ Done | cudarc 0.19, nvrtc JIT, Array\<T\>, L40 tested |
| Sprint 2 | `#[kernel]` proc macro | ✅ Done | Rust→CUDA emitter, 7 tests, OnceLock caching |
| Sprint 3 | `#[func]` + launch + demo | ✅ Done | `#[func]`, ForgeError, launch variants, GPU particles |

### Issues Closed
- #1 CUDA Driver API wrapper ✅
- #2 GPU Memory (Array\<T\>) ✅
- #3 `#[kernel]` proc macro ✅
- #4 `#[func]` proc macro ✅
- #5 Builtin math (Phase 1) ✅
- #6 Kernel launch dispatch ✅
- #10 Integration demo (particle sim) ✅

### Issues Deferred
- #7 Spatial types (Vec3f as kernel param) → M2
- #8 CI/CD setup → M2
- #9 Benchmarks (criterion) → M2

## Execution Order (M1)

The dependency graph for M1 dictates build order:

```
                    ┌──────────────┐
                    │ #8 CI/CD     │  ← can start immediately
                    └──────────────┘

                    ┌──────────────┐
                    │ #7 Spatial   │  ← can start immediately (types only)
                    │    Types     │
                    └──────────────┘

┌──────────────┐
│ #1 CUDA      │  ← START HERE
│    Wrapper   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ #2 GPU       │
│    Memory    │
└──────┬───────┘
       │
       ├───────────────────┐
       ▼                   ▼
┌──────────────┐   ┌──────────────┐
│ #3 #[kernel] │   │ #5 Builtins  │
│    Proc Macro│   │    (Phase 1) │
└──────┬───────┘   └──────┬───────┘
       │                   │
       ▼                   │
┌──────────────┐           │
│ #4 #[func]   │           │
└──────┬───────┘           │
       │                   │
       ├───────────────────┘
       ▼
┌──────────────┐
│ #6 Kernel    │
│    Launch    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ #10 🎯 Demo  │  ← M1 FINISH LINE
│    Particle  │
└──────────────┘

       │
       ▼
┌──────────────┐
│ #9 Benchmarks│  ← after demo works
└──────────────┘
```

## Recommended Sprint Plan

### Sprint 1 (Week 1-2): Foundation
**Goal:** CUDA wrapper + GPU memory working

| Issue | Task | Estimate |
|-------|------|----------|
| #1 | CUDA Driver API wrapper | 3-4 days |
| #2 | GPU memory (Array<T> on device) | 2-3 days |
| #8 | CI/CD setup | 1 day |
| #7 | Spatial types (can parallel) | 2 days |

**Exit criteria:** `Array<f32>::from_vec(data, Device::Cuda(0)).to_vec()` round-trips correctly.

### Sprint 2 (Week 3-4): Codegen
**Goal:** A simple kernel compiles and runs on GPU

| Issue | Task | Estimate |
|-------|------|----------|
| #3 | `#[kernel]` proc macro | 5-7 days |
| #5 | Builtin math (Phase 1) | 2-3 days |

**Exit criteria:** `#[kernel] fn add_one(data: &mut Array<f32>) { data[thread_id()] += 1.0; }` runs on GPU.

### Sprint 3 (Week 5-6): Launch + Integration
**Goal:** Full particle demo on GPU

| Issue | Task | Estimate |
|-------|------|----------|
| #4 | `#[func]` proc macro | 2-3 days |
| #6 | Kernel launch dispatch | 3-4 days |
| #10 | Integration demo | 2-3 days |
| #9 | Benchmarks | 2 days |

**Exit criteria:** 100K particle simulation running on GPU, matching Warp's output.

## Key Technical Decisions Needed

### 1. CUDA Bindings: `cudarc` vs raw FFI?

| Option | Pro | Con |
|--------|-----|-----|
| **cudarc** | Already safe, well-maintained, handles driver/runtime API | Extra dependency, may not expose everything we need |
| **Raw FFI** | Full control, no dependency | More boilerplate, must maintain safety ourselves |

**Recommendation:** Start with `cudarc` for speed, replace with raw FFI only if we hit limitations.

### 2. Codegen: nvrtc (runtime) vs nvcc (build-time)?

| Option | Pro | Con |
|--------|-----|-----|
| **nvrtc (JIT)** | No nvcc needed at build time, dynamic kernel generation | Runtime compilation latency on first launch |
| **nvcc (AOT)** | Pre-compiled, no first-launch latency | Requires CUDA toolkit at build time |
| **Hybrid** | Best of both: AOT by default, JIT for dynamic kernels | More complex |

**Recommendation:** Start with nvrtc (JIT) for faster iteration. Add AOT later via build script.

### 3. Proc macro crate structure

Need a separate `forge-macros` crate (proc macros must be in their own crate in Rust). Structure:

```
forge-macros/         # proc-macro crate (only exports #[kernel], #[func])
├── Cargo.toml        # proc-macro = true
└── src/
    ├── lib.rs        # proc macro entry points
    ├── kernel.rs     # #[kernel] expansion
    ├── func.rs       # #[func] expansion
    └── cuda_emit.rs  # Rust AST → CUDA C++ string
```

## Rust Crate Dependencies (Planned)

```toml
# forge-runtime
[dependencies]
cudarc = "0.12"           # Safe CUDA bindings

# forge-macros
[dependencies]
syn = { version = "2", features = ["full"] }
quote = "1"
proc-macro2 = "1"

# forge (top-level)
[dev-dependencies]
criterion = "0.5"         # Benchmarks
```

## Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| nvrtc compilation slow on first launch | Medium | High | Cache compiled kernels (hash-based, like Warp) |
| Proc macro debugging is painful | Medium | High | Extensive snapshot tests, `cargo expand` |
| `cudarc` missing features we need | Low | Medium | Fork or add raw FFI for specific calls |
| Complex kernels don't codegen correctly | High | Medium | Incremental: start simple, add features one at a time |
| Performance regression vs Warp | Medium | Low | Benchmark continuously, profile CUDA code |

---

*Last updated: 2026-04-12*
