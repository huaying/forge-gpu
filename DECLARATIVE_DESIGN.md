# Forge Declarative Language — Design Document

> **Status**: RFC (Request for Comments)
> **Goal**: Design a TOML-based declarative simulation language that covers 90% of use cases via config, and allows custom Rust/CUDA code for the remaining 10%.

---

## 1. Design Principles

1. **Config for common, code for uncommon** — If 100 users need gravity, make it a one-liner. If 1 user needs a weird force field, let them write a kernel.
2. **Modular composition** — Build complex simulations by snapping together modules. Cloth = grid mesh + springs + gravity + collision. Fluid = particles + SPH + pressure solve.
3. **Progressive disclosure** — Simple sims are simple. Complex sims are possible. You don't see complexity until you need it.
4. **AI-writable** — An LLM should be able to generate a valid manifest from a natural language description. That means: clear schema, no hidden state, self-documenting.

---

## 2. Scene Survey — What do people actually simulate?

### Tier 1: Particle Systems (easiest)
| Scene | Components | Difficulty |
|-------|-----------|-----------|
| Rain/snow | particles + gravity + ground | ⭐ |
| Fireworks | particles + gravity + emission + lifetime | ⭐ |
| Sparks | particles + gravity + drag + bounce | ⭐ |
| Sand/granular | particles + gravity + friction + collision | ⭐⭐ |

### Tier 2: Spring-Mass Systems
| Scene | Components | Difficulty |
|-------|-----------|-----------|
| Rope | 1D chain + springs + gravity | ⭐⭐ |
| Cloth | 2D grid + springs + gravity + collision | ⭐⭐⭐ |
| Hair | Strand curves + bending + collision | ⭐⭐⭐⭐ |
| Soft body | Tetrahedral mesh + volume springs | ⭐⭐⭐ |

### Tier 3: Fluids
| Scene | Components | Difficulty |
|-------|-----------|-----------|
| SPH fluid | particles + neighbor query + pressure + viscosity | ⭐⭐⭐ |
| FLIP fluid | particles + grid + pressure solve | ⭐⭐⭐⭐ |
| Smoke/fire | grid + advection + buoyancy + diffusion | ⭐⭐⭐⭐ |

### Tier 4: Rigid Bodies & Contact
| Scene | Components | Difficulty |
|-------|-----------|-----------|
| Stacking boxes | rigid bodies + contact + friction | ⭐⭐⭐⭐ |
| Ragdoll | articulated body + joints + contact | ⭐⭐⭐⭐⭐ |

### Tier 5: Multi-Physics
| Scene | Components | Difficulty |
|-------|-----------|-----------|
| Cloth in wind + on body | cloth + aerodynamics + body collision | ⭐⭐⭐⭐ |
| Fluid + floating objects | SPH + rigid body coupling | ⭐⭐⭐⭐⭐ |

---

## 3. Module System — The Building Blocks

Key insight: every simulation is a **pipeline of modules**, and every module operates on **fields** (arrays of data on GPU).

### 3.1 Core Concepts

```
Field     = named GPU array (pos: Vec3f[N], vel: Vec3f[N], mass: f32[N])
Module    = a unit of computation (reads fields, writes fields)
Pipeline  = ordered sequence of modules executed per timestep
Emitter   = creates new particles (for particle systems)
Collider  = geometric object for collision detection
```

### 3.2 Module Categories

#### Integrators (how to advance time)
| Module | What it does |
|--------|-------------|
| `symplectic_euler` | vel += force * dt; pos += vel * dt |
| `verlet` | position-based Verlet integration |
| `xpbd` | Extended Position-Based Dynamics (modern cloth/soft body) |
| `implicit_euler` | For stiff systems (needs linear solve) |

#### Forces (what pushes things)
| Module | Params | Reads | Writes |
|--------|--------|-------|--------|
| `gravity` | direction, magnitude | mass | force |
| `drag` | coefficient | vel | force |
| `wind` | direction, strength, turbulence | pos, vel | force |
| `spring` | stiffness, damping, rest_length | pos, vel, spring_pairs | force |
| `bending` | stiffness | pos, triangle_pairs | force |
| `pressure` | gas_constant, rest_density | pos, density | force |
| `viscosity` | coefficient | vel, neighbors | force |
| `custom` | expr / kernel_path | user-defined | user-defined |

#### Constraints (what limits things)
| Module | Params |
|--------|--------|
| `ground_plane` | y, restitution, friction |
| `sphere_collider` | center, radius, restitution |
| `box_collider` | min, max, restitution |
| `mesh_collider` | mesh_path, restitution |
| `distance_constraint` | pairs, rest_length, stiffness |
| `volume_constraint` | tets, rest_volume, stiffness |
| `pin` | particle_indices, positions |

#### Spatial Queries (acceleration structures)
| Module | When needed |
|--------|-------------|
| `hashgrid` | SPH fluid, granular |
| `bvh` | mesh collision, ray cast |

#### Emitters (particle creation)
| Module | Params |
|--------|--------|
| `point_emitter` | position, rate, velocity, lifetime |
| `mesh_emitter` | mesh surface, rate |
| `volume_emitter` | box/sphere region, count |

---

## 4. Proposed TOML Schema

### 4.1 Simple: Particle Rain

```toml
[simulation]
name = "particle-rain"
dt = 0.001
substeps = 4
duration = 5.0

[particles]
count = 100_000
position = { init = "random", min = [-5, 10, -5], max = [5, 20, 5] }
velocity = { init = "zero" }

[[pipeline]]
module = "gravity"

[[pipeline]]
module = "integrate"
method = "symplectic_euler"

[[pipeline]]
module = "ground_plane"
y = 0.0
restitution = 0.7
```

### 4.2 Medium: Cloth on Sphere

```toml
[simulation]
name = "cloth-on-sphere"
dt = 0.0001
substeps = 20
duration = 3.0

[particles]
count = 10000
position = { init = "grid", spacing = 0.01, origin = [0, 2, 0], dims = [100, 100] }
velocity = { init = "zero" }
mass = { init = "constant", value = 0.001 }
inv_mass = { init = "constant", value = 1000.0 }

[topology]
# Auto-generate spring connections from grid
springs = { type = "grid", structural = true, shear = true, bend = true }

[[pipeline]]
module = "gravity"

[[pipeline]]
module = "spring"
stiffness = 5000.0
damping = 10.0

[[pipeline]]
module = "integrate"
method = "symplectic_euler"

[[pipeline]]
module = "sphere_collider"
center = [0, 0.5, 0]
radius = 0.5
friction = 0.3

# Pin top-left and top-right corners
[[pipeline]]
module = "pin"
indices = [0, 99]
```

### 4.3 Hard: SPH Fluid

```toml
[simulation]
name = "dam-break"
dt = 0.0001
substeps = 10
duration = 2.0

[particles]
count = 50000
position = { init = "box", min = [0, 0, 0], max = [1, 2, 0.5] }
velocity = { init = "zero" }
mass = { init = "constant", value = 0.001 }
density = { init = "constant", value = 1000.0 }
pressure = { init = "zero" }

[spatial]
type = "hashgrid"
cell_size = 0.05  # ~2x smoothing radius
grid_dims = [40, 60, 20]

[[pipeline]]
module = "sph_density"        # compute density from neighbors
smoothing_radius = 0.025

[[pipeline]]
module = "sph_pressure"       # equation of state: P = k * (rho - rho0)
gas_constant = 2000.0
rest_density = 1000.0

[[pipeline]]
module = "sph_viscosity"
coefficient = 0.01

[[pipeline]]
module = "gravity"

[[pipeline]]
module = "integrate"
method = "symplectic_euler"

[[pipeline]]
module = "box_collider"
min = [-0.1, 0, -0.1]
max = [2.0, 3.0, 0.6]
restitution = 0.3
```

### 4.4 Extreme: Custom Kernel Escape Hatch

```toml
[simulation]
name = "custom-force-field"
dt = 0.001
duration = 10.0

[particles]
count = 100_000
position = { init = "random", min = [-10, -10, -10], max = [10, 10, 10] }
velocity = { init = "zero" }

[[pipeline]]
module = "gravity"

# Custom force: attract to origin with 1/r² falloff
[[pipeline]]
module = "custom"
kernel = "kernels/attractor.rs"   # path to Rust source with #[kernel]
reads = ["position"]
writes = ["velocity"]
params = { strength = 10.0, center = [0, 0, 0] }

[[pipeline]]
module = "integrate"
method = "symplectic_euler"
```

And `kernels/attractor.rs`:
```rust
use forge_macros::kernel;
use forge_runtime::Array;

#[kernel]
fn attractor(
    px: &Array<f32>, py: &Array<f32>, pz: &Array<f32>,
    vx: &mut Array<f32>, vy: &mut Array<f32>, vz: &mut Array<f32>,
    cx: f32, cy: f32, cz: f32,
    strength: f32, dt: f32, n: i32,
) {
    let i = thread_id();
    if i < n {
        let dx = cx - px[i];
        let dy = cy - py[i];
        let dz = cz - pz[i];
        let r2 = dx*dx + dy*dy + dz*dz + 0.01;
        let inv_r = 1.0 / sqrt(r2);
        let f = strength * inv_r * inv_r;
        vx[i] = vx[i] + dx * inv_r * f * dt;
        vy[i] = vy[i] + dy * inv_r * f * dt;
        vz[i] = vz[i] + dz * inv_r * f * dt;
    }
}
```

---

## 5. Architecture

```
                    ┌─────────────────┐
                    │   TOML Manifest  │ ← AI writes this
                    └────────┬────────┘
                             │ parse
                    ┌────────▼────────┐
                    │   SimManifest    │ ← validated schema
                    └────────┬────────┘
                             │ compile
                    ┌────────▼────────┐
                    │  Pipeline Plan   │ ← ordered module list + field layout
                    └────────┬────────┘
                             │ resolve
                    ┌────────▼────────┐
                    │  Module Registry │ ← builtin kernels + custom kernels
                    │                  │
                    │  gravity ────→ CUDA kernel (cached)
                    │  spring  ────→ CUDA kernel (cached)
                    │  custom  ────→ JIT compile from .rs
                    └────────┬────────┘
                             │ execute
                    ┌────────▼────────┐
                    │  GPU Execution   │ ← field arrays on GPU
                    │  for each step:  │
                    │    module1(fields)│
                    │    module2(fields)│
                    │    ...           │
                    └──────────────────┘
```

### Key Design Decision: Module = Function over Fields

Every module has the same interface:
```rust
trait SimModule {
    fn name(&self) -> &str;
    fn reads(&self) -> &[&str];      // which fields it reads
    fn writes(&self) -> &[&str];     // which fields it writes
    fn execute(&self, fields: &mut FieldSet, dt: f32) -> Result<(), ForgeError>;
}
```

This means:
- **Automatic dependency tracking** — we know which modules can run in parallel
- **Validation** — we can check at parse time that all required fields exist
- **Composability** — any module can be used in any simulation

---

## 6. What Needs to Be Built

### Phase 1: Foundation ✅
- [x] Module trait + registry
- [x] Field system (named GPU arrays)
- [x] Pipeline executor
- [x] Builtin modules: gravity, integrate, ground_plane

### Phase 2: Spring/Cloth ✅
- [x] Topology system (spring pairs, grid generation)
- [x] Spring module + bending module
- [x] Pin constraint
- [x] Sphere/box colliders
- [x] Cloth demo manifest

### Phase 3: SPH Fluid ✅
- [x] GPU HashGrid build in pipeline (counting sort + Blelloch prefix sum)
- [x] SPH density, pressure, viscosity modules
- [x] Automatic kernel fusion (3 modules → 2 passes)
- [x] Shared memory tiling for SPH kernels
- [x] Box collider for fluid containment
- [x] Dam break demo manifest (50K particles, 2.22e8 p-steps/s)

### Phase 4: Custom Kernels
- [ ] Custom module loader (parse .rs, JIT compile)
- [ ] Param passing from TOML → kernel args
- [ ] Hot reload (recompile on file change)

### Phase 5: Output
- [ ] Frame export (JSON, binary)
- [ ] USD export
- [ ] Live visualization (future)

---

## 7. Open Questions

1. **Expression language vs custom .rs files?**
   - Expressions are more AI-friendly (`expr = "vel += force * dt"`)
   - .rs files are more powerful (full Rust + proc macros)
   - Could support both: simple expressions inline, complex logic in .rs

2. **Multi-physics coupling?**
   - Cloth + rigid body needs bidirectional force exchange
   - Current pipeline is one-directional — need "interaction modules"?

3. **Adaptive timestep?**
   - Some simulations need dt to vary (CFL condition for fluids)
   - Current schema has fixed dt

4. **GPU memory management?**
   - Particle emission/deletion means dynamic array sizing
   - Current Array is fixed-size — need a growable GPU buffer?

5. **Differentiation through manifests?**
   - Can we make manifest simulations differentiable automatically?
   - Would enable: "optimize rest lengths to match target shape"

---

## 8. Success Criteria

The declarative language is "done" when:

1. ✅ A simple particle sim runs from TOML (particle-rain.toml)
2. ✅ A cloth sim runs from TOML (cloth-on-sphere.toml, no custom code)
3. ✅ A SPH fluid runs from TOML (dam-break.toml, no custom code)
4. 🔲 A custom force field works via .rs escape hatch
5. 🔲 An LLM can generate a valid manifest from "simulate cloth falling on a sphere"
