# Forge TOML Manifest Reference

Complete reference for writing simulation manifests.

## Quick Start

```toml
[simulation]
name = "my-sim"
type = "particles"
dt = 0.001
substeps = 4
duration = 5.0
count = 10000

[[fields]]
name = "position"
type = "vec3f"
count = 10000
init = { type = "random", min = [-5, 10, -5], max = [5, 20, 5] }

[[fields]]
name = "velocity"
type = "vec3f"
init = { type = "zero" }

[[forces]]
type = "gravity"

[[constraints]]
type = "ground_plane"
y = 0.0
restitution = 0.7
```

```bash
forge run my-sim.toml
forge run my-sim.toml --serve 8080   # 3D viewer
```

---

## `[simulation]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | required | Simulation name |
| `type` | string | `"particles"` | `"particles"` or `"springs"` |
| `dt` | float | `0.001` | Time step (seconds) |
| `substeps` | int | `1` | Substeps per frame |
| `duration` | float | `5.0` | Total simulation time (seconds) |
| `count` | int | from fields | Particle count (can also be set per-field) |

---

## `[[fields]]`

Define simulation data arrays.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | required | `"position"`, `"velocity"`, or custom |
| `type` | string | required | `"vec3f"` or `"f32"` |
| `count` | int | from simulation | Number of elements |
| `init` | table | `{ type = "zero" }` | Initialization method |

### Init types

```toml
# Zero-initialized
init = { type = "zero" }

# Random uniform
init = { type = "random", min = [-5, 0, -5], max = [5, 10, 5] }

# Specific value (vec3f)
init = { type = "value", value = [0.0, 10.0, 0.0] }
```

Standard field names:
- `"position"` → stored as `pos_x`, `pos_y`, `pos_z`
- `"velocity"` → stored as `vel_x`, `vel_y`, `vel_z`

---

## `[[forces]]`

Forces applied each substep. Order matters for SPH.

### `gravity`

```toml
[[forces]]
type = "gravity"
value = [0.0, -9.81, 0.0]  # optional, default: [0, -9.81, 0]
```

### `drag`

```toml
[[forces]]
type = "drag"
coefficient = 0.01  # optional, default: 0.01
```

### `sph_density`

Compute particle density using Poly6 kernel. **Must come before sph_pressure/sph_viscosity.**

```toml
[[forces]]
type = "sph_density"
smoothing_radius = 0.025
```

### `sph_pressure`

Pressure gradient force using Spiky kernel.

```toml
[[forces]]
type = "sph_pressure"
gas_constant = 2000.0
rest_density = 1000.0
```

### `sph_viscosity`

Viscosity smoothing using Laplacian kernel.

```toml
[[forces]]
type = "sph_viscosity"
coefficient = 0.01
```

> **Auto-optimization:** When all three SPH forces are present, Forge automatically fuses them into 2 GPU passes instead of 3 (shared memory tiling, single neighbor traversal for pressure+viscosity).

### `custom`

Inline expression — compiled to CUDA at runtime.

```toml
[[forces]]
type = "custom"
expr = "vel.x += (-pos.z * 2.0) * dt; vel.z += (pos.x * 2.0) * dt"
```

**Available in expressions:**
- Fields: `pos.x`, `pos.y`, `pos.z`, `vel.x`, `vel.y`, `vel.z`, `density`
- Operators: `+`, `-`, `*`, `/`, `%`, `+=`, `-=`, `*=`, `=`
- Functions: `sin`, `cos`, `sqrt`, `abs`, `min`, `max`, `floor`, `ceil`, `exp`, `log`, `pow`, `atan2`, `tan`, `asin`, `acos`, `round`
- Constants: `pi`, `dt` (timestep), `n` (particle count)
- Multi-statement: separate with `;`

---

## `[[constraints]]`

Applied after forces + integration.

### `ground_plane`

```toml
[[constraints]]
type = "ground_plane"
y = 0.0          # height
restitution = 0.5  # bounciness (0-1)
```

### `sphere`

```toml
[[constraints]]
type = "sphere"
center = [0.0, 1.0, 0.0]
radius = 0.5
restitution = 0.3
```

### `box`

Axis-aligned bounding box. Particles are clamped inside.

```toml
[[constraints]]
type = "box"
min = [-1.0, 0.0, -1.0]
max = [2.0, 3.0, 1.0]
restitution = 0.3
```

### `pin`

Pin particles to their initial positions (first N particles).

```toml
[[constraints]]
type = "pin"
count = 50  # number of pinned particles
```

---

## `[spatial]`

Spatial acceleration structure for neighbor queries (used by SPH).

```toml
[spatial]
type = "hashgrid"
cell_size = 0.05
grid_dims = [40, 60, 20]  # optional, default: [32, 32, 32]
```

> Cell size should be ≥ 2× the smoothing radius for correct neighbor queries.

---

## `[springs]`

Spring topology for cloth/spring simulations.

```toml
[springs]
stiffness = 10000.0
damping = 10.0
topology = "grid"  # auto-generate from grid layout
grid_width = 50
grid_height = 50
```

---

## `[output]`

Output configuration (planned, not yet implemented).

```toml
[output]
format = "json"  # "json", "binary", "ply" (planned)
fps = 60
path = "output/"
```

---

## CLI

```bash
# Run simulation
forge run sim.toml

# Validate without running
forge check sim.toml

# Run with live 3D viewer (Three.js, browser-based)
forge run sim.toml --serve 8080

# Run with Phantom H.264 streaming (requires --features phantom)
forge run sim.toml --stream 8080
```

---

## Examples

See `examples/` directory:
- `particle-rain.toml` — 100K particles falling with ground bounce
- `cloth-on-sphere.toml` — 2500-particle cloth draped on sphere
- `dam-break.toml` — 50K SPH fluid particles (classic benchmark)
- `vortex.toml` — Custom vortex force field using expressions

---

## Auto-Optimizations

These happen transparently — your TOML stays clean:

| Optimization | Trigger | Effect |
|-------------|---------|--------|
| SPH kernel fusion | All 3 SPH forces present | 3 neighbor passes → 2 |
| Shared memory tiling | SPH modules | ~2x less global memory traffic |
| GPU hash grid | `[spatial]` config | Full GPU build pipeline |
| GPU prefix sum | Hash grid > 1 block | Blelloch scan, no CPU |
| Expression JIT | `type = "custom"` | Compile once, cache kernel |
