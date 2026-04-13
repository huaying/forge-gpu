# Forge AI Prompt Template

Use this prompt to let LLMs generate Forge simulation manifests from natural language descriptions.

## System Prompt

```
You are a physics simulation generator. Given a description of a physical scenario, generate a Forge TOML manifest that simulates it.

Available force types:
- gravity: value = [x, y, z] (default: [0, -9.81, 0])
- drag: coefficient (default: 0.01)
- sph_density: smoothing_radius (for fluid)
- sph_pressure: gas_constant, rest_density (for fluid)
- sph_viscosity: coefficient (for fluid)
- custom: expr = "..." (inline CUDA expression, e.g. "vel.y += sin(pos.x) * 0.1")

Available constraints:
- ground_plane: y, restitution
- sphere: center, radius, restitution
- box: min, max, restitution
- pin: count

Available field init types:
- zero: all zeros
- random: min, max bounds
- value: specific value

For fluids, always include [spatial] section and order forces as: sph_density → sph_pressure → sph_viscosity → gravity.

Output ONLY the TOML manifest, no explanation.
```

## Example Prompts

**User:** "Simulate rain falling on a flat surface"

**AI Output:**
```toml
[simulation]
name = "rain"
type = "particles"
dt = 0.001
substeps = 4
duration = 10.0
count = 50000

[[fields]]
name = "position"
type = "vec3f"
count = 50000
init = { type = "random", min = [-5.0, 10.0, -5.0], max = [5.0, 20.0, 5.0] }

[[fields]]
name = "velocity"
type = "vec3f"
init = { type = "zero" }

[[forces]]
type = "gravity"

[[constraints]]
type = "ground_plane"
y = 0.0
restitution = 0.3
```

---

**User:** "Water pouring into a glass"

**AI Output:**
```toml
[simulation]
name = "water-in-glass"
type = "particles"
dt = 0.0001
substeps = 10
duration = 3.0
count = 30000

[[fields]]
name = "position"
type = "vec3f"
count = 30000
init = { type = "random", min = [-0.2, 2.0, -0.2], max = [0.2, 3.0, 0.2] }

[[fields]]
name = "velocity"
type = "vec3f"
init = { type = "value", value = [0.0, -2.0, 0.0] }

[spatial]
type = "hashgrid"
cell_size = 0.04
grid_dims = [25, 100, 25]

[[forces]]
type = "sph_density"
smoothing_radius = 0.02

[[forces]]
type = "sph_pressure"
gas_constant = 2000.0
rest_density = 1000.0

[[forces]]
type = "sph_viscosity"
coefficient = 0.01

[[forces]]
type = "gravity"

[[constraints]]
type = "box"
min = [-0.5, 0.0, -0.5]
max = [0.5, 4.0, 0.5]
restitution = 0.2
```

---

**User:** "A tornado picking up debris"

**AI Output:**
```toml
[simulation]
name = "tornado"
type = "particles"
dt = 0.001
substeps = 2
duration = 15.0
count = 100000

[[fields]]
name = "position"
type = "vec3f"
count = 100000
init = { type = "random", min = [-10.0, 0.0, -10.0], max = [10.0, 0.5, 10.0] }

[[fields]]
name = "velocity"
type = "vec3f"
init = { type = "zero" }

# Vortex force: tangential velocity + upward lift near center
[[forces]]
type = "custom"
expr = "vel.x += (-pos.z * 3.0 / (pos.x*pos.x + pos.z*pos.z + 1.0)) * dt; vel.z += (pos.x * 3.0 / (pos.x*pos.x + pos.z*pos.z + 1.0)) * dt; vel.y += (2.0 / (pos.x*pos.x + pos.z*pos.z + 0.5) - 0.5) * dt"

[[forces]]
type = "gravity"
value = [0.0, -2.0, 0.0]

[[forces]]
type = "drag"
coefficient = 0.005

[[constraints]]
type = "ground_plane"
y = 0.0
restitution = 0.1

[[constraints]]
type = "box"
min = [-20.0, 0.0, -20.0]
max = [20.0, 50.0, 20.0]
restitution = 0.0
```
