#!/usr/bin/env python3
"""
🔥 Forge Scene Generator — Describe a physics scene in natural language,
   Forge generates TOML + runs simulation + opens WebGL viewer.

Usage:
  python3 demos/ai_scene_gen.py "a ball dropping onto a trampoline"
  python3 demos/ai_scene_gen.py "rain falling on a flat surface"
  python3 demos/ai_scene_gen.py "two water columns crashing into each other"
"""

import sys
import os
import subprocess
import json

# Pre-built scene templates (no LLM needed — pattern matching)
TEMPLATES = {
    "rain": {
        "name": "particle-rain",
        "count": 100000,
        "init_min": [-3.0, 5.0, -3.0],
        "init_max": [3.0, 15.0, 3.0],
        "forces": [
            {"type": "gravity", "value": [0.0, -9.81, 0.0]},
        ],
        "constraints": [
            {"type": "box", "min": [-4.0, 0.0, -4.0], "max": [4.0, 20.0, 4.0], "restitution": 0.4},
        ],
        "dt": 0.001,
        "substeps": 4,
        "duration": 5.0,
    },
    "dam": {
        "name": "dam-break",
        "count": 50000,
        "init_min": [-1.0, 0.0, -0.5],
        "init_max": [0.0, 3.0, 0.5],
        "forces": [
            {"type": "sph_density", "smoothing_radius": 0.02},
            {"type": "sph_pressure", "gas_constant": 2000.0, "rest_density": 1000.0},
            {"type": "sph_viscosity", "coefficient": 0.008},
            {"type": "gravity", "value": [0.0, -9.81, 0.0]},
        ],
        "constraints": [
            {"type": "box", "min": [-1.5, 0.0, -0.5], "max": [2.0, 4.0, 0.5], "restitution": 0.3},
        ],
        "spatial": {"type": "hashgrid", "cell_size": 0.04, "grid_dims": [75, 100, 40]},
        "dt": 0.0001,
        "substeps": 10,
        "duration": 2.0,
    },
    "collision": {
        "name": "collision",
        "count": 100000,
        "init_blocks": [
            {"count": 50000, "min": [-2.0, 0.0, -0.5], "max": [-0.5, 3.0, 0.5]},
            {"count": 50000, "min": [0.5, 0.0, -0.5], "max": [2.0, 3.0, 0.5]},
        ],
        "forces": [
            {"type": "sph_density", "smoothing_radius": 0.02},
            {"type": "sph_pressure", "gas_constant": 2000.0, "rest_density": 1000.0},
            {"type": "sph_viscosity", "coefficient": 0.008},
            {"type": "gravity", "value": [0.0, -9.81, 0.0]},
        ],
        "constraints": [
            {"type": "box", "min": [-2.5, 0.0, -1.0], "max": [2.5, 5.0, 1.0], "restitution": 0.2},
        ],
        "spatial": {"type": "hashgrid", "cell_size": 0.04, "grid_dims": [125, 100, 40]},
        "dt": 0.0001,
        "substeps": 10,
        "duration": 3.0,
    },
    "fountain": {
        "name": "fountain",
        "count": 50000,
        "init_min": [-0.2, 0.0, -0.2],
        "init_max": [0.2, 0.5, 0.2],
        "vel_min": [-0.5, 8.0, -0.5],
        "vel_max": [0.5, 12.0, 0.5],
        "forces": [
            {"type": "gravity", "value": [0.0, -9.81, 0.0]},
        ],
        "constraints": [
            {"type": "box", "min": [-5.0, 0.0, -5.0], "max": [5.0, 20.0, 5.0], "restitution": 0.3},
        ],
        "dt": 0.001,
        "substeps": 4,
        "duration": 5.0,
    },
    "drop": {
        "name": "ball-drop",
        "count": 20000,
        "init_min": [-0.3, 3.0, -0.3],
        "init_max": [0.3, 4.0, 0.3],
        "forces": [
            {"type": "sph_density", "smoothing_radius": 0.03},
            {"type": "sph_pressure", "gas_constant": 1500.0, "rest_density": 1000.0},
            {"type": "sph_viscosity", "coefficient": 0.01},
            {"type": "gravity", "value": [0.0, -9.81, 0.0]},
        ],
        "constraints": [
            {"type": "box", "min": [-2.0, 0.0, -2.0], "max": [2.0, 5.0, 2.0], "restitution": 0.3},
        ],
        "spatial": {"type": "hashgrid", "cell_size": 0.06, "grid_dims": [67, 84, 67]},
        "dt": 0.0001,
        "substeps": 10,
        "duration": 2.0,
    },
}

KEYWORDS = {
    "rain": ["rain", "fall", "dropping", "shower"],
    "dam": ["dam", "water", "flood", "wave", "splash"],
    "collision": ["collid", "crash", "slam", "two", "dual", "versus"],
    "fountain": ["fountain", "jet", "spray", "geyser", "shoot"],
    "drop": ["drop", "ball", "sphere", "blob", "trampoline", "bounce"],
}


def match_scene(description: str) -> str:
    desc = description.lower()
    scores = {}
    for scene, keywords in KEYWORDS.items():
        scores[scene] = sum(1 for kw in keywords if kw in desc)
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "rain"  # default
    return best


def generate_toml(template: dict) -> str:
    lines = []
    lines.append(f'[simulation]')
    lines.append(f'name = "{template["name"]}"')
    lines.append(f'type = "particles"')
    lines.append(f'dt = {template["dt"]}')
    lines.append(f'substeps = {template["substeps"]}')
    lines.append(f'duration = {template["duration"]}')
    lines.append(f'count = {template["count"]}')
    lines.append('')

    if "init_blocks" in template:
        for i, block in enumerate(template["init_blocks"]):
            lines.append(f'[[fields]]')
            lines.append(f'name = "position"')
            lines.append(f'type = "vec3f"')
            lines.append(f'count = {block["count"]}')
            lines.append(f'init = {{ type = "random", min = {block["min"]}, max = {block["max"]} }}')
            lines.append('')
    else:
        lines.append(f'[[fields]]')
        lines.append(f'name = "position"')
        lines.append(f'type = "vec3f"')
        lines.append(f'count = {template["count"]}')
        lines.append(f'init = {{ type = "random", min = {template["init_min"]}, max = {template["init_max"]} }}')
        lines.append('')

    if "vel_min" in template:
        lines.append(f'[[fields]]')
        lines.append(f'name = "velocity"')
        lines.append(f'type = "vec3f"')
        lines.append(f'init = {{ type = "random", min = {template["vel_min"]}, max = {template["vel_max"]} }}')
    else:
        lines.append(f'[[fields]]')
        lines.append(f'name = "velocity"')
        lines.append(f'type = "vec3f"')
        lines.append('init = { type = "zero" }')
    lines.append('')

    if "spatial" in template:
        s = template["spatial"]
        lines.append(f'[spatial]')
        lines.append(f'type = "{s["type"]}"')
        lines.append(f'cell_size = {s["cell_size"]}')
        lines.append(f'grid_dims = {s["grid_dims"]}')
        lines.append('')

    for force in template["forces"]:
        lines.append(f'[[forces]]')
        for k, v in force.items():
            if isinstance(v, str):
                lines.append(f'{k} = "{v}"')
            elif isinstance(v, list):
                lines.append(f'{k} = {v}')
            else:
                lines.append(f'{k} = {v}')
        lines.append('')

    for c in template["constraints"]:
        lines.append(f'[[constraints]]')
        for k, v in c.items():
            if isinstance(v, str):
                lines.append(f'{k} = "{v}"')
            elif isinstance(v, list):
                lines.append(f'{k} = {v}')
            else:
                lines.append(f'{k} = {v}')
        lines.append('')

    return '\n'.join(lines)


def main():
    if len(sys.argv) < 2:
        print("🔥 Forge Scene Generator")
        print()
        print("Usage: python3 demos/ai_scene_gen.py \"<description>\"")
        print()
        print("Examples:")
        print('  "rain falling on the ground"')
        print('  "a dam break flooding a valley"')
        print('  "two water columns colliding"')
        print('  "a fountain shooting up"')
        print('  "a ball dropping from height"')
        return

    description = " ".join(sys.argv[1:])
    print(f"🔥 Forge Scene Generator")
    print(f"   Description: \"{description}\"")

    scene = match_scene(description)
    template = TEMPLATES[scene]
    print(f"   Matched scene: {scene} ({template['name']})")
    print(f"   Particles: {template['count']:,}")

    toml = generate_toml(template)
    path = f"/tmp/forge_scene_{scene}.toml"
    with open(path, "w") as f:
        f.write(toml)
    print(f"   Generated: {path}")

    print(f"\n   Running simulation with WebGL viewer...")
    print(f"   Open http://localhost:8080 in your browser\n")

    forge_bin = os.path.join(os.path.dirname(__file__), "..", "target", "release", "forge")
    os.execvp(forge_bin, [forge_bin, "run", path, "--serve", "8080"])


if __name__ == "__main__":
    main()
