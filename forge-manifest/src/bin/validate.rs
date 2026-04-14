/// Validate all demo TOML files by running them and checking physics sanity.
use forge_manifest::{SimManifest, ForceDef, InitDef, ConstraintDef};
use forge_manifest::run_manifest;
use std::fs;

fn analyze(name: &str, positions: &[f32], count: usize, manifest: &SimManifest) {
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_z = f32::MAX;
    let mut max_z = f32::MIN;
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;
    let mut below_ground = 0usize;
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_z = 0.0f64;
    let mut valid_count = 0usize;

    for i in 0..count {
        let x = positions[i * 3];
        let y = positions[i * 3 + 1];
        let z = positions[i * 3 + 2];

        if x.is_nan() || y.is_nan() || z.is_nan() { nan_count += 1; continue; }
        if x.is_infinite() || y.is_infinite() || z.is_infinite() { inf_count += 1; continue; }

        valid_count += 1;
        min_x = min_x.min(x); max_x = max_x.max(x);
        min_y = min_y.min(y); max_y = max_y.max(y);
        min_z = min_z.min(z); max_z = max_z.max(z);
        sum_x += x as f64;
        sum_y += y as f64;
        sum_z += z as f64;

        if y < -0.1 { below_ground += 1; }
    }

    let avg_x = if valid_count > 0 { sum_x / valid_count as f64 } else { 0.0 };
    let avg_y = if valid_count > 0 { sum_y / valid_count as f64 } else { 0.0 };
    let avg_z = if valid_count > 0 { sum_z / valid_count as f64 } else { 0.0 };

    println!("  NaN: {}  Inf: {}  Valid: {}", nan_count, inf_count, valid_count);
    println!("  X: [{:.4}, {:.4}]  avg {:.4}", min_x, max_x, avg_x);
    println!("  Y: [{:.4}, {:.4}]  avg {:.4}", min_y, max_y, avg_y);
    println!("  Z: [{:.4}, {:.4}]  avg {:.4}", min_z, max_z, avg_z);
    println!("  Below ground (y < -0.1): {}", below_ground);

    // Compute Y variance (for checking if particles are all stuck at same height)
    let mut var_y = 0.0f64;
    for i in 0..count {
        let y = positions[i * 3 + 1];
        if !y.is_nan() && !y.is_infinite() {
            let dy = y as f64 - avg_y;
            var_y += dy * dy;
        }
    }
    let std_y = if valid_count > 1 { (var_y / valid_count as f64).sqrt() } else { 0.0 };
    println!("  Y stdev: {:.4}", std_y);

    // Print a sample of 10 particles
    println!("  Sample particles (first 10):");
    for i in 0..10.min(count) {
        let x = positions[i * 3];
        let y = positions[i * 3 + 1];
        let z = positions[i * 3 + 2];
        println!("    [{}] ({:.4}, {:.4}, {:.4})", i, x, y, z);
    }

    // Physics checks
    let mut ok = true;

    if nan_count > 0 {
        println!("  ❌ NaN particles — simulation EXPLODED");
        ok = false;
    }
    if inf_count > 0 {
        println!("  ❌ Inf particles — simulation EXPLODED");
        ok = false;
    }

    let spread = (max_x - min_x).max(max_y - min_y).max(max_z - min_z);
    if spread > 200.0 {
        println!("  ❌ Massive spread {:.1} — particles scattered unrealistically", spread);
        ok = false;
    }

    // Check if gravity pulled particles down
    let has_gravity = manifest.forces.iter().any(|f| matches!(f, ForceDef::Gravity { .. }));
    if has_gravity {
        let (init_min_y, init_max_y) = manifest.fields.iter().find_map(|f| {
            if f.name == "position" || f.name == "pos" {
                match &f.init {
                    Some(InitDef::Random { min, max }) => {
                        Some((min.get(1).copied().unwrap_or(0.0), max.get(1).copied().unwrap_or(0.0)))
                    }
                    Some(InitDef::Grid { origin, .. }) => {
                        let oy = origin.get(1).copied().unwrap_or(0.0);
                        Some((oy, oy + 2.0))
                    }
                    _ => None,
                }
            } else { None }
        }).unwrap_or((0.0, 0.0));

        let init_avg = (init_min_y + init_max_y) / 2.0;
        println!("  Initial Y: [{:.2}, {:.2}] avg ~{:.2}", init_min_y, init_max_y, init_avg);

        if avg_y > init_avg as f64 + 2.0 {
            println!("  ❌ Particles ROSE (avg Y {:.2} > initial {:.2}) — gravity broken or explosion", avg_y, init_avg);
            ok = false;
        }
    }

    // Check ground penetration
    let has_ground = manifest.constraints.iter().any(|c| matches!(c, ConstraintDef::GroundPlane { .. }));
    let has_box = manifest.constraints.iter().any(|c| matches!(c, ConstraintDef::Box { .. }));
    if (has_ground || has_box) && below_ground > count / 5 {
        println!("  ⚠️  {}% below ground despite constraint", below_ground * 100 / count);
    }

    // Check for particle collapse
    if spread < 0.001 && count > 100 {
        println!("  ❌ All particles collapsed to same point");
        ok = false;
    }

    // Check all-zero (nothing happened)
    if max_x.abs() < 1e-6 && max_y.abs() < 1e-6 && max_z.abs() < 1e-6 && count > 100 {
        println!("  ❌ All positions at origin — simulation didn't run");
        ok = false;
    }

    // SPH-specific: check density makes sense
    let has_sph = manifest.forces.iter().any(|f| matches!(f, ForceDef::SphDensity { .. }));
    if has_sph {
        // Water should pool at bottom, not all at same height
        if std_y < 0.01 && count > 1000 {
            println!("  ⚠️  SPH water has no Y variation — all particles at same height (std={:.4})", std_y);
        }
        // Water should spread in X/Z due to pressure
        let x_spread = max_x - min_x;
        let z_spread = max_z - min_z;
        println!("  SPH spread: X={:.3}, Z={:.3}", x_spread, z_spread);
    }

    if ok { println!("  ✅ Sanity OK"); }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let demos = if args.is_empty() {
        vec![
            "examples/particle-rain.toml".to_string(),
            "examples/drop-bounce.toml".to_string(),
            "examples/vortex.toml".to_string(),
            "examples/cloth-on-sphere.toml".to_string(),
            "examples/dam-break.toml".to_string(),
            "examples/waterfall.toml".to_string(),
        ]
    } else {
        args
    };

    for path in demos.iter().map(|s| s.as_str()) {
        let contents = match fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => { println!("\n❌ Can't read {}: {}", path, e); continue; }
        };

        let manifest: SimManifest = match toml::from_str(&contents) {
            Ok(m) => m,
            Err(e) => { println!("\n❌ Can't parse {}: {}", path, e); continue; }
        };

        println!("\n========================================");
        println!("Running: {} ({})", path, manifest.simulation.name);
        println!("  dt={} substeps={} duration={}s count={}",
            manifest.simulation.dt, manifest.simulation.substeps,
            manifest.simulation.duration, manifest.particle_count());
        println!("  forces: {:?}", manifest.forces.iter().map(|f| match f {
            ForceDef::Gravity { .. } => "gravity",
            ForceDef::Drag { .. } => "drag",
            ForceDef::SphDensity { .. } => "sph_density",
            ForceDef::SphPressure { .. } => "sph_pressure",
            ForceDef::SphViscosity { .. } => "sph_viscosity",
            ForceDef::Custom { .. } => "custom",
            _ => "other",
        }).collect::<Vec<_>>());

        match run_manifest(&manifest) {
            Ok(result) => {
                println!("  Completed: {} steps in {:.2}s ({:.0} steps/s)",
                    result.steps, result.elapsed_secs,
                    result.steps as f64 / result.elapsed_secs);
                analyze(&manifest.simulation.name, &result.positions, result.count, &manifest);
            }
            Err(e) => {
                println!("  ❌ FAILED: {}", e);
            }
        }
    }

    println!("\n========================================");
    println!("Validation complete.");
}
