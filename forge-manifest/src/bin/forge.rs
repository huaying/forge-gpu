//! # forge — CLI for running simulation manifests.
//!
//! Usage:
//!   forge run sim.toml
//!   forge check sim.toml

use forge_manifest::{SimManifest, run_manifest, validate};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage:");
        eprintln!("  forge run <manifest.toml>   — compile and run simulation");
        eprintln!("  forge check <manifest.toml> — validate without running");
        std::process::exit(1);
    }

    let command = &args[1];
    let path = &args[2];

    let manifest = match SimManifest::from_file(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("❌ {}", e);
            std::process::exit(1);
        }
    };

    match command.as_str() {
        "check" => {
            match validate(&manifest) {
                Ok(()) => {
                    println!("✅ Manifest '{}' is valid.", manifest.simulation.name);
                    println!("  Type: {}", manifest.simulation.r#type);
                    println!("  Particles: {}", manifest.particle_count());
                    println!("  Duration: {}s, dt: {}, substeps: {}",
                        manifest.simulation.duration,
                        manifest.simulation.dt,
                        manifest.simulation.substeps);
                    println!("  Forces: {}", manifest.forces.len());
                    println!("  Constraints: {}", manifest.constraints.len());
                }
                Err(errs) => {
                    eprintln!("❌ Validation errors:");
                    for e in &errs {
                        eprintln!("  - {}", e);
                    }
                    std::process::exit(1);
                }
            }
        }

        "run" => {
            println!("🔥 Forge Simulation Runner");
            println!("  Manifest: {}", path);
            println!("  Name: {}", manifest.simulation.name);
            println!("  Type: {}", manifest.simulation.r#type);
            println!("  Particles: {}", manifest.particle_count());
            println!("  Duration: {}s (dt={}, substeps={})",
                manifest.simulation.duration,
                manifest.simulation.dt,
                manifest.simulation.substeps);
            println!();

            match run_manifest(&manifest) {
                Ok(result) => {
                    println!("✅ Simulation complete!");
                    println!("  Steps: {}", result.steps);
                    println!("  Time: {:.3}s", result.elapsed_secs);
                    println!("  Throughput: {:.2e} particle-steps/s",
                        (result.count * result.steps) as f64 / result.elapsed_secs);

                    // Print sample positions
                    let sample = 5.min(result.count);
                    println!("\n  Sample final positions:");
                    for i in 0..sample {
                        let x = result.positions[i * 3];
                        let y = result.positions[i * 3 + 1];
                        let z = result.positions[i * 3 + 2];
                        println!("    #{}: ({:.3}, {:.3}, {:.3})", i, x, y, z);
                    }

                    if !result.frames.is_empty() {
                        println!("\n  Captured {} frames", result.frames.len());
                    }
                }
                Err(e) => {
                    eprintln!("❌ Simulation failed: {}", e);
                    std::process::exit(1);
                }
            }
        }

        other => {
            eprintln!("Unknown command: '{}'. Use 'run' or 'check'.", other);
            std::process::exit(1);
        }
    }
}
