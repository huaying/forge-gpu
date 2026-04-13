//! # forge — CLI for running simulation manifests.
//!
//! Usage:
//!   forge run sim.toml              — run simulation
//!   forge run sim.toml --serve 8080 — run with live 3D viewer (Three.js)
//!   forge run sim.toml --stream 8080 — run with Phantom H.264 streaming (requires phantom feature)
//!   forge check sim.toml            — validate without running

use forge_manifest::{SimManifest, run_manifest, run_manifest_streaming, validate};
use forge_manifest::serve::{self, FrameBuffer};
use std::env;
use std::sync::{Arc, Mutex};

fn main() {
    let args: Vec<String> = env::args().collect();

    // Handle no-argument commands first
    if args.len() >= 2 {
        match args[1].as_str() {
            "info" => {
                print_info();
                return;
            }
            "version" | "--version" | "-v" => {
                println!("🔥 Forge GPU v{}", env!("CARGO_PKG_VERSION"));
                return;
            }
            _ => {}
        }
    }

    if args.len() < 3 {
        eprintln!("🔥 Forge GPU Compute Framework v{}", env!("CARGO_PKG_VERSION"));
        eprintln!();
        eprintln!("Usage:");
        eprintln!("  forge run <manifest.toml>              — compile and run simulation");
        eprintln!("  forge run <manifest.toml> --serve 8080 — run with live 3D viewer");
        eprintln!("  forge check <manifest.toml>            — validate without running");
        eprintln!("  forge info                             — show GPU device info");
        eprintln!("  forge version                          — show version");
        std::process::exit(1);
    }

    let command = &args[1];
    let path = &args[2];

    // Parse --serve flag
    let serve_port: Option<u16> = args.iter().position(|a| a == "--serve").and_then(|i| {
        args.get(i + 1).and_then(|p| p.parse().ok()).or(Some(8080))
    });

    // Parse --stream flag (Phantom H.264 streaming)
    let stream_port: Option<u16> = args.iter().position(|a| a == "--stream").and_then(|i| {
        args.get(i + 1).and_then(|p| p.parse().ok()).or(Some(8080))
    });

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

            if let Some(port) = stream_port {
                // ── Phantom streaming mode ──
                #[cfg(feature = "phantom")]
                {
                    use forge_manifest::phantom_bridge::phantom_bridge::ForgeStreamSource;
                    use phantom_stream::StreamServer;

                    println!("  Mode: Phantom H.264 streaming\n");

                    let source = ForgeStreamSource::new(&manifest, 1280, 720)
                        .expect("failed to create stream source");

                    let server = StreamServer::new(source, port)
                        .expect("failed to create stream server");

                    println!("  Open http://localhost:{} in your browser\n", port);
                    server.run().expect("stream server error");
                }
                #[cfg(not(feature = "phantom"))]
                {
                    eprintln!("❌ --stream requires the 'phantom' feature.");
                    eprintln!("   Build with: cargo build --features phantom");
                    std::process::exit(1);
                }
            } else if let Some(port) = serve_port {
                // ── Streaming mode ──
                println!();
                let frame_buf = Arc::new(Mutex::new(FrameBuffer::new()));

                // Start unified HTTP + WebSocket server (single port)
                serve::start_server(port, Arc::clone(&frame_buf));

                println!();
                println!("  Open http://localhost:{} in your browser", port);
                println!("  Press Ctrl+C to stop");
                println!();

                // Give servers a moment to start
                std::thread::sleep(std::time::Duration::from_millis(100));

                // Determine frame interval: target ~30 fps in viewer
                // Each "step" is substeps × dt seconds of sim time
                let sim_step_time = manifest.simulation.dt * manifest.simulation.substeps as f64;
                let frame_interval = (1.0 / 30.0 / sim_step_time).max(1.0) as usize;

                let fb = Arc::clone(&frame_buf);
                match run_manifest_streaming(&manifest, frame_interval, |frame_num, positions| {
                    let mut buf = fb.lock().unwrap();
                    buf.set_frame(frame_num, positions);
                }) {
                    Ok(result) => {
                        println!("✅ Simulation complete!");
                        println!("  Steps: {}", result.steps);
                        println!("  Time: {:.3}s", result.elapsed_secs);
                        println!("  Throughput: {:.2e} particle-steps/s",
                            (result.count * result.steps) as f64 / result.elapsed_secs);

                        // Keep server alive for viewing the final state
                        println!("\n  Server still running. Press Ctrl+C to exit.");
                        loop {
                            std::thread::sleep(std::time::Duration::from_secs(1));
                        }
                    }
                    Err(e) => {
                        eprintln!("❌ Simulation failed: {}", e);
                        std::process::exit(1);
                    }
                }
            } else {
                // ── Normal mode ──
                println!();

                match run_manifest(&manifest) {
                    Ok(result) => {
                        println!("✅ Simulation complete!");
                        println!("  Steps: {}", result.steps);
                        println!("  Time: {:.3}s", result.elapsed_secs);
                        println!("  Throughput: {:.2e} particle-steps/s",
                            (result.count * result.steps) as f64 / result.elapsed_secs);

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
        }

        other => {
            eprintln!("Unknown command: '{}'. Use 'run', 'check', or 'info'.", other);
            std::process::exit(1);
        }
    }
}

fn print_info() {
    println!("🔥 Forge GPU Compute Framework v{}", env!("CARGO_PKG_VERSION"));
    println!();

    forge_runtime::cuda::init();
    let count = forge_runtime::cuda::device_count();

    if count == 0 {
        println!("  ⚠️  No CUDA devices found");
        println!("  CPU fallback: planned (not yet available)");
        return;
    }

    println!("  CUDA Devices: {}", count);
    println!();

    for i in 0..count {
        let name = forge_runtime::cuda::device_name(i);
        let (major, minor) = forge_runtime::cuda::compute_capability(i);
        let (free, total) = forge_runtime::cuda::mem_info(i);

        println!("  GPU {}: {}", i, name);
        println!("    Compute Capability: {}.{}", major, minor);
        println!("    Memory: {:.1} GB free / {:.1} GB total",
            free as f64 / 1e9, total as f64 / 1e9);
    }

    println!();
    println!("  Features:");
    println!("    ✅ Kernel JIT (nvrtc)");
    println!("    ✅ CUDA Graphs (auto capture)");
    println!("    ✅ Autodiff (reverse mode)");
    println!("    ✅ Multi-dim arrays (1D-4D)");
    println!("    ✅ Tile primitives (shared mem)");
    println!("    ✅ Spatial queries (HashGrid, BVH, Mesh)");
    println!("    ✅ PyTorch interop (__cuda_array_interface__)");
    println!("    ✅ Declarative TOML manifests");
    println!("    ✅ WebGL live viewer");
}
