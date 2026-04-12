//! # Particle Simulation Example
//!
//! This is the Forge equivalent of our Warp particle demo —
//! 100K particles falling under gravity with ground bounce.
//!
//! ## How It Would Work (M1 target)
//!
//! ```rust,ignore
//! use forge::prelude::*;
//!
//! #[kernel]
//! fn integrate(
//!     pos: &mut Array<Vec3f>,
//!     vel: &mut Array<Vec3f>,
//!     dt: f32,
//!     gravity: f32,
//!     ground_y: f32,
//!     restitution: f32,
//! ) {
//!     let tid = thread_id();
//!
//!     // Apply gravity
//!     vel[tid] = vel[tid] + Vec3f::new(0.0, gravity * dt, 0.0);
//!
//!     // Integrate position
//!     pos[tid] = pos[tid] + vel[tid] * dt;
//!
//!     // Ground collision
//!     if pos[tid].y < ground_y {
//!         pos[tid] = Vec3f::new(pos[tid].x, ground_y, pos[tid].z);
//!         vel[tid] = Vec3f::new(vel[tid].x, -vel[tid].y * restitution, vel[tid].z);
//!     }
//! }
//!
//! fn main() {
//!     let ctx = Forge::init();
//!     let n = 100_000;
//!
//!     let mut pos = Array::<Vec3f>::from_fn(n, |i| {
//!         Vec3f::new(
//!             ((i * 7) % 200) as f32 / 10.0 - 10.0,  // X: -10 to 10
//!             (i % 200) as f32 / 10.0 + 5.0,          // Y: 5 to 25
//!             ((i * 13) % 200) as f32 / 10.0 - 10.0,  // Z: -10 to 10
//!         )
//!     }, Device::Cuda(0));
//!
//!     let mut vel = Array::<Vec3f>::zeros(n, Device::Cuda(0));
//!
//!     let dt = 1.0 / 60.0;
//!     let gravity = -9.81;
//!     let steps = 300;
//!
//!     for _ in 0..steps {
//!         integrate.launch(n, &mut pos, &mut vel, dt, gravity, 0.0, 0.7);
//!     }
//!
//!     ctx.synchronize();
//!     println!("Simulated {} particles for {} steps", n, steps);
//! }
//! ```
//!
//! ## Current State
//!
//! This example demonstrates the CPU-side type system that's already implemented.

use forge_core::*;

fn main() {
    println!("🔥 Forge Particle Demo (CPU-only preview)");
    println!();

    let n = 1000; // Smaller count for CPU demo
    let dt = 1.0f32 / 60.0;
    let gravity = -9.81f32;
    let restitution = 0.7f32;
    let steps = 300;

    // Initialize positions and velocities using forge-core types
    let mut positions: Vec<Vec3f> = (0..n)
        .map(|i| {
            Vec3f::new(
                ((i * 7) % 200) as f32 / 10.0 - 10.0,
                (i % 200) as f32 / 10.0 + 5.0,
                ((i * 13) % 200) as f32 / 10.0 - 10.0,
            )
        })
        .collect();

    let mut velocities: Vec<Vec3f> = vec![Vec3f::zero(); n];

    // Simulate
    let start = std::time::Instant::now();

    for _ in 0..steps {
        for i in 0..n {
            // Apply gravity
            velocities[i] = velocities[i] + Vec3f::new(0.0, gravity * dt, 0.0);

            // Integrate position
            positions[i] = positions[i] + velocities[i] * dt;

            // Ground collision
            if positions[i].y < 0.0 {
                positions[i] = Vec3f::new(positions[i].x, 0.0, positions[i].z);
                velocities[i] = Vec3f::new(
                    velocities[i].x,
                    -velocities[i].y * restitution,
                    velocities[i].z,
                );
            }
        }
    }

    let elapsed = start.elapsed();

    println!("  Particles : {}", n);
    println!("  Steps     : {}", steps);
    println!("  Wall time : {:.4} s", elapsed.as_secs_f64());
    println!(
        "  Throughput: {:.0} particle-steps/s",
        (n * steps) as f64 / elapsed.as_secs_f64()
    );
    println!();

    // Show a few particles
    println!("Sample particles (position):");
    for i in 0..5 {
        let p = positions[i];
        println!("  #{}: ({:+.2}, {:+.2}, {:+.2})", i, p.x, p.y, p.z);
    }
}
