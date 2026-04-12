//! # Forge GPU Particle Simulation 🔥
//!
//! 100K particles under gravity with ground bounce — running on GPU.
//! Uses `Array<Vec3f>` for clean, natural particle code.
//!
//! Run: `CARGO_HOME=/home/horde/.cargo cargo run --example particles --release`

use forge_macros::kernel;
use forge_core::Vec3f;
use forge_runtime::{cuda, Array, Device};

#[kernel]
fn integrate(
    pos: &mut Array<Vec3f>,
    vel: &mut Array<Vec3f>,
    dt: f32,
    gravity: f32,
    ground_y: f32,
    restitution: f32,
    n: i32,
) {
    let tid = thread_id();
    if tid < n {
        // Apply gravity
        vel[tid] = vel[tid] + Vec3f::new(0.0, gravity * dt, 0.0);

        // Integrate position
        pos[tid] = pos[tid] + vel[tid] * dt;

        // Ground collision (check y component)
        if pos[tid].y < ground_y {
            // Reset y to ground, flip y velocity
            let px = pos[tid].x;
            let pz = pos[tid].z;
            pos[tid] = Vec3f::new(px, ground_y, pz);
            let vx = vel[tid].x;
            let vy = vel[tid].y;
            let vz = vel[tid].z;
            vel[tid] = Vec3f::new(vx, vy * restitution * -1.0, vz);
        }
    }
}

fn main() {
    println!("🔥 Forge Particle Simulation");
    println!("=============================\n");

    let n = 100_000usize;
    let dt = 1.0f32 / 60.0;
    let gravity = -9.81f32;
    let ground_y = 0.0f32;
    let restitution = 0.7f32;
    let steps = 300;

    // ----- GPU simulation -----
    println!("⚡ GPU Mode (CUDA) — using Array<Vec3f>");
    cuda::init();
    let gpu_count = cuda::device_count();
    if gpu_count == 0 {
        println!("  No GPU found — skipping GPU benchmark.\n");
    } else {
        println!("  Device count: {}", gpu_count);

        // Init positions: spread in XZ plane, Y = 5..25
        let pos_data: Vec<Vec3f> = (0..n)
            .map(|i| Vec3f::new(
                ((i * 7) % 200) as f32 / 10.0 - 10.0,
                (i % 200) as f32 / 10.0 + 5.0,
                ((i * 13) % 200) as f32 / 10.0 - 10.0,
            ))
            .collect();
        let vel_data = vec![Vec3f::new(0.0, 0.0, 0.0); n];

        let mut pos = Array::from_vec(pos_data, Device::Cuda(0));
        let mut vel = Array::from_vec(vel_data, Device::Cuda(0));

        let start = std::time::Instant::now();

        for _ in 0..steps {
            integrate::launch_async(
                &mut pos,
                &mut vel,
                dt,
                gravity,
                ground_y,
                restitution,
                n as i32,
                n,
                0,
            )
            .expect("kernel launch failed");
        }
        cuda::synchronize(0);

        let gpu_elapsed = start.elapsed();
        let gpu_throughput = (n * steps) as f64 / gpu_elapsed.as_secs_f64();

        println!("  Particles : {}", n);
        println!("  Steps     : {}", steps);
        println!("  Wall time : {:.4} s", gpu_elapsed.as_secs_f64());
        println!("  Throughput: {:.2e} particle-steps/s", gpu_throughput);

        // Verify
        let final_pos = pos.to_vec();
        let below_ground = final_pos.iter().filter(|p| p.y < ground_y - 0.001).count();
        println!(
            "  Verify    : {} particles below ground (should be 0)",
            below_ground
        );

        println!("\n  Sample particles:");
        for i in 0..5 {
            let p = final_pos[i];
            println!(
                "    #{}: ({:+.2}, {:+.2}, {:+.2})",
                i, p.x, p.y, p.z
            );
        }
        println!();
    }

    // ----- CPU simulation for comparison -----
    println!("🐢 CPU Mode (single-threaded)");

    let mut cpu_pos: Vec<Vec3f> = (0..n)
        .map(|i| Vec3f::new(
            ((i * 7) % 200) as f32 / 10.0 - 10.0,
            (i % 200) as f32 / 10.0 + 5.0,
            ((i * 13) % 200) as f32 / 10.0 - 10.0,
        ))
        .collect();
    let mut cpu_vel = vec![Vec3f::new(0.0, 0.0, 0.0); n];

    let start = std::time::Instant::now();

    for _ in 0..steps {
        for i in 0..n {
            cpu_vel[i] = cpu_vel[i] + Vec3f::new(0.0, gravity * dt, 0.0);
            cpu_pos[i] = cpu_pos[i] + cpu_vel[i] * dt;
            if cpu_pos[i].y < ground_y {
                cpu_pos[i] = Vec3f::new(cpu_pos[i].x, ground_y, cpu_pos[i].z);
                cpu_vel[i] = Vec3f::new(cpu_vel[i].x, -cpu_vel[i].y * restitution, cpu_vel[i].z);
            }
        }
    }

    let cpu_elapsed = start.elapsed();
    let cpu_throughput = (n * steps) as f64 / cpu_elapsed.as_secs_f64();

    println!("  Particles : {}", n);
    println!("  Steps     : {}", steps);
    println!("  Wall time : {:.4} s", cpu_elapsed.as_secs_f64());
    println!("  Throughput: {:.2e} particle-steps/s", cpu_throughput);

    println!("\n  Sample particles:");
    for i in 0..5 {
        let p = cpu_pos[i];
        println!(
            "    #{}: ({:+.2}, {:+.2}, {:+.2})",
            i, p.x, p.y, p.z
        );
    }

    // ----- Comparison -----
    if gpu_count > 0 {
        println!("\n📊 Comparison");

        // Re-run GPU for warm comparison
        let pos_data: Vec<Vec3f> = (0..n)
            .map(|i| Vec3f::new(
                ((i * 7) % 200) as f32 / 10.0 - 10.0,
                (i % 200) as f32 / 10.0 + 5.0,
                ((i * 13) % 200) as f32 / 10.0 - 10.0,
            ))
            .collect();
        let vel_data = vec![Vec3f::new(0.0, 0.0, 0.0); n];
        let mut pos = Array::from_vec(pos_data, Device::Cuda(0));
        let mut vel = Array::from_vec(vel_data, Device::Cuda(0));

        let start = std::time::Instant::now();
        for _ in 0..steps {
            integrate::launch_async(
                &mut pos, &mut vel,
                dt, gravity, ground_y, restitution, n as i32, n, 0,
            ).expect("kernel launch failed");
        }
        cuda::synchronize(0);
        let gpu_elapsed = start.elapsed();

        let speedup = cpu_elapsed.as_secs_f64() / gpu_elapsed.as_secs_f64();
        println!("  GPU is {:.0}x faster than single-threaded CPU", speedup);
    }

    println!("\n🏁 Done.");
}
