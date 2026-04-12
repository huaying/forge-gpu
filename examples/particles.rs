//! # Forge GPU Particle Simulation 🔥
//!
//! 100K particles under gravity with ground bounce — running on GPU.
//! This is the M1 finish line demo.
//!
//! Run: `CARGO_HOME=/home/horde/.cargo cargo run --example particles`

use forge_macros::kernel;
use forge_runtime::{cuda, Array, Device};

#[kernel]
fn integrate(
    pos_x: &mut Array<f32>,
    pos_y: &mut Array<f32>,
    pos_z: &mut Array<f32>,
    vel_x: &mut Array<f32>,
    vel_y: &mut Array<f32>,
    vel_z: &mut Array<f32>,
    dt: f32,
    gravity: f32,
    ground_y: f32,
    restitution: f32,
    n: i32,
) {
    let tid = thread_id();
    if tid < n {
        // Apply gravity
        vel_y[tid] = vel_y[tid] + gravity * dt;

        // Integrate position
        pos_x[tid] = pos_x[tid] + vel_x[tid] * dt;
        pos_y[tid] = pos_y[tid] + vel_y[tid] * dt;
        pos_z[tid] = pos_z[tid] + vel_z[tid] * dt;

        // Ground collision
        if pos_y[tid] < ground_y {
            pos_y[tid] = ground_y;
            vel_y[tid] = vel_y[tid] * restitution * -1.0;
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
    println!("⚡ GPU Mode (CUDA)");
    cuda::init();
    let gpu_count = cuda::device_count();
    if gpu_count == 0 {
        println!("  No GPU found — skipping GPU benchmark.\n");
    } else {
        println!("  Device count: {}", gpu_count);

        // Init positions: spread in XZ plane, Y = 5..25
        let px: Vec<f32> = (0..n).map(|i| ((i * 7) % 200) as f32 / 10.0 - 10.0).collect();
        let py: Vec<f32> = (0..n).map(|i| (i % 200) as f32 / 10.0 + 5.0).collect();
        let pz: Vec<f32> = (0..n).map(|i| ((i * 13) % 200) as f32 / 10.0 - 10.0).collect();

        let mut pos_x = Array::from_vec(px, Device::Cuda(0));
        let mut pos_y = Array::from_vec(py, Device::Cuda(0));
        let mut pos_z = Array::from_vec(pz, Device::Cuda(0));
        let mut vel_x = Array::<f32>::zeros(n, Device::Cuda(0));
        let mut vel_y = Array::<f32>::zeros(n, Device::Cuda(0));
        let mut vel_z = Array::<f32>::zeros(n, Device::Cuda(0));

        let start = std::time::Instant::now();

        for _ in 0..steps {
            integrate::launch_async(
                &mut pos_x,
                &mut pos_y,
                &mut pos_z,
                &mut vel_x,
                &mut vel_y,
                &mut vel_z,
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
        let final_y = pos_y.to_vec();
        let below_ground = final_y.iter().filter(|&&y| y < ground_y - 0.001).count();
        println!(
            "  Verify    : {} particles below ground (should be 0)",
            below_ground
        );

        // Show sample
        let fy = pos_y.to_vec();
        let fx = pos_x.to_vec();
        let fz = pos_z.to_vec();
        println!("\n  Sample particles:");
        for i in 0..5 {
            println!(
                "    #{}: ({:+.2}, {:+.2}, {:+.2})",
                i, fx[i], fy[i], fz[i]
            );
        }
        println!();
    }

    // ----- CPU simulation for comparison -----
    println!("🐢 CPU Mode (single-threaded)");

    let mut cpu_px: Vec<f32> = (0..n).map(|i| ((i * 7) % 200) as f32 / 10.0 - 10.0).collect();
    let mut cpu_py: Vec<f32> = (0..n).map(|i| (i % 200) as f32 / 10.0 + 5.0).collect();
    let mut cpu_pz: Vec<f32> = (0..n).map(|i| ((i * 13) % 200) as f32 / 10.0 - 10.0).collect();
    let cpu_vx: Vec<f32> = vec![0.0; n];
    let mut cpu_vy: Vec<f32> = vec![0.0; n];
    let cpu_vz: Vec<f32> = vec![0.0; n];

    let start = std::time::Instant::now();

    for _ in 0..steps {
        for i in 0..n {
            cpu_vy[i] += gravity * dt;
            cpu_px[i] += cpu_vx[i] * dt;
            cpu_py[i] += cpu_vy[i] * dt;
            cpu_pz[i] += cpu_vz[i] * dt;
            if cpu_py[i] < ground_y {
                cpu_py[i] = ground_y;
                cpu_vy[i] = -cpu_vy[i] * restitution;
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
        println!(
            "    #{}: ({:+.2}, {:+.2}, {:+.2})",
            i, cpu_px[i], cpu_py[i], cpu_pz[i]
        );
    }

    // ----- Comparison -----
    if gpu_count > 0 {
        println!("\n📊 Comparison");
        let gpu_elapsed = {
            // Re-run GPU to get timing (use the already-compiled kernel)
            let px: Vec<f32> = (0..n).map(|i| ((i * 7) % 200) as f32 / 10.0 - 10.0).collect();
            let py: Vec<f32> = (0..n).map(|i| (i % 200) as f32 / 10.0 + 5.0).collect();
            let pz: Vec<f32> = (0..n).map(|i| ((i * 13) % 200) as f32 / 10.0 - 10.0).collect();

            let mut pos_x = Array::from_vec(px, Device::Cuda(0));
            let mut pos_y = Array::from_vec(py, Device::Cuda(0));
            let mut pos_z = Array::from_vec(pz, Device::Cuda(0));
            let mut vel_x = Array::<f32>::zeros(n, Device::Cuda(0));
            let mut vel_y = Array::<f32>::zeros(n, Device::Cuda(0));
            let mut vel_z = Array::<f32>::zeros(n, Device::Cuda(0));

            let start = std::time::Instant::now();
            for _ in 0..steps {
                integrate::launch_async(
                    &mut pos_x,
                    &mut pos_y,
                    &mut pos_z,
                    &mut vel_x,
                    &mut vel_y,
                    &mut vel_z,
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
            start.elapsed()
        };

        let speedup = cpu_elapsed.as_secs_f64() / gpu_elapsed.as_secs_f64();
        println!(
            "  GPU is {:.0}x faster than single-threaded CPU",
            speedup
        );
    }

    println!("\n🏁 Done.");
}
