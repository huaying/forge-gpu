//! # Forge Differentiable Spring Simulation 🧠🔥
//!
//! A complete differentiable physics demo using autodiff.
//! Spring system with gradient descent optimization.
//!
//! Run: `CARGO_HOME=/home/horde/.cargo cargo run --example spring_sim --release`

use forge_macros::kernel;
use forge_core::Vec3f;
use forge_runtime::{cuda, Array, Device};

/// Compute spring energy between consecutive particle pairs.
/// Particles 0-1, 2-3, 4-5, ... form springs.
/// energy[i] = 0.5 * k * (|pos[2i] - pos[2i+1]| - rest)^2
///
/// Decomposed into scalar ops for autodiff compatibility.
#[kernel(autodiff)]
fn pair_spring_energy(
    px: &Array<f32>,
    py: &Array<f32>,
    pz: &Array<f32>,
    rest_len: &Array<f32>,
    stiffness: f32,
    energy: &mut Array<f32>,
    n_springs: i32,
) {
    let tid = thread_id();
    if tid < n_springs {
        // Indices: spring tid connects particle 2*tid and 2*tid+1
        let i0 = tid * 2;
        let i1 = tid * 2 + 1;

        let dx = px[i0] - px[i1];
        let dy = py[i0] - py[i1];
        let dz = pz[i0] - pz[i1];

        let dist_sq = dx * dx + dy * dy + dz * dz;
        let dist = sqrt(dist_sq);

        let r = rest_len[tid];
        let stretch = dist - r;
        energy[tid] = stiffness * stretch * stretch * 0.5;
    }
}

/// Gradient descent step on flattened position arrays.
#[kernel]
fn grad_step(
    px: &mut Array<f32>,
    py: &mut Array<f32>,
    pz: &mut Array<f32>,
    gx: &Array<f32>,
    gy: &Array<f32>,
    gz: &Array<f32>,
    lr: f32,
    n: i32,
) {
    let tid = thread_id();
    if tid < n {
        px[tid] = px[tid] - lr * gx[tid];
        py[tid] = py[tid] - lr * gy[tid];
        pz[tid] = pz[tid] - lr * gz[tid];
    }
}

fn main() {
    println!("🧠🔥 Forge Differentiable Spring Simulation");
    println!("=============================================\n");

    cuda::init();
    if cuda::device_count() == 0 {
        println!("No GPU found.");
        return;
    }

    // ── Setup: 4 springs, 8 particles (pairs) ──
    // Spring 0: particle 0 ←→ particle 1
    // Spring 1: particle 2 ←→ particle 3
    // Spring 2: particle 4 ←→ particle 5
    // Spring 3: particle 6 ←→ particle 7

    let n_springs = 4;
    let n_particles = n_springs * 2;
    let rest_length = 1.0f32;
    let stiffness = 50.0f32;
    let lr = 0.001f32;
    let steps = 200;

    // Initial positions: springs are stretched/compressed
    //                  particle_a    particle_b    (rest = 1.0)
    // Spring 0:       (0, 0, 0)    (2, 0, 0)     stretched (dist=2.0)
    // Spring 1:       (0, 1, 0)    (0.3, 1, 0)   compressed (dist=0.3)
    // Spring 2:       (3, 0, 0)    (3, 1.5, 0)   stretched (dist=1.5)
    // Spring 3:       (5, 5, 0)    (5, 5, 0.5)   compressed (dist=0.5)

    let init_px = vec![0.0f32, 2.0,   0.0, 0.3,  3.0, 3.0,  5.0, 5.0];
    let init_py = vec![0.0f32, 0.0,   1.0, 1.0,  0.0, 1.5,  5.0, 5.0];
    let init_pz = vec![0.0f32, 0.0,   0.0, 0.0,  0.0, 0.0,  0.0, 0.5];

    let mut px = Array::from_vec(init_px.clone(), Device::Cuda(0));
    let mut py = Array::from_vec(init_py.clone(), Device::Cuda(0));
    let mut pz = Array::from_vec(init_pz.clone(), Device::Cuda(0));
    let rest = Array::from_vec(vec![rest_length; n_springs], Device::Cuda(0));
    let mut energy = Array::<f32>::zeros(n_springs, Device::Cuda(0));

    println!("Setup: {} springs, stiffness={}, rest_length={}, lr={}", n_springs, stiffness, rest_length, lr);

    // Print initial spring lengths
    println!("\nInitial spring lengths:");
    for s in 0..n_springs {
        let a = s * 2;
        let b = s * 2 + 1;
        let dx = init_px[a] - init_px[b];
        let dy = init_py[a] - init_py[b];
        let dz = init_pz[a] - init_pz[b];
        let len = (dx*dx + dy*dy + dz*dz).sqrt();
        println!("  Spring {}: {:.4} (target: {:.4})", s, len, rest_length);
    }

    // Forward to get initial energy
    pair_spring_energy::launch(
        &px, &py, &pz, &rest, stiffness, &mut energy,
        n_springs as i32, n_springs, 0,
    ).expect("forward failed");
    let init_energy: f32 = energy.to_vec().iter().sum();
    println!("\nInitial total energy: {:.6}", init_energy);

    // ── Optimization loop ──
    println!("\nOptimizing...");
    let start = std::time::Instant::now();

    for step in 0..steps {
        // Forward
        pair_spring_energy::launch(
            &px, &py, &pz, &rest, stiffness, &mut energy,
            n_springs as i32, n_springs, 0,
        ).expect("forward");

        // Backward: compute gradients of energy w.r.t. positions
        let mut adj_px = Array::<f32>::zeros(n_particles, Device::Cuda(0));
        let mut adj_py = Array::<f32>::zeros(n_particles, Device::Cuda(0));
        let mut adj_pz = Array::<f32>::zeros(n_particles, Device::Cuda(0));
        let mut adj_rest = Array::<f32>::zeros(n_springs, Device::Cuda(0));
        let mut adj_energy = Array::from_vec(vec![1.0f32; n_springs], Device::Cuda(0));

        pair_spring_energy::launch_adjoint(
            &px, &py, &pz, &rest, stiffness, &mut energy, n_springs as i32,
            &mut adj_px, &mut adj_py, &mut adj_pz, &mut adj_rest, &mut adj_energy,
            n_springs, 0,
        ).expect("adjoint");

        // Gradient descent step
        grad_step::launch(
            &mut px, &mut py, &mut pz,
            &adj_px, &adj_py, &adj_pz,
            lr, n_particles as i32, n_particles, 0,
        ).expect("step");

        if step % 50 == 0 || step == steps - 1 {
            let e: f32 = energy.to_vec().iter().sum();
            println!("  Step {:4}: energy = {:.6}", step, e);
        }
    }

    let elapsed = start.elapsed();

    // Final state
    pair_spring_energy::launch(
        &px, &py, &pz, &rest, stiffness, &mut energy,
        n_springs as i32, n_springs, 0,
    ).expect("forward");

    let final_energy: f32 = energy.to_vec().iter().sum();
    let fpx = px.to_vec();
    let fpy = py.to_vec();
    let fpz = pz.to_vec();

    println!("\n=============================================");
    println!("Results ({} steps in {:.3}s):", steps, elapsed.as_secs_f64());
    println!("  Energy: {:.6} → {:.6} ({:.1}% reduction)",
        init_energy, final_energy, (1.0 - final_energy / init_energy) * 100.0);

    println!("\nFinal spring lengths:");
    for s in 0..n_springs {
        let a = s * 2;
        let b = s * 2 + 1;
        let dx = fpx[a] - fpx[b];
        let dy = fpy[a] - fpy[b];
        let dz = fpz[a] - fpz[b];
        let len = (dx*dx + dy*dy + dz*dz).sqrt();
        println!("  Spring {}: {:.4} (target: {:.4}, error: {:.4})", s, len, rest_length, (len - rest_length).abs());
    }

    println!("\n🏁 Done.");
}
