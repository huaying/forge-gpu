//! # Forge Spring Optimization — Differentiable Physics 🧠🔥
//!
//! Demonstrates autodiff by optimizing spring rest lengths.
//! Creates 4 springs with random rest lengths, then uses gradient descent
//! to minimize total energy by moving particles to equilibrium.
//!
//! Run: `CARGO_HOME=/home/horde/.cargo cargo run --release --example spring_optimize`

use forge_macros::kernel;
use forge_runtime::{cuda, Array, Device};

/// Spring energy: E = 0.5 * k * (|p_a - p_b| - rest)^2
/// Decomposed to scalar arrays for autodiff.
#[kernel(autodiff)]
fn spring_energy(
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
        let i0 = tid * 2;
        let i1 = tid * 2 + 1;

        let dx = px[i0] - px[i1];
        let dy = py[i0] - py[i1];
        let dz = pz[i0] - pz[i1];

        let dist_sq = dx * dx + dy * dy + dz * dz;
        let dist = sqrt(dist_sq);

        let stretch = dist - rest_len[tid];
        energy[tid] = stiffness * stretch * stretch * 0.5;
    }
}

/// Gradient descent update on positions.
#[kernel]
fn update_positions(
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
    println!("🧠🔥 Forge Spring Optimization Demo");
    println!("  Differentiable physics with autodiff");
    println!("  ────────────────────────────────────\n");

    cuda::init();
    if cuda::device_count() == 0 {
        println!("No GPU found — need CUDA to run.");
        return;
    }

    // ── Setup ──
    // 4 springs with random rest lengths, 8 particles in pairs
    // Springs are initially far from equilibrium
    let n_springs = 4i32;
    let n_particles = (n_springs * 2) as usize;
    let stiffness = 100.0f32;
    let lr = 0.0005f32;
    let iterations = 300;

    // Random rest lengths for each spring
    let rest_lengths = vec![0.8f32, 1.3, 0.5, 1.7];

    // Initial positions: pairs with wrong distances
    //  Spring 0: particles at (0,0,0) and (3,0,0) → dist=3.0, rest=0.8
    //  Spring 1: particles at (0,2,0) and (0,2,0.1) → dist=0.1, rest=1.3
    //  Spring 2: particles at (4,0,0) and (4,3,0) → dist=3.0, rest=0.5
    //  Spring 3: particles at (1,1,1) and (1,1,1.2) → dist=0.2, rest=1.7
    let init_px = vec![0.0f32, 3.0, 0.0, 0.0, 4.0, 4.0, 1.0, 1.0];
    let init_py = vec![0.0f32, 0.0, 2.0, 2.0, 0.0, 3.0, 1.0, 1.0];
    let init_pz = vec![0.0f32, 0.0, 0.0, 0.1, 0.0, 0.0, 1.0, 1.2];

    let mut px = Array::from_vec(init_px.clone(), Device::Cuda(0));
    let mut py = Array::from_vec(init_py.clone(), Device::Cuda(0));
    let mut pz = Array::from_vec(init_pz.clone(), Device::Cuda(0));
    let rest = Array::from_vec(rest_lengths.clone(), Device::Cuda(0));
    let mut energy = Array::<f32>::zeros(n_springs as usize, Device::Cuda(0));

    println!("Springs: {}", n_springs);
    println!("Stiffness: {}", stiffness);
    println!("Learning rate: {}", lr);
    println!("Iterations: {}\n", iterations);

    // Print initial state
    println!("Initial state:");
    println!("  {:>8} {:>10} {:>10} {:>10}", "Spring", "Distance", "Rest len", "Error");
    for s in 0..n_springs as usize {
        let a = s * 2;
        let b = s * 2 + 1;
        let dx = init_px[a] - init_px[b];
        let dy = init_py[a] - init_py[b];
        let dz = init_pz[a] - init_pz[b];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        println!(
            "  {:>8} {:>10.4} {:>10.4} {:>10.4}",
            s, dist, rest_lengths[s], (dist - rest_lengths[s]).abs()
        );
    }

    // Compute initial total energy
    spring_energy::launch(
        &px, &py, &pz, &rest, stiffness, &mut energy,
        n_springs, n_springs as usize, 0,
    )
    .expect("forward failed");
    let init_energy: f32 = energy.to_vec().iter().sum();
    println!("\n  Total energy: {:.4}\n", init_energy);

    // ── Optimization loop ──
    println!("Optimizing...");
    let start = std::time::Instant::now();

    for step in 0..iterations {
        // Forward pass
        spring_energy::launch(
            &px, &py, &pz, &rest, stiffness, &mut energy,
            n_springs, n_springs as usize, 0,
        )
        .expect("forward");

        // Backward pass — compute gradients of energy w.r.t. positions
        let mut adj_px = Array::<f32>::zeros(n_particles, Device::Cuda(0));
        let mut adj_py = Array::<f32>::zeros(n_particles, Device::Cuda(0));
        let mut adj_pz = Array::<f32>::zeros(n_particles, Device::Cuda(0));
        let mut adj_rest = Array::<f32>::zeros(n_springs as usize, Device::Cuda(0));
        let mut adj_energy = Array::from_vec(vec![1.0f32; n_springs as usize], Device::Cuda(0));

        spring_energy::launch_adjoint(
            &px, &py, &pz, &rest, stiffness, &mut energy, n_springs,
            &mut adj_px, &mut adj_py, &mut adj_pz, &mut adj_rest, &mut adj_energy,
            n_springs as usize, 0,
        )
        .expect("adjoint");

        // Gradient descent step
        update_positions::launch(
            &mut px, &mut py, &mut pz,
            &adj_px, &adj_py, &adj_pz,
            lr, n_particles as i32, n_particles, 0,
        )
        .expect("update");

        // Print progress
        if step % 50 == 0 || step == iterations - 1 {
            let e: f32 = energy.to_vec().iter().sum();
            println!("  Step {:>4}: energy = {:.6}", step, e);
        }
    }

    let elapsed = start.elapsed();

    // ── Final results ──
    spring_energy::launch(
        &px, &py, &pz, &rest, stiffness, &mut energy,
        n_springs, n_springs as usize, 0,
    )
    .expect("final forward");

    let final_energy: f32 = energy.to_vec().iter().sum();
    let fpx = px.to_vec();
    let fpy = py.to_vec();
    let fpz = pz.to_vec();

    println!("\n────────────────────────────────────");
    println!("Results ({} iterations in {:.3}s):", iterations, elapsed.as_secs_f64());
    println!(
        "  Energy: {:.4} → {:.6} ({:.1}% reduction)",
        init_energy,
        final_energy,
        (1.0 - final_energy / init_energy) * 100.0
    );

    println!("\nFinal spring state:");
    println!("  {:>8} {:>10} {:>10} {:>10}", "Spring", "Distance", "Rest len", "Error");
    for s in 0..n_springs as usize {
        let a = s * 2;
        let b = s * 2 + 1;
        let dx = fpx[a] - fpx[b];
        let dy = fpy[a] - fpy[b];
        let dz = fpz[a] - fpz[b];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        println!(
            "  {:>8} {:>10.4} {:>10.4} {:>10.4}",
            s, dist, rest_lengths[s], (dist - rest_lengths[s]).abs()
        );
    }

    println!("\n🏁 Optimization complete — springs converged to rest lengths.");
}
