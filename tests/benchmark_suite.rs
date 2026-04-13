//! Benchmark suite for Forge GPU — measures key performance metrics.
//!
//! Run with: cargo test --release --features cuda --test benchmark_suite -- --nocapture --ignored

use forge_macros::kernel;
use forge_runtime::{Array, Device};

fn skip_if_no_gpu() -> bool {
    forge_runtime::cuda::init();
    forge_runtime::cuda::device_count() == 0
}

// ── Kernels for benchmarks ──

#[kernel]
fn empty_kernel(data: &Array<f32>, n: i32) {
    let _i = thread_id();
}

#[kernel]
fn add_one(data: &mut Array<f32>, n: i32) {
    let i = thread_id();
    if i < n {
        data[i] += 1.0;
    }
}

#[kernel]
fn saxpy(x: &Array<f32>, y: &mut Array<f32>, a: f32, n: i32) {
    let i = thread_id();
    if i < n {
        y[i] = a * x[i] + y[i];
    }
}

#[kernel]
fn reduce_sum_partial(data: &Array<f32>, partial: &mut Array<f32>, n: i32) {
    let i = thread_id();
    if i < n {
        // Simple: each thread adds its value to partial[0] via atomic
        let val = data[i];
        partial[i] = val;  // Just copy for now; real reduction done host-side
    }
}

// ── Benchmark helpers ──

fn bench<F: FnMut()>(name: &str, mut f: F, iterations: usize) {
    // Warm up
    f();

    let start = std::time::Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iterations as u32;

    if per_iter.as_micros() < 1000 {
        println!("BENCH {}: {:.1} µs", name, per_iter.as_nanos() as f64 / 1000.0);
    } else if per_iter.as_millis() < 1000 {
        println!("BENCH {}: {:.2} ms", name, per_iter.as_micros() as f64 / 1000.0);
    } else {
        println!("BENCH {}: {:.3} s", name, per_iter.as_secs_f64());
    }
}

// ── Benchmarks ──

#[test]
#[ignore]
fn bench_kernel_launch_overhead() {
    if skip_if_no_gpu() { return; }
    let data = Array::<f32>::zeros(1, Device::Cuda(0));
    let n = 1i32;

    // Sync launch (like before)
    bench("kernel_launch_sync", || {
        empty_kernel::launch(&data, n, 1, 0).unwrap();
    }, 1000);

    // Async launch (fair comparison with Warp)
    bench("kernel_launch_async", || {
        empty_kernel::launch_async(&data, n, 1, 0).unwrap();
    }, 1000);
    // Final sync
    forge_runtime::cuda::synchronize(0);
}

#[test]
#[ignore]
fn bench_memcpy_htod() {
    if skip_if_no_gpu() { return; }

    for &size in &[1_000_000usize, 10_000_000, 100_000_000] {
        let data: Vec<f32> = vec![1.0; size];
        let label = format!("memcpy_htod_{}M", size / 1_000_000);

        // Warm up memory pool
        let _ = Array::from_slice(&data, Device::Cuda(0));

        let iterations: u32 = if size >= 100_000_000 { 5 } else { 20 };

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _gpu = Array::from_slice(&data, Device::Cuda(0));
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed / iterations;
        let bytes = size * 4;
        let gbps = bytes as f64 / per_iter.as_secs_f64() / 1e9;

        println!("BENCH {}: {:.3} ms ({:.1} GB/s)", label, per_iter.as_micros() as f64 / 1000.0, gbps);
    }
}

#[test]
#[ignore]
fn bench_memcpy_dtoh() {
    if skip_if_no_gpu() { return; }

    for &size in &[1_000_000usize, 10_000_000, 100_000_000] {
        let gpu = Array::<f32>::fill(size, 1.0, Device::Cuda(0));
        let label = format!("memcpy_dtoh_{}M", size / 1_000_000);
        let iterations = if size >= 100_000_000 { 5 } else { 20 };

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _v = gpu.to_vec();
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed / iterations as u32;
        let bytes = size * 4;
        let gbps = bytes as f64 / per_iter.as_secs_f64() / 1e9;

        println!("BENCH {}: {:.2} ms ({:.1} GB/s)", label, per_iter.as_millis(), gbps);
    }
}

#[test]
#[ignore]
fn bench_saxpy() {
    if skip_if_no_gpu() { return; }

    for &size in &[1_000_000usize, 10_000_000, 100_000_000] {
        let x = Array::<f32>::fill(size, 1.0, Device::Cuda(0));
        let mut y = Array::<f32>::fill(size, 2.0, Device::Cuda(0));
        let a = 3.0f32;
        let n = size as i32;
        let iterations = 100;

        // Warm up
        saxpy::launch_async(&x, &mut y, a, n, size, 0).unwrap();
        forge_runtime::cuda::synchronize(0);

        // Async launch (fair comparison with Warp)
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            saxpy::launch_async(&x, &mut y, a, n, size, 0).unwrap();
        }
        forge_runtime::cuda::synchronize(0);
        let elapsed = start.elapsed();
        let per_iter = elapsed / iterations as u32;
        let bytes = 3 * size * 4;
        let gbps = bytes as f64 / per_iter.as_secs_f64() / 1e9;
        let label = format!("saxpy_{}M", size / 1_000_000);
        println!("BENCH {}: {:.3} ms ({:.0} GB/s)", label, per_iter.as_micros() as f64 / 1000.0, gbps);
    }
}

#[test]
#[ignore]
fn bench_reduction() {
    if skip_if_no_gpu() { return; }

    for &size in &[1_000_000usize, 10_000_000] {
        let data = Array::<f32>::fill(size, 1.0, Device::Cuda(0));
        let label = format!("sum_dtoh_{}M", size / 1_000_000);

        bench(&label, || {
            let v = data.to_vec();
            let _sum: f32 = v.iter().sum();
        }, 20);
    }
}

#[test]
#[ignore]
fn bench_sph_simulation() {
    if skip_if_no_gpu() { return; }

    println!("BENCH SPH simulation (via forge CLI):");

    for &count in &[50_000usize, 100_000, 500_000] {
        let toml = format!(r#"
[simulation]
name = "bench-{}k"
type = "particles"
dt = 0.0001
substeps = 10
duration = 0.5
count = {}

[[fields]]
name = "position"
type = "vec3f"
count = {}
init = {{ type = "random", min = [-1.0, 0.0, -0.5], max = [1.0, 3.0, 0.5] }}

[[fields]]
name = "velocity"
type = "vec3f"
init = {{ type = "zero" }}

[spatial]
type = "hashgrid"
cell_size = 0.04
grid_dims = [75, 100, 40]

[[forces]]
type = "sph_density"
smoothing_radius = 0.02

[[forces]]
type = "sph_pressure"
gas_constant = 2000.0
rest_density = 1000.0

[[forces]]
type = "sph_viscosity"
coefficient = 0.008

[[forces]]
type = "gravity"
value = [0.0, -9.81, 0.0]

[[constraints]]
type = "box"
min = [-1.5, 0.0, -0.5]
max = [2.0, 4.0, 0.5]
restitution = 0.3
"#, count / 1000, count, count);

        let path = format!("/tmp/bench_sph_{}k.toml", count / 1000);
        std::fs::write(&path, toml).unwrap();

        let output = std::process::Command::new("target/release/forge")
            .args(["run", &path])
            .current_dir("/home/horde/.openclaw/workspace/forge-gpu")
            .output()
            .expect("failed to run forge");

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if let Some(line) = stdout.lines().chain(stderr.lines()).find(|l| l.contains("Throughput")) {
            println!("BENCH sph_{}k: {}", count / 1000, line.trim());
        } else {
            println!("BENCH sph_{}k: FAILED", count / 1000);
        }
    }
}

#[test]
#[ignore]
fn bench_cpu_vs_gpu() {
    if skip_if_no_gpu() { return; }

    let size = 1_000_000usize;
    let mut cpu_arr = Array::<f32>::fill(size, 1.0, Device::Cpu);
    let mut gpu_arr = Array::<f32>::fill(size, 1.0, Device::Cuda(0));
    let n = size as i32;

    // CPU
    let start = std::time::Instant::now();
    for _ in 0..10 {
        add_one::launch_cpu(&mut cpu_arr, n, size).unwrap();
    }
    let cpu_time = start.elapsed() / 10;

    // GPU
    let start = std::time::Instant::now();
    for _ in 0..100 {
        add_one::launch(&mut gpu_arr, n, size, 0).unwrap();
    }
    let gpu_time = start.elapsed() / 100;

    let speedup = cpu_time.as_nanos() as f64 / gpu_time.as_nanos() as f64;

    println!("BENCH cpu_vs_gpu_1M:");
    println!("  CPU: {:.2} ms", cpu_time.as_micros() as f64 / 1000.0);
    println!("  GPU: {:.2} ms", gpu_time.as_micros() as f64 / 1000.0);
    println!("  Speedup: {:.1}x", speedup);
}
