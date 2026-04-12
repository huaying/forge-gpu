//! # Kernel Macro Example
//!
//! Demonstrates using `#[kernel]` to write GPU kernels in Rust syntax.
//! The macro generates CUDA C++ and compiles it via nvrtc at runtime.

use forge_macros::kernel;
use forge_runtime::Array;
use forge_runtime::Device;

#[kernel]
fn add_one(data: &mut Array<f32>, n: i32) {
    let i = thread_id();
    if i < n {
        data[i] += 1.0;
    }
}

fn main() {
    println!("🔥 Forge #[kernel] Macro Demo\n");

    // Print the generated CUDA source
    println!("Generated CUDA source:");
    println!("─────────────────────────────");
    println!("{}", add_one::CUDA_SOURCE);
    println!("─────────────────────────────\n");

    // Initialize CUDA
    forge_runtime::cuda::init();
    let device_count = forge_runtime::cuda::device_count();
    println!("CUDA devices: {}", device_count);

    if device_count == 0 {
        println!("No GPU available, skipping launch.");
        return;
    }

    let n = 1024;
    let data_vec: Vec<f32> = vec![0.0; n];
    let mut data = Array::from_vec(data_vec, Device::Cuda(0));

    println!("Before: first 5 elements = {:?}", &data.to_vec()[..5]);

    // Launch the kernel!
    add_one::launch(&mut data, n as i32, n, 0).expect("kernel launch failed");

    let result = data.to_vec();
    println!("After:  first 5 elements = {:?}", &result[..5]);

    // Verify
    assert!(result.iter().all(|&x| (x - 1.0).abs() < 1e-6));
    println!("\n✅ All {} elements are 1.0 — kernel works!", n);
}
