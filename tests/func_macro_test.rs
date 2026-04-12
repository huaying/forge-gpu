//! Tests for #[func] and #[kernel] with device functions.

use forge_macros::{func, kernel};
use forge_runtime::Array;

// ── Device function: clamp ────────────────────────────────

#[func]
fn clamp_val(x: f32, lo: f32, hi: f32) -> f32 {
    if x < lo {
        return lo;
    }
    if x > hi {
        return hi;
    }
    return x;
}

#[test]
fn test_clamp_func_cuda_source() {
    let src = clamp_val::CUDA_SOURCE;
    assert!(src.contains("__device__"), "should have __device__ qualifier");
    assert!(src.contains("float clamp_val"), "should have function name");
    assert!(src.contains("float x"), "should have param x");
    assert!(src.contains("float lo"), "should have param lo");
    assert!(src.contains("float hi"), "should have param hi");
    assert!(src.contains("return lo"), "should return lo");
    assert!(src.contains("return hi"), "should return hi");
    assert!(src.contains("return x"), "should return x");
    println!("clamp_val CUDA:\n{}", src);
}

// ── Device function: square ───────────────────────────────

#[func]
fn square(x: f32) -> f32 {
    return x * x;
}

#[test]
fn test_square_func_cuda_source() {
    let src = square::CUDA_SOURCE;
    assert!(src.contains("__device__ float square"));
    assert!(src.contains("(x * x)"));
    println!("square CUDA:\n{}", src);
}

// ── Kernel that uses a device function ────────────────────

#[kernel]
fn apply_clamp(data: &mut Array<f32>, lo: f32, hi: f32, n: i32) {
    let i = thread_id();
    if i < n {
        data[i] = clamp_val(data[i], lo, hi);
    }
}

#[test]
fn test_apply_clamp_cuda_source() {
    let src = apply_clamp::CUDA_SOURCE;
    // The kernel itself calls clamp_val but doesn't include the __device__ function
    // That's expected — it's prepended at launch time via launch_with_funcs
    assert!(src.contains("clamp_val(data[i], lo, hi)"));
    println!("apply_clamp CUDA:\n{}", src);
}

// ── GPU test: kernel with device function ─────────────────

#[test]
fn test_clamp_gpu_execution() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let n = 1024;
    // Data: -5.0, -4.0, ..., 0, 1, ..., 1018
    let data_vec: Vec<f32> = (0..n).map(|i| i as f32 - 5.0).collect();
    let mut data = Array::from_vec(data_vec, forge_runtime::Device::Cuda(0));

    // Clamp to [0.0, 100.0]
    apply_clamp::launch_with_funcs(
        &mut data,
        0.0,
        100.0,
        n as i32,
        n,
        0,
        &[clamp_val::CUDA_SOURCE],
    )
    .expect("launch_with_funcs failed");

    let result = data.to_vec();
    // First 5 values were negative → clamped to 0.0
    for i in 0..5 {
        assert!(
            (result[i] - 0.0).abs() < 1e-6,
            "result[{}] = {}, expected 0.0",
            i,
            result[i]
        );
    }
    // Values 6..106 are in range [1.0, 100.0]
    for i in 6..106 {
        let expected = i as f32 - 5.0;
        assert!(
            (result[i] - expected).abs() < 1e-6,
            "result[{}] = {}, expected {}",
            i,
            result[i],
            expected
        );
    }
    // Values beyond 105 should be clamped to 100.0
    for i in 106..n {
        assert!(
            (result[i] - 100.0).abs() < 1e-6,
            "result[{}] = {}, expected 100.0",
            i,
            result[i]
        );
    }
    eprintln!("✅ clamp_val device function: {} elements clamped correctly", n);
}

// ── GPU test: kernel with square device function ──────────

#[kernel]
fn apply_square(data: &mut Array<f32>, n: i32) {
    let i = thread_id();
    if i < n {
        data[i] = square(data[i]);
    }
}

#[test]
fn test_square_gpu_execution() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let n = 256;
    let data_vec: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let mut data = Array::from_vec(data_vec, forge_runtime::Device::Cuda(0));

    apply_square::launch_with_funcs(&mut data, n as i32, n, 0, &[square::CUDA_SOURCE])
        .expect("launch_with_funcs failed");

    let result = data.to_vec();
    for i in 0..n {
        let expected = (i as f32) * (i as f32);
        assert!(
            (result[i] - expected).abs() < 1e-3,
            "result[{}] = {}, expected {}",
            i,
            result[i],
            expected
        );
    }
    eprintln!("✅ square device function: {} elements squared correctly", n);
}

// ── Test: launch_async (no sync) ──────────────────────────

#[kernel]
fn fill_val(data: &mut Array<f32>, val: f32, n: i32) {
    let i = thread_id();
    if i < n {
        data[i] = val;
    }
}

#[test]
fn test_launch_async() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let n = 512;
    let mut data = Array::<f32>::zeros(n, forge_runtime::Device::Cuda(0));

    // Launch async — no sync inside
    fill_val::launch_async(&mut data, 42.0, n as i32, n, 0).expect("launch_async failed");

    // Must sync manually
    forge_runtime::cuda::synchronize(0);

    let result = data.to_vec();
    assert!(result.iter().all(|&x| (x - 42.0).abs() < 1e-6));
    eprintln!("✅ launch_async: {} elements filled correctly", n);
}
