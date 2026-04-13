//! Tests for CPU fallback execution of #[kernel] functions.

use forge_macros::kernel;
use forge_runtime::Array;

// ── Test kernel: add_one_cpu ────────────────────────────

#[kernel]
fn cpu_add_one(data: &mut Array<f32>, n: i32) {
    let i = thread_id();
    if i < n {
        data[i] += 1.0;
    }
}

#[test]
fn test_cpu_add_one_on_cpu_array() {
    let n = 1024;
    let mut data = Array::from_vec(vec![0.0f32; n], forge_runtime::Device::Cpu);
    cpu_add_one::launch_cpu(&mut data, n as i32, n).expect("CPU launch failed");
    let result = data.to_vec();
    assert!(result.iter().all(|&x| (x - 1.0).abs() < 1e-6),
        "Expected all 1.0, got {:?}", &result[..5]);
}

// ── Test kernel: scale_cpu ──────────────────────────────

#[kernel]
fn cpu_scale(data: &mut Array<f32>, factor: f32, n: i32) {
    let i = thread_id();
    if i < n {
        data[i] = data[i] * factor;
    }
}

#[test]
fn test_cpu_scale_on_cpu_array() {
    let n = 512;
    let mut data = Array::from_vec(vec![2.0f32; n], forge_runtime::Device::Cpu);
    cpu_scale::launch_cpu(&mut data, 3.0, n as i32, n).expect("CPU launch failed");
    let result = data.to_vec();
    assert!(result.iter().all(|&x| (x - 6.0).abs() < 1e-6),
        "Expected all 6.0, got {:?}", &result[..5]);
}

// ── Test kernel: copy_cpu ───────────────────────────────

#[kernel]
fn cpu_copy(dst: &mut Array<f32>, src: &Array<f32>, n: i32) {
    let i = thread_id();
    if i < n {
        dst[i] = src[i];
    }
}

#[test]
fn test_cpu_copy_on_cpu_arrays() {
    let n = 256;
    let src_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let src = Array::from_vec(src_data.clone(), forge_runtime::Device::Cpu);
    let mut dst = Array::from_vec(vec![0.0f32; n], forge_runtime::Device::Cpu);
    cpu_copy::launch_cpu(&mut dst, &src, n as i32, n).expect("CPU launch failed");
    let result = dst.to_vec();
    assert_eq!(result, src_data);
}

// ── Test: CPU fallback with GPU arrays (auto-transfer) ──

#[test]
fn test_cpu_add_one_on_gpu_array() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let n = 512;
    let mut data = Array::from_vec(vec![5.0f32; n], forge_runtime::Device::Cuda(0));
    cpu_add_one::launch_cpu(&mut data, n as i32, n).expect("CPU launch failed");
    let result = data.to_vec();
    assert!(result.iter().all(|&x| (x - 6.0).abs() < 1e-6),
        "Expected all 6.0, got {:?}", &result[..5]);
    // Verify data was written back to GPU
    assert_eq!(data.device(), forge_runtime::Device::Cuda(0));
}

// ── Test: CPU fallback matches GPU result ───────────────

#[kernel]
fn cpu_saxpy(y: &mut Array<f32>, x: &Array<f32>, a: f32, n: i32) {
    let i = thread_id();
    if i < n {
        y[i] = a * x[i] + y[i];
    }
}

#[test]
fn test_cpu_gpu_result_match() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let n = 1024;
    let x_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let y_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();

    // GPU execution
    let x_gpu = Array::from_vec(x_data.clone(), forge_runtime::Device::Cuda(0));
    let mut y_gpu = Array::from_vec(y_data.clone(), forge_runtime::Device::Cuda(0));
    cpu_saxpy::launch(&mut y_gpu, &x_gpu, 2.0, n as i32, n, 0).expect("GPU launch failed");
    let gpu_result = y_gpu.to_vec();

    // CPU execution
    let x_cpu = Array::from_vec(x_data, forge_runtime::Device::Cpu);
    let mut y_cpu = Array::from_vec(y_data, forge_runtime::Device::Cpu);
    cpu_saxpy::launch_cpu(&mut y_cpu, &x_cpu, 2.0, n as i32, n).expect("CPU launch failed");
    let cpu_result = y_cpu.to_vec();

    // Compare results
    for i in 0..n {
        let diff = (gpu_result[i] - cpu_result[i]).abs();
        assert!(diff < 1e-4, "Mismatch at index {}: gpu={}, cpu={}", i, gpu_result[i], cpu_result[i]);
    }
}

// ── Test: CPU fallback with math operations ─────────────

#[kernel]
fn cpu_math_ops(out: &mut Array<f32>, input: &Array<f32>, n: i32) {
    let i = thread_id();
    if i < n {
        out[i] = sqrt(input[i] * input[i] + 1.0);
    }
}

#[test]
fn test_cpu_math_ops() {
    let n = 256;
    let input_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
    let input = Array::from_vec(input_data.clone(), forge_runtime::Device::Cpu);
    let mut out = Array::from_vec(vec![0.0f32; n], forge_runtime::Device::Cpu);
    cpu_math_ops::launch_cpu(&mut out, &input, n as i32, n).expect("CPU launch failed");

    let result = out.to_vec();
    for i in 0..n {
        let x = input_data[i];
        let expected = (x * x + 1.0f32).sqrt();
        let diff = (result[i] - expected).abs();
        assert!(diff < 1e-5, "Mismatch at {}: got {}, expected {}", i, result[i], expected);
    }
}
