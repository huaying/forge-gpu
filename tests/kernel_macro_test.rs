//! Snapshot tests for CUDA code generation from #[kernel].

use forge_macros::kernel;
use forge_runtime::Array;

// ── Test 1: simple add_one ────────────────────────────────

#[kernel]
fn add_one(data: &mut Array<f32>, n: i32) {
    let i = thread_id();
    if i < n {
        data[i] += 1.0;
    }
}

#[test]
fn test_add_one_cuda_source() {
    let src = add_one::CUDA_SOURCE;
    assert!(src.contains("extern \"C\" __global__ void add_one"));
    assert!(src.contains("float* data"));
    assert!(src.contains("int n"));
    assert!(src.contains("blockIdx.x * blockDim.x + threadIdx.x"));
    assert!(src.contains("data[i] += 1.0f"));
    println!("add_one CUDA:\n{}", src);
}

// ── Test 2: scale kernel with multiply ────────────────────

#[kernel]
fn scale(data: &mut Array<f32>, factor: f32, n: i32) {
    let i = thread_id();
    if i < n {
        data[i] = data[i] * factor;
    }
}

#[test]
fn test_scale_cuda_source() {
    let src = scale::CUDA_SOURCE;
    assert!(src.contains("float* data"));
    assert!(src.contains("float factor"));
    assert!(src.contains("int n"));
    assert!(src.contains("data[i] = (data[i] * factor)"));
    println!("scale CUDA:\n{}", src);
}

// ── Test 3: read-only input ──────────────────────────────

#[kernel]
fn copy_array(dst: &mut Array<f32>, src: &Array<f32>, n: i32) {
    let i = thread_id();
    if i < n {
        dst[i] = src[i];
    }
}

#[test]
fn test_copy_array_cuda_source() {
    let src = copy_array::CUDA_SOURCE;
    assert!(src.contains("float* dst"));
    assert!(src.contains("const float* src"));
    assert!(src.contains("dst[i] = src[i]"));
    println!("copy_array CUDA:\n{}", src);
}

// ── Test 4: math builtins ────────────────────────────────

#[kernel]
fn apply_sin(out: &mut Array<f32>, input: &Array<f32>, n: i32) {
    let i = thread_id();
    if i < n {
        out[i] = sin(input[i]);
    }
}

#[test]
fn test_apply_sin_cuda_source() {
    let src = apply_sin::CUDA_SOURCE;
    assert!(src.contains("sinf(input[i])"));
    println!("apply_sin CUDA:\n{}", src);
}

// ── Test 5: GPU execution (add_one) ─────────────────────

#[test]
fn test_add_one_gpu_execution() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let n = 2048;
    let mut data = Array::from_vec(vec![0.0f32; n], forge_runtime::Device::Cuda(0));
    add_one::launch(&mut data, n as i32, n, 0).expect("launch failed");
    let result = data.to_vec();
    assert!(result.iter().all(|&x| (x - 1.0).abs() < 1e-6));
}

// ── Test 6: GPU execution (scale) ───────────────────────

#[test]
fn test_scale_gpu_execution() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let n = 1024;
    let mut data = Array::from_vec(vec![2.0f32; n], forge_runtime::Device::Cuda(0));
    scale::launch(&mut data, 3.0, n as i32, n, 0).expect("launch failed");
    let result = data.to_vec();
    assert!(result.iter().all(|&x| (x - 6.0).abs() < 1e-6));
}

// ── Test 7: GPU execution (copy) ────────────────────────

#[test]
fn test_copy_gpu_execution() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let n = 512;
    let src_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let src = Array::from_vec(src_data.clone(), forge_runtime::Device::Cuda(0));
    let mut dst = Array::from_vec(vec![0.0f32; n], forge_runtime::Device::Cuda(0));
    copy_array::launch(&mut dst, &src, n as i32, n, 0).expect("launch failed");
    let result = dst.to_vec();
    assert_eq!(result, src_data);
}
