//! Tests for 3D noise functions (GPU preamble).
//!
//! Verifies noise output range and smoothness properties
//! by compiling and running CUDA kernels that use forge_noise3.

use forge_runtime::{Array, Device};

#[test]
fn test_noise3_output_range() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let n = 1024;
    let device = Device::Cuda(0);

    // Generate test coordinates
    let coords: Vec<f32> = (0..n).map(|i| (i as f32) * 0.37 - 50.0).collect();
    let input = Array::from_vec(coords, device);
    let mut output = Array::<f32>::zeros(n, device);

    let source = r#"
__device__ float forge_fract(float x) { return x - floorf(x); }
__device__ float forge_hash(float n) {
    return forge_fract(sinf(n) * 43758.5453123f);
}
__device__ float forge_noise3(float x, float y, float z) {
    float ix = floorf(x), iy = floorf(y), iz = floorf(z);
    float fx = x - ix, fy = y - iy, fz = z - iz;
    float ux = fx*fx*(3.0f - 2.0f*fx);
    float uy = fy*fy*(3.0f - 2.0f*fy);
    float uz = fz*fz*(3.0f - 2.0f*fz);
    float n = ix + iy * 157.0f + iz * 113.0f;
    float a  = forge_hash(n);
    float b  = forge_hash(n + 1.0f);
    float c  = forge_hash(n + 157.0f);
    float d  = forge_hash(n + 158.0f);
    float e  = forge_hash(n + 113.0f);
    float ff = forge_hash(n + 114.0f);
    float g  = forge_hash(n + 270.0f);
    float h  = forge_hash(n + 271.0f);
    float x1 = a + (b-a)*ux + (c-a)*uy + (a-b-c+d)*ux*uy;
    float x2 = e + (ff-e)*ux + (g-e)*uy + (e-ff-g+h)*ux*uy;
    return x1 + (x2 - x1)*uz;
}

extern "C" __global__ void test_noise(const float* coords, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = coords[i];
        out[i] = forge_noise3(x, x * 1.3f, x * 0.7f);
    }
}
"#;

    let kernel = forge_runtime::CompiledKernel::compile(source, "test_noise").unwrap();
    let func = kernel.get_function(0).unwrap();
    let stream = forge_runtime::cuda::default_stream(0);
    let config = forge_runtime::cuda::LaunchConfig::for_num_elems(n as u32);

    unsafe {
        use forge_runtime::cuda::PushKernelArg;
        let mut builder = stream.launch_builder(&func);
        let n_i32 = n as i32;
        builder.arg(input.cuda_slice().unwrap());
        builder.arg(output.cuda_slice_mut().unwrap());
        builder.arg(&n_i32);
        builder.launch(config).unwrap();
    }
    stream.synchronize().unwrap();

    let result = output.to_vec();
    for (i, &val) in result.iter().enumerate() {
        assert!(
            val >= 0.0 && val <= 1.0,
            "noise3 output out of range at index {}: {}",
            i, val
        );
    }

    eprintln!("✅ noise3 output range: all {} values in [0, 1]", n);
}

#[test]
fn test_noise3_smoothness() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let n = 1024;
    let device = Device::Cuda(0);

    // Use very small steps to verify smoothness
    let step = 0.001f32;
    let coords: Vec<f32> = (0..n).map(|i| (i as f32) * step).collect();
    let input = Array::from_vec(coords, device);
    let mut output = Array::<f32>::zeros(n, device);

    let source = r#"
__device__ float forge_fract(float x) { return x - floorf(x); }
__device__ float forge_hash(float n) {
    return forge_fract(sinf(n) * 43758.5453123f);
}
__device__ float forge_noise3(float x, float y, float z) {
    float ix = floorf(x), iy = floorf(y), iz = floorf(z);
    float fx = x - ix, fy = y - iy, fz = z - iz;
    float ux = fx*fx*(3.0f - 2.0f*fx);
    float uy = fy*fy*(3.0f - 2.0f*fy);
    float uz = fz*fz*(3.0f - 2.0f*fz);
    float n = ix + iy * 157.0f + iz * 113.0f;
    float a  = forge_hash(n);
    float b  = forge_hash(n + 1.0f);
    float c  = forge_hash(n + 157.0f);
    float d  = forge_hash(n + 158.0f);
    float e  = forge_hash(n + 113.0f);
    float ff = forge_hash(n + 114.0f);
    float g  = forge_hash(n + 270.0f);
    float h  = forge_hash(n + 271.0f);
    float x1 = a + (b-a)*ux + (c-a)*uy + (a-b-c+d)*ux*uy;
    float x2 = e + (ff-e)*ux + (g-e)*uy + (e-ff-g+h)*ux*uy;
    return x1 + (x2 - x1)*uz;
}

extern "C" __global__ void test_noise_smooth(const float* coords, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = coords[i];
        out[i] = forge_noise3(x, x, x);
    }
}
"#;

    let kernel = forge_runtime::CompiledKernel::compile(source, "test_noise_smooth").unwrap();
    let func = kernel.get_function(0).unwrap();
    let stream = forge_runtime::cuda::default_stream(0);
    let config = forge_runtime::cuda::LaunchConfig::for_num_elems(n as u32);

    unsafe {
        use forge_runtime::cuda::PushKernelArg;
        let mut builder = stream.launch_builder(&func);
        let n_i32 = n as i32;
        builder.arg(input.cuda_slice().unwrap());
        builder.arg(output.cuda_slice_mut().unwrap());
        builder.arg(&n_i32);
        builder.launch(config).unwrap();
    }
    stream.synchronize().unwrap();

    let result = output.to_vec();

    // Check that adjacent samples (0.001 apart) are close to each other
    // For smooth noise, the max derivative is bounded
    let mut max_diff = 0.0f32;
    for i in 1..n {
        let diff = (result[i] - result[i - 1]).abs();
        max_diff = max_diff.max(diff);
    }

    // For step=0.001, the max difference between adjacent samples should be small
    assert!(
        max_diff < 0.1,
        "noise3 not smooth: max adjacent diff = {} (expected < 0.1 for step={})",
        max_diff, step
    );

    eprintln!("✅ noise3 smoothness: max adjacent diff = {:.6} (step={})", max_diff, step);
}

#[test]
fn test_noise3_not_constant() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let n = 256;
    let device = Device::Cuda(0);

    let coords: Vec<f32> = (0..n).map(|i| (i as f32) * 1.7).collect();
    let input = Array::from_vec(coords, device);
    let mut output = Array::<f32>::zeros(n, device);

    let source = r#"
__device__ float forge_fract(float x) { return x - floorf(x); }
__device__ float forge_hash(float n) {
    return forge_fract(sinf(n) * 43758.5453123f);
}
__device__ float forge_noise3(float x, float y, float z) {
    float ix = floorf(x), iy = floorf(y), iz = floorf(z);
    float fx = x - ix, fy = y - iy, fz = z - iz;
    float ux = fx*fx*(3.0f - 2.0f*fx);
    float uy = fy*fy*(3.0f - 2.0f*fy);
    float uz = fz*fz*(3.0f - 2.0f*fz);
    float n = ix + iy * 157.0f + iz * 113.0f;
    float a  = forge_hash(n);
    float b  = forge_hash(n + 1.0f);
    float c  = forge_hash(n + 157.0f);
    float d  = forge_hash(n + 158.0f);
    float e  = forge_hash(n + 113.0f);
    float ff = forge_hash(n + 114.0f);
    float g  = forge_hash(n + 270.0f);
    float h  = forge_hash(n + 271.0f);
    float x1 = a + (b-a)*ux + (c-a)*uy + (a-b-c+d)*ux*uy;
    float x2 = e + (ff-e)*ux + (g-e)*uy + (e-ff-g+h)*ux*uy;
    return x1 + (x2 - x1)*uz;
}

extern "C" __global__ void test_noise_var(const float* coords, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = coords[i];
        out[i] = forge_noise3(x, x * 0.5f, x * 0.3f);
    }
}
"#;

    let kernel = forge_runtime::CompiledKernel::compile(source, "test_noise_var").unwrap();
    let func = kernel.get_function(0).unwrap();
    let stream = forge_runtime::cuda::default_stream(0);
    let config = forge_runtime::cuda::LaunchConfig::for_num_elems(n as u32);

    unsafe {
        use forge_runtime::cuda::PushKernelArg;
        let mut builder = stream.launch_builder(&func);
        let n_i32 = n as i32;
        builder.arg(input.cuda_slice().unwrap());
        builder.arg(output.cuda_slice_mut().unwrap());
        builder.arg(&n_i32);
        builder.launch(config).unwrap();
    }
    stream.synchronize().unwrap();

    let result = output.to_vec();

    // Verify the noise has some variance (not all the same value)
    let mean: f32 = result.iter().sum::<f32>() / n as f32;
    let variance: f32 = result.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;

    assert!(
        variance > 0.001,
        "noise3 has no variance: mean={}, var={} — noise is constant!",
        mean, variance
    );

    eprintln!("✅ noise3 variance: mean={:.4}, var={:.4}", mean, variance);
}
