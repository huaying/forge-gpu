//! Tests for vector math CUDA builtins (dot3, length3, normalize3, cross3, outer3).

use forge_runtime::{Array, Device};

#[test]
fn test_dot3_gpu() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let device = Device::Cuda(0);
    let n = 4;

    // Input: 4 pairs of 3D vectors (stored as ax,ay,az, bx,by,bz per element)
    // Pair 0: (1,0,0) · (0,1,0) = 0
    // Pair 1: (1,2,3) · (4,5,6) = 4+10+18 = 32
    // Pair 2: (1,1,1) · (1,1,1) = 3
    // Pair 3: (3,0,4) · (0,5,0) = 0
    let ax = Array::from_vec(vec![1.0f32, 1.0, 1.0, 3.0], device);
    let ay = Array::from_vec(vec![0.0f32, 2.0, 1.0, 0.0], device);
    let az = Array::from_vec(vec![0.0f32, 3.0, 1.0, 4.0], device);
    let bx = Array::from_vec(vec![0.0f32, 4.0, 1.0, 0.0], device);
    let by = Array::from_vec(vec![1.0f32, 5.0, 1.0, 5.0], device);
    let bz = Array::from_vec(vec![0.0f32, 6.0, 1.0, 0.0], device);
    let mut out = Array::<f32>::zeros(n, device);

    let source = r#"
__device__ float forge_dot3(float ax, float ay, float az,
                             float bx, float by, float bz) {
    return ax*bx + ay*by + az*bz;
}
extern "C" __global__ void test_dot3(
    const float* ax, const float* ay, const float* az,
    const float* bx, const float* by, const float* bz,
    float* out, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = forge_dot3(ax[i], ay[i], az[i], bx[i], by[i], bz[i]);
    }
}
"#;

    let kernel = forge_runtime::CompiledKernel::compile(source, "test_dot3").unwrap();
    let func = kernel.get_function(0).unwrap();
    let stream = forge_runtime::cuda::default_stream(0);
    let config = forge_runtime::cuda::LaunchConfig::for_num_elems(n as u32);
    let n_i32 = n as i32;

    unsafe {
        use forge_runtime::cuda::PushKernelArg;
        let mut builder = stream.launch_builder(&func);
        builder.arg(ax.cuda_slice().unwrap());
        builder.arg(ay.cuda_slice().unwrap());
        builder.arg(az.cuda_slice().unwrap());
        builder.arg(bx.cuda_slice().unwrap());
        builder.arg(by.cuda_slice().unwrap());
        builder.arg(bz.cuda_slice().unwrap());
        builder.arg(out.cuda_slice_mut().unwrap());
        builder.arg(&n_i32);
        builder.launch(config).unwrap();
    }
    stream.synchronize().unwrap();

    let result = out.to_vec();
    assert!((result[0] - 0.0).abs() < 1e-5, "dot (1,0,0)·(0,1,0) = {}", result[0]);
    assert!((result[1] - 32.0).abs() < 1e-5, "dot (1,2,3)·(4,5,6) = {}", result[1]);
    assert!((result[2] - 3.0).abs() < 1e-5, "dot (1,1,1)·(1,1,1) = {}", result[2]);
    assert!((result[3] - 0.0).abs() < 1e-5, "dot (3,0,4)·(0,5,0) = {}", result[3]);

    eprintln!("✅ dot3 GPU: all dot products correct");
}

#[test]
fn test_normalize3_gpu() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let device = Device::Cuda(0);
    let n = 3;

    // Input vectors to normalize
    let vx = Array::from_vec(vec![3.0f32, 0.0, 1.0], device);
    let vy = Array::from_vec(vec![0.0f32, 4.0, 1.0], device);
    let vz = Array::from_vec(vec![4.0f32, 0.0, 1.0], device);
    let mut ox = Array::<f32>::zeros(n, device);
    let mut oy = Array::<f32>::zeros(n, device);
    let mut oz = Array::<f32>::zeros(n, device);

    let source = r#"
__device__ void forge_normalize3(float x, float y, float z,
                                  float* ox, float* oy, float* oz) {
    float len = sqrtf(x*x + y*y + z*z);
    float inv = (len > 1e-8f) ? 1.0f/len : 0.0f;
    *ox = x*inv; *oy = y*inv; *oz = z*inv;
}
extern "C" __global__ void test_normalize(
    const float* vx, const float* vy, const float* vz,
    float* ox, float* oy, float* oz, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        forge_normalize3(vx[i], vy[i], vz[i], &ox[i], &oy[i], &oz[i]);
    }
}
"#;

    let kernel = forge_runtime::CompiledKernel::compile(source, "test_normalize").unwrap();
    let func = kernel.get_function(0).unwrap();
    let stream = forge_runtime::cuda::default_stream(0);
    let config = forge_runtime::cuda::LaunchConfig::for_num_elems(n as u32);
    let n_i32 = n as i32;

    unsafe {
        use forge_runtime::cuda::PushKernelArg;
        let mut builder = stream.launch_builder(&func);
        builder.arg(vx.cuda_slice().unwrap());
        builder.arg(vy.cuda_slice().unwrap());
        builder.arg(vz.cuda_slice().unwrap());
        builder.arg(ox.cuda_slice_mut().unwrap());
        builder.arg(oy.cuda_slice_mut().unwrap());
        builder.arg(oz.cuda_slice_mut().unwrap());
        builder.arg(&n_i32);
        builder.launch(config).unwrap();
    }
    stream.synchronize().unwrap();

    let rx = ox.to_vec();
    let ry = oy.to_vec();
    let rz = oz.to_vec();

    // (3,0,4) → len=5 → (0.6, 0, 0.8)
    assert!((rx[0] - 0.6).abs() < 1e-5);
    assert!((ry[0] - 0.0).abs() < 1e-5);
    assert!((rz[0] - 0.8).abs() < 1e-5);

    // (0,4,0) → len=4 → (0, 1, 0)
    assert!((rx[1] - 0.0).abs() < 1e-5);
    assert!((ry[1] - 1.0).abs() < 1e-5);

    // All outputs should have unit length
    for i in 0..n {
        let len = (rx[i] * rx[i] + ry[i] * ry[i] + rz[i] * rz[i]).sqrt();
        assert!((len - 1.0).abs() < 1e-5, "normalized vec {} has len {}", i, len);
    }

    eprintln!("✅ normalize3 GPU: correct unit vectors");
}

#[test]
fn test_cross3_gpu() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let device = Device::Cuda(0);
    let n = 2;

    // cross(x_hat, y_hat) = z_hat
    // cross((1,2,3), (4,5,6)) = (2*6-3*5, 3*4-1*6, 1*5-2*4) = (-3, 6, -3)
    let ax = Array::from_vec(vec![1.0f32, 1.0], device);
    let ay = Array::from_vec(vec![0.0f32, 2.0], device);
    let az = Array::from_vec(vec![0.0f32, 3.0], device);
    let bx = Array::from_vec(vec![0.0f32, 4.0], device);
    let by = Array::from_vec(vec![1.0f32, 5.0], device);
    let bz = Array::from_vec(vec![0.0f32, 6.0], device);
    let mut ox = Array::<f32>::zeros(n, device);
    let mut oy = Array::<f32>::zeros(n, device);
    let mut oz = Array::<f32>::zeros(n, device);

    let source = r#"
__device__ void forge_cross3(float ax, float ay, float az,
                              float bx, float by, float bz,
                              float* ox, float* oy, float* oz) {
    *ox = ay*bz - az*by;
    *oy = az*bx - ax*bz;
    *oz = ax*by - ay*bx;
}
extern "C" __global__ void test_cross(
    const float* ax, const float* ay, const float* az,
    const float* bx, const float* by, const float* bz,
    float* ox, float* oy, float* oz, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        forge_cross3(ax[i], ay[i], az[i], bx[i], by[i], bz[i], &ox[i], &oy[i], &oz[i]);
    }
}
"#;

    let kernel = forge_runtime::CompiledKernel::compile(source, "test_cross").unwrap();
    let func = kernel.get_function(0).unwrap();
    let stream = forge_runtime::cuda::default_stream(0);
    let config = forge_runtime::cuda::LaunchConfig::for_num_elems(n as u32);
    let n_i32 = n as i32;

    unsafe {
        use forge_runtime::cuda::PushKernelArg;
        let mut builder = stream.launch_builder(&func);
        builder.arg(ax.cuda_slice().unwrap());
        builder.arg(ay.cuda_slice().unwrap());
        builder.arg(az.cuda_slice().unwrap());
        builder.arg(bx.cuda_slice().unwrap());
        builder.arg(by.cuda_slice().unwrap());
        builder.arg(bz.cuda_slice().unwrap());
        builder.arg(ox.cuda_slice_mut().unwrap());
        builder.arg(oy.cuda_slice_mut().unwrap());
        builder.arg(oz.cuda_slice_mut().unwrap());
        builder.arg(&n_i32);
        builder.launch(config).unwrap();
    }
    stream.synchronize().unwrap();

    let rx = ox.to_vec();
    let ry = oy.to_vec();
    let rz = oz.to_vec();

    // cross(x_hat, y_hat) = z_hat
    assert!((rx[0] - 0.0).abs() < 1e-5);
    assert!((ry[0] - 0.0).abs() < 1e-5);
    assert!((rz[0] - 1.0).abs() < 1e-5);

    // cross((1,2,3), (4,5,6)) = (-3, 6, -3)
    assert!((rx[1] - (-3.0)).abs() < 1e-5, "cross x = {}", rx[1]);
    assert!((ry[1] - 6.0).abs() < 1e-5, "cross y = {}", ry[1]);
    assert!((rz[1] - (-3.0)).abs() < 1e-5, "cross z = {}", rz[1]);

    eprintln!("✅ cross3 GPU: cross products correct");
}

#[test]
fn test_length3_gpu() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let device = Device::Cuda(0);
    let n = 3;

    let vx = Array::from_vec(vec![3.0f32, 0.0, 1.0], device);
    let vy = Array::from_vec(vec![4.0f32, 0.0, 0.0], device);
    let vz = Array::from_vec(vec![0.0f32, 5.0, 0.0], device);
    let mut out = Array::<f32>::zeros(n, device);

    let source = r#"
__device__ float forge_length3(float x, float y, float z) {
    return sqrtf(x*x + y*y + z*z);
}
extern "C" __global__ void test_length(
    const float* vx, const float* vy, const float* vz,
    float* out, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = forge_length3(vx[i], vy[i], vz[i]);
    }
}
"#;

    let kernel = forge_runtime::CompiledKernel::compile(source, "test_length").unwrap();
    let func = kernel.get_function(0).unwrap();
    let stream = forge_runtime::cuda::default_stream(0);
    let config = forge_runtime::cuda::LaunchConfig::for_num_elems(n as u32);
    let n_i32 = n as i32;

    unsafe {
        use forge_runtime::cuda::PushKernelArg;
        let mut builder = stream.launch_builder(&func);
        builder.arg(vx.cuda_slice().unwrap());
        builder.arg(vy.cuda_slice().unwrap());
        builder.arg(vz.cuda_slice().unwrap());
        builder.arg(out.cuda_slice_mut().unwrap());
        builder.arg(&n_i32);
        builder.launch(config).unwrap();
    }
    stream.synchronize().unwrap();

    let result = out.to_vec();
    assert!((result[0] - 5.0).abs() < 1e-5, "len(3,4,0) = {}", result[0]);
    assert!((result[1] - 5.0).abs() < 1e-5, "len(0,0,5) = {}", result[1]);
    assert!((result[2] - 1.0).abs() < 1e-5, "len(1,0,0) = {}", result[2]);

    eprintln!("✅ length3 GPU: lengths correct");
}
