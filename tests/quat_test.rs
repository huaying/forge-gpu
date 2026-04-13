//! Tests for quaternion operations (Rust side + CUDA GPU).

use forge_core::{Quatf, Vec3f};

// ── Rust-side tests ──

#[test]
fn test_quat_slerp_endpoints() {
    let a = Quatf::identity();
    let b = Quatf::from_axis_angle(Vec3f::new(0.0, 0.0, 1.0), std::f32::consts::FRAC_PI_2);

    // t=0 → a
    let s0 = Quatf::slerp(a, b, 0.0);
    assert!((s0.w - a.w).abs() < 1e-5, "slerp(t=0).w = {}", s0.w);
    assert!((s0.x - a.x).abs() < 1e-5);
    assert!((s0.y - a.y).abs() < 1e-5);
    assert!((s0.z - a.z).abs() < 1e-5);

    // t=1 → b
    let s1 = Quatf::slerp(a, b, 1.0);
    assert!((s1.w - b.w).abs() < 1e-5, "slerp(t=1).w = {} vs {}", s1.w, b.w);
    assert!((s1.z - b.z).abs() < 1e-5, "slerp(t=1).z = {} vs {}", s1.z, b.z);

    eprintln!("✅ quat slerp endpoints correct");
}

#[test]
fn test_quat_slerp_midpoint() {
    let a = Quatf::identity();
    let b = Quatf::from_axis_angle(Vec3f::new(0.0, 0.0, 1.0), std::f32::consts::FRAC_PI_2);

    // t=0.5 should give 45-degree rotation
    let mid = Quatf::slerp(a, b, 0.5);
    let v = Vec3f::new(1.0, 0.0, 0.0);
    let rotated = mid.rotate(v);

    // 45-degree rotation of (1,0,0) around Z → (cos45, sin45, 0) ≈ (0.707, 0.707, 0)
    let cos45 = std::f32::consts::FRAC_PI_4.cos();
    let sin45 = std::f32::consts::FRAC_PI_4.sin();
    assert!((rotated.x - cos45).abs() < 1e-4, "mid.x = {}", rotated.x);
    assert!((rotated.y - sin45).abs() < 1e-4, "mid.y = {}", rotated.y);
    assert!(rotated.z.abs() < 1e-4);

    eprintln!("✅ quat slerp midpoint: 45° rotation correct");
}

#[test]
fn test_quat_axis_angle_roundtrip() {
    let axis = Vec3f::new(0.0, 1.0, 0.0); // Y axis
    let angle = 1.23f32;
    let q = Quatf::from_axis_angle(axis, angle);
    let (ax2, ang2) = q.to_axis_angle();

    assert!((ax2.x - axis.x).abs() < 1e-5, "axis.x = {}", ax2.x);
    assert!((ax2.y - axis.y).abs() < 1e-5, "axis.y = {}", ax2.y);
    assert!((ax2.z - axis.z).abs() < 1e-5, "axis.z = {}", ax2.z);
    assert!((ang2 - angle).abs() < 1e-5, "angle = {} vs {}", ang2, angle);

    eprintln!("✅ quat axis-angle roundtrip correct");
}

#[test]
fn test_quat_axis_angle_roundtrip_diagonal() {
    let axis_raw = Vec3f::new(1.0, 1.0, 1.0);
    let axis = axis_raw.normalize();
    let angle = 2.0f32;
    let q = Quatf::from_axis_angle(axis, angle);
    let (ax2, ang2) = q.to_axis_angle();

    assert!((ax2.x - axis.x).abs() < 1e-4, "axis.x = {} vs {}", ax2.x, axis.x);
    assert!((ax2.y - axis.y).abs() < 1e-4, "axis.y = {} vs {}", ax2.y, axis.y);
    assert!((ax2.z - axis.z).abs() < 1e-4, "axis.z = {} vs {}", ax2.z, axis.z);
    assert!((ang2 - angle).abs() < 1e-4, "angle = {} vs {}", ang2, angle);

    eprintln!("✅ quat axis-angle roundtrip (diagonal axis) correct");
}

#[test]
fn test_quat_rotate_vector() {
    // 90° around Z: (1,0,0) → (0,1,0)
    let q = Quatf::from_axis_angle(Vec3f::new(0.0, 0.0, 1.0), std::f32::consts::FRAC_PI_2);
    let v = Vec3f::new(1.0, 0.0, 0.0);
    let r = q.rotate_vec(v);

    assert!((r.x - 0.0).abs() < 1e-5, "rotated.x = {}", r.x);
    assert!((r.y - 1.0).abs() < 1e-5, "rotated.y = {}", r.y);
    assert!(r.z.abs() < 1e-5);

    // 180° around Y: (1,0,0) → (-1,0,0)
    let q2 = Quatf::from_axis_angle(Vec3f::new(0.0, 1.0, 0.0), std::f32::consts::PI);
    let r2 = q2.rotate_vec(v);
    assert!((r2.x - (-1.0)).abs() < 1e-4, "180° rotated.x = {}", r2.x);
    assert!(r2.y.abs() < 1e-4);
    assert!(r2.z.abs() < 1e-4);

    eprintln!("✅ quat rotate_vec correct");
}

#[test]
fn test_quat_euler_roundtrip() {
    let roll = 0.3f32;
    let pitch = 0.5f32;
    let yaw = 0.7f32;
    let q = Quatf::from_euler(roll, pitch, yaw);
    let (r2, p2, y2) = q.to_euler();

    assert!((r2 - roll).abs() < 1e-5, "roll: {} vs {}", r2, roll);
    assert!((p2 - pitch).abs() < 1e-5, "pitch: {} vs {}", p2, pitch);
    assert!((y2 - yaw).abs() < 1e-5, "yaw: {} vs {}", y2, yaw);

    eprintln!("✅ quat euler roundtrip correct");
}

#[test]
fn test_quat_inverse() {
    let q = Quatf::from_axis_angle(Vec3f::new(1.0, 0.0, 0.0), 1.0);
    let qi = q.inverse();
    let product = q * qi;

    assert!((product.w - 1.0).abs() < 1e-5, "q*q^-1 w = {}", product.w);
    assert!(product.x.abs() < 1e-5);
    assert!(product.y.abs() < 1e-5);
    assert!(product.z.abs() < 1e-5);

    eprintln!("✅ quat inverse correct");
}

#[test]
fn test_quat_conjugate() {
    let q = Quatf::new(1.0, 2.0, 3.0, 4.0);
    let c = q.conjugate();
    assert_eq!(c, Quatf::new(-1.0, -2.0, -3.0, 4.0));

    eprintln!("✅ quat conjugate correct");
}

// ── GPU-side quaternion tests ──

#[test]
fn test_quat_slerp_gpu() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    use forge_runtime::{Array, Device};
    let device = Device::Cuda(0);

    // slerp between identity and 90° rotation around Z at t=0.5
    let a = Quatf::identity();
    let b = Quatf::from_axis_angle(Vec3f::new(0.0, 0.0, 1.0), std::f32::consts::FRAC_PI_2);

    let aw = Array::from_vec(vec![a.w], device);
    let ax = Array::from_vec(vec![a.x], device);
    let ay = Array::from_vec(vec![a.y], device);
    let az = Array::from_vec(vec![a.z], device);
    let bw = Array::from_vec(vec![b.w], device);
    let bx = Array::from_vec(vec![b.x], device);
    let by = Array::from_vec(vec![b.y], device);
    let bz = Array::from_vec(vec![b.z], device);
    let t_arr = Array::from_vec(vec![0.5f32], device);
    let mut ow = Array::<f32>::zeros(1, device);
    let mut ox = Array::<f32>::zeros(1, device);
    let mut oy = Array::<f32>::zeros(1, device);
    let mut oz = Array::<f32>::zeros(1, device);

    let source = r#"
__device__ void forge_quat_slerp(float aw, float ax, float ay, float az,
                                  float bw, float bx, float by, float bz,
                                  float t,
                                  float* ow, float* ox, float* oy, float* oz) {
    float d = aw*bw + ax*bx + ay*by + az*bz;
    if (d < 0.0f) { bw=-bw; bx=-bx; by=-by; bz=-bz; d=-d; }
    if (d > 0.9995f) {
        *ow = aw + t*(bw-aw); *ox = ax + t*(bx-ax);
        *oy = ay + t*(by-ay); *oz = az + t*(bz-az);
    } else {
        float theta = acosf(d);
        float sn = sinf(theta);
        float wa = sinf((1.0f-t)*theta)/sn;
        float wb = sinf(t*theta)/sn;
        *ow = wa*aw + wb*bw; *ox = wa*ax + wb*bx;
        *oy = wa*ay + wb*by; *oz = wa*az + wb*bz;
    }
    float len = sqrtf((*ow)*(*ow) + (*ox)*(*ox) + (*oy)*(*oy) + (*oz)*(*oz));
    if (len > 1e-8f) { *ow/=len; *ox/=len; *oy/=len; *oz/=len; }
}

extern "C" __global__ void test_slerp(
    const float* aw, const float* ax, const float* ay, const float* az,
    const float* bw, const float* bx, const float* by, const float* bz,
    const float* t,
    float* ow, float* ox, float* oy, float* oz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        forge_quat_slerp(aw[0], ax[0], ay[0], az[0],
                         bw[0], bx[0], by[0], bz[0],
                         t[0],
                         &ow[0], &ox[0], &oy[0], &oz[0]);
    }
}
"#;

    let kernel = forge_runtime::CompiledKernel::compile(source, "test_slerp").unwrap();
    let func = kernel.get_function(0).unwrap();
    let stream = forge_runtime::cuda::default_stream(0);
    let config = forge_runtime::cuda::LaunchConfig::for_num_elems(1);

    unsafe {
        use forge_runtime::cuda::PushKernelArg;
        let mut builder = stream.launch_builder(&func);
        builder.arg(aw.cuda_slice().unwrap());
        builder.arg(ax.cuda_slice().unwrap());
        builder.arg(ay.cuda_slice().unwrap());
        builder.arg(az.cuda_slice().unwrap());
        builder.arg(bw.cuda_slice().unwrap());
        builder.arg(bx.cuda_slice().unwrap());
        builder.arg(by.cuda_slice().unwrap());
        builder.arg(bz.cuda_slice().unwrap());
        builder.arg(t_arr.cuda_slice().unwrap());
        builder.arg(ow.cuda_slice_mut().unwrap());
        builder.arg(ox.cuda_slice_mut().unwrap());
        builder.arg(oy.cuda_slice_mut().unwrap());
        builder.arg(oz.cuda_slice_mut().unwrap());
        builder.launch(config).unwrap();
    }
    stream.synchronize().unwrap();

    let rw = ow.to_vec()[0];
    let rx = ox.to_vec()[0];
    let ry = oy.to_vec()[0];
    let rz = oz.to_vec()[0];

    // Compare with CPU slerp
    let cpu_result = Quatf::slerp(a, b, 0.5);
    assert!((rw - cpu_result.w).abs() < 1e-4, "GPU slerp w = {} vs {}", rw, cpu_result.w);
    assert!((rx - cpu_result.x).abs() < 1e-4);
    assert!((ry - cpu_result.y).abs() < 1e-4);
    assert!((rz - cpu_result.z).abs() < 1e-4, "GPU slerp z = {} vs {}", rz, cpu_result.z);

    eprintln!("✅ quat slerp GPU matches CPU");
}

#[test]
fn test_quat_rotate_gpu() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    use forge_runtime::{Array, Device};
    let device = Device::Cuda(0);

    // 90° around Z: rotate (1,0,0) → (0,1,0)
    let q = Quatf::from_axis_angle(Vec3f::new(0.0, 0.0, 1.0), std::f32::consts::FRAC_PI_2);

    let qw = Array::from_vec(vec![q.w], device);
    let qx = Array::from_vec(vec![q.x], device);
    let qy = Array::from_vec(vec![q.y], device);
    let qz = Array::from_vec(vec![q.z], device);
    let vx = Array::from_vec(vec![1.0f32], device);
    let vy = Array::from_vec(vec![0.0f32], device);
    let vz = Array::from_vec(vec![0.0f32], device);
    let mut ox = Array::<f32>::zeros(1, device);
    let mut oy = Array::<f32>::zeros(1, device);
    let mut oz = Array::<f32>::zeros(1, device);

    let source = r#"
__device__ void forge_quat_rotate(float qw, float qx, float qy, float qz,
                                   float vx, float vy, float vz,
                                   float* ox, float* oy, float* oz) {
    float tx = 2.0f*(qy*vz - qz*vy);
    float ty = 2.0f*(qz*vx - qx*vz);
    float tz = 2.0f*(qx*vy - qy*vx);
    *ox = vx + qw*tx + (qy*tz - qz*ty);
    *oy = vy + qw*ty + (qz*tx - qx*tz);
    *oz = vz + qw*tz + (qx*ty - qy*tx);
}

extern "C" __global__ void test_qrot(
    const float* qw, const float* qx, const float* qy, const float* qz,
    const float* vx, const float* vy, const float* vz,
    float* ox, float* oy, float* oz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        forge_quat_rotate(qw[0], qx[0], qy[0], qz[0],
                          vx[0], vy[0], vz[0],
                          &ox[0], &oy[0], &oz[0]);
    }
}
"#;

    let kernel = forge_runtime::CompiledKernel::compile(source, "test_qrot").unwrap();
    let func = kernel.get_function(0).unwrap();
    let stream = forge_runtime::cuda::default_stream(0);
    let config = forge_runtime::cuda::LaunchConfig::for_num_elems(1);

    unsafe {
        use forge_runtime::cuda::PushKernelArg;
        let mut builder = stream.launch_builder(&func);
        builder.arg(qw.cuda_slice().unwrap());
        builder.arg(qx.cuda_slice().unwrap());
        builder.arg(qy.cuda_slice().unwrap());
        builder.arg(qz.cuda_slice().unwrap());
        builder.arg(vx.cuda_slice().unwrap());
        builder.arg(vy.cuda_slice().unwrap());
        builder.arg(vz.cuda_slice().unwrap());
        builder.arg(ox.cuda_slice_mut().unwrap());
        builder.arg(oy.cuda_slice_mut().unwrap());
        builder.arg(oz.cuda_slice_mut().unwrap());
        builder.launch(config).unwrap();
    }
    stream.synchronize().unwrap();

    let rx = ox.to_vec()[0];
    let ry = oy.to_vec()[0];
    let rz = oz.to_vec()[0];

    assert!((rx - 0.0).abs() < 1e-4, "rotated x = {}", rx);
    assert!((ry - 1.0).abs() < 1e-4, "rotated y = {}", ry);
    assert!(rz.abs() < 1e-4, "rotated z = {}", rz);

    eprintln!("✅ quat rotate GPU: (1,0,0) → (0,1,0) via 90° Z rotation");
}
