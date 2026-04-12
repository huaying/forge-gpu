//! Tests for kernels using Vec3f and other Forge struct types.

use forge_macros::kernel;
use forge_runtime::{Array, Device};
use forge_core::Vec3f;

// ── Kernel with Vec3f arrays ──────────────────────────────

#[kernel]
fn integrate_vec3(
    pos: &mut Array<Vec3f>,
    vel: &Array<Vec3f>,
    dt: f32,
    n: i32,
) {
    let tid = thread_id();
    if tid < n {
        pos[tid] = pos[tid] + vel[tid] * dt;
    }
}

#[test]
fn test_integrate_vec3_cuda_source() {
    let src = integrate_vec3::CUDA_SOURCE;
    println!("integrate_vec3 CUDA:\n{}", src);
    assert!(src.contains("forge_vec3f"), "should use forge_vec3f struct");
    assert!(src.contains("forge_vec3f* pos"), "pos should be forge_vec3f*");
    assert!(src.contains("const forge_vec3f* vel"), "vel should be const forge_vec3f*");
    assert!(src.contains("float dt"), "dt should be float");
    assert!(src.contains("struct forge_vec3f"), "should define the struct");
    assert!(src.contains("float x;"), "struct should have x field");
    assert!(src.contains("float y;"), "struct should have y field");
    assert!(src.contains("float z;"), "struct should have z field");
    assert!(src.contains("operator+"), "should have + operator");
    assert!(src.contains("operator*"), "should have * operator");
}

// ── GPU execution: Vec3f integrate ────────────────────────

#[test]
fn test_integrate_vec3_gpu() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let n = 1024;
    let dt = 1.0f32;

    // All positions at origin, all velocities = (1, 2, 3)
    let pos_data = vec![Vec3f::new(0.0, 0.0, 0.0); n];
    let vel_data = vec![Vec3f::new(1.0, 2.0, 3.0); n];

    let mut pos = Array::from_vec(pos_data, Device::Cuda(0));
    let vel = Array::from_vec(vel_data, Device::Cuda(0));

    integrate_vec3::launch(&mut pos, &vel, dt, n as i32, n, 0)
        .expect("launch failed");

    let result = pos.to_vec();
    for i in 0..n {
        assert!(
            (result[i].x - 1.0).abs() < 1e-6 &&
            (result[i].y - 2.0).abs() < 1e-6 &&
            (result[i].z - 3.0).abs() < 1e-6,
            "result[{}] = {:?}, expected (1, 2, 3)",
            i, result[i]
        );
    }
    eprintln!("✅ Vec3f integrate: {} elements correct", n);
}

// ── Kernel with Vec3f constructor ─────────────────────────

#[kernel]
fn apply_gravity(
    vel: &mut Array<Vec3f>,
    gravity: f32,
    dt: f32,
    n: i32,
) {
    let tid = thread_id();
    if tid < n {
        vel[tid] = vel[tid] + Vec3f::new(0.0, gravity * dt, 0.0);
    }
}

#[test]
fn test_apply_gravity_cuda_source() {
    let src = apply_gravity::CUDA_SOURCE;
    println!("apply_gravity CUDA:\n{}", src);
    // Vec3f::new should become forge_vec3f{...}
    assert!(src.contains("forge_vec3f{"), "should use brace-init for Vec3f::new");
}

#[test]
fn test_apply_gravity_gpu() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let n = 512;
    let vel_data = vec![Vec3f::new(0.0, 0.0, 0.0); n];
    let mut vel = Array::from_vec(vel_data, Device::Cuda(0));

    let gravity = -9.81f32;
    let dt = 1.0f32 / 60.0;

    apply_gravity::launch(&mut vel, gravity, dt, n as i32, n, 0)
        .expect("launch failed");

    let result = vel.to_vec();
    let expected_vy = gravity * dt;
    for i in 0..n {
        assert!(
            (result[i].x).abs() < 1e-6 &&
            (result[i].y - expected_vy).abs() < 1e-4 &&
            (result[i].z).abs() < 1e-6,
            "result[{}] = {:?}, expected (0, {}, 0)",
            i, result[i], expected_vy
        );
    }
    eprintln!("✅ Vec3f gravity: {} elements correct", n);
}

// ── Kernel with field access ──────────────────────────────

#[kernel]
fn extract_y(
    pos: &Array<Vec3f>,
    y_values: &mut Array<f32>,
    n: i32,
) {
    let tid = thread_id();
    if tid < n {
        y_values[tid] = pos[tid].y;
    }
}

#[test]
fn test_extract_y_gpu() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let n = 256;
    let pos_data: Vec<Vec3f> = (0..n).map(|i| Vec3f::new(0.0, i as f32 * 10.0, 0.0)).collect();
    let pos = Array::from_vec(pos_data, Device::Cuda(0));
    let mut y_values = Array::<f32>::zeros(n, Device::Cuda(0));

    extract_y::launch(&pos, &mut y_values, n as i32, n, 0)
        .expect("launch failed");

    let result = y_values.to_vec();
    for i in 0..n {
        assert!(
            (result[i] - i as f32 * 10.0).abs() < 1e-3,
            "result[{}] = {}, expected {}",
            i, result[i], i as f32 * 10.0
        );
    }
    eprintln!("✅ Vec3f field access: {} elements correct", n);
}
