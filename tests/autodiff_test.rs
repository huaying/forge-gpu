//! Tests for #[kernel(autodiff)] — automatic differentiation.

use forge_macros::kernel;
use forge_runtime::{Array, Device};

// ── Simple test: y = x^2 (via y = x * x) ─────────────────
// dy/dx = 2x

#[kernel(autodiff)]
fn square_kernel(input: &Array<f32>, output: &mut Array<f32>, n: i32) {
    let i = thread_id();
    if i < n {
        let x = input[i];
        let y = x * x;
        output[i] = y;
    }
}

#[test]
fn test_square_autodiff_source() {
    println!("=== Forward CUDA ===");
    println!("{}", square_kernel::CUDA_SOURCE);
    println!("=== Adjoint CUDA ===");
    println!("{}", square_kernel::ADJOINT_CUDA_SOURCE);

    assert!(square_kernel::ADJOINT_CUDA_SOURCE.contains("adjoint"));
    assert!(square_kernel::ADJOINT_CUDA_SOURCE.contains("adj_"));
}

#[test]
fn test_square_autodiff_gpu() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let n = 256;
    // Input: [0, 1, 2, ..., 255]
    let input_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let input = Array::from_vec(input_data.clone(), Device::Cuda(0));
    let mut output = Array::<f32>::zeros(n, Device::Cuda(0));

    // Forward pass: output = input^2
    square_kernel::launch(&input, &mut output, n as i32, n, 0)
        .expect("forward launch failed");

    let fwd_result = output.to_vec();
    for i in 0..n {
        let expected = (i as f32) * (i as f32);
        assert!(
            (fwd_result[i] - expected).abs() < 1e-3,
            "forward[{}] = {}, expected {}",
            i, fwd_result[i], expected
        );
    }

    // Backward pass: seed adj_output = 1.0, expect adj_input = 2*x
    let mut adj_input = Array::<f32>::zeros(n, Device::Cuda(0));
    let mut adj_output = Array::from_vec(vec![1.0f32; n], Device::Cuda(0));

    square_kernel::launch_adjoint(
        &input, &mut output, n as i32,
        &mut adj_input, &mut adj_output,
        n, 0,
    ).expect("adjoint launch failed");

    let grad = adj_input.to_vec();
    for i in 0..n {
        let expected_grad = 2.0 * (i as f32);
        assert!(
            (grad[i] - expected_grad).abs() < 1e-2,
            "grad[{}] = {}, expected {} (2*x where x={})",
            i, grad[i], expected_grad, i
        );
    }
    eprintln!("✅ Autodiff square: gradients correct for {} elements", n);
}

// ── Test: y = a*x + b (linear) ────────────────────────────
// dy/dx = a (scalar), dy/da = x, dy/db = 1

#[kernel(autodiff)]
fn saxpy_kernel(x: &Array<f32>, y: &mut Array<f32>, a: f32, b: f32, n: i32) {
    let i = thread_id();
    if i < n {
        let val = x[i];
        let scaled = val * a;
        let result = scaled + b;
        y[i] = result;
    }
}

#[test]
fn test_saxpy_autodiff_gpu() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let n = 128;
    let a = 3.0f32;
    let b = 1.0f32;
    let x_data: Vec<f32> = (0..n).map(|i| i as f32).collect();

    let x = Array::from_vec(x_data.clone(), Device::Cuda(0));
    let mut y = Array::<f32>::zeros(n, Device::Cuda(0));

    // Forward: y = 3*x + 1
    saxpy_kernel::launch(&x, &mut y, a, b, n as i32, n, 0)
        .expect("forward failed");

    let fwd = y.to_vec();
    assert!((fwd[0] - 1.0).abs() < 1e-6);   // 3*0 + 1
    assert!((fwd[1] - 4.0).abs() < 1e-6);   // 3*1 + 1
    assert!((fwd[10] - 31.0).abs() < 1e-4); // 3*10 + 1

    // Backward: dy/dx should be a=3 for all elements
    let mut adj_x = Array::<f32>::zeros(n, Device::Cuda(0));
    let mut adj_y = Array::from_vec(vec![1.0f32; n], Device::Cuda(0));

    saxpy_kernel::launch_adjoint(
        &x, &mut y, a, b, n as i32,
        &mut adj_x, &mut adj_y,
        n, 0,
    ).expect("adjoint failed");

    let grad_x = adj_x.to_vec();
    for i in 0..n {
        assert!(
            (grad_x[i] - a).abs() < 1e-4,
            "grad_x[{}] = {}, expected {} (= a)",
            i, grad_x[i], a
        );
    }
    eprintln!("✅ Autodiff saxpy: dy/dx = {} (correct) for {} elements", a, n);
}
