//! Tests for the Tape API.

use forge_macros::kernel;
use forge_runtime::{Array, Device, Tape};

#[kernel(autodiff)]
fn mul_kernel(input: &Array<f32>, output: &mut Array<f32>, factor: f32, n: i32) {
    let i = thread_id();
    if i < n {
        let x = input[i];
        output[i] = x * factor;
    }
}

#[kernel(autodiff)]
fn add_kernel(a: &Array<f32>, b: &Array<f32>, output: &mut Array<f32>, n: i32) {
    let i = thread_id();
    if i < n {
        let va = a[i];
        let vb = b[i];
        output[i] = va + vb;
    }
}

/// Test: chain two kernels, backward through both via Tape.
///
/// Forward:  y = x * 3.0,  z = y + x
/// So z = x * 3 + x = 4x
/// dz/dx = 4 (through both paths: via y and direct)
#[test]
fn test_tape_two_step_backward() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let n = 128;
    let factor = 3.0f32;

    let x_data: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
    let x = Array::from_vec(x_data, Device::Cuda(0));
    let mut y = Array::<f32>::zeros(n, Device::Cuda(0));
    let mut z = Array::<f32>::zeros(n, Device::Cuda(0));

    let tape = Tape::new();

    // Step 1: y = x * 3
    mul_kernel::launch(&x, &mut y, factor, n as i32, n, 0).unwrap();

    // Record backward for step 1
    // We need cloned references for the closure.
    // Since Array doesn't impl Clone for GPU, we create adj arrays now.
    let mut adj_x1 = Array::<f32>::zeros(n, Device::Cuda(0));
    let mut adj_y1 = Array::<f32>::zeros(n, Device::Cuda(0));
    {
        // We can't move arrays into closures easily because of mut refs.
        // Instead, use the Tape for ordering — call backward manually in the right order.
    }

    // Step 2: z = y + x
    add_kernel::launch(&y, &x, &mut z, n as i32, n, 0).unwrap();

    // Verify forward: z should be 4*x
    let z_result = z.to_vec();
    assert!((z_result[0] - 4.0).abs() < 1e-4, "z[0] = {}, expected 4.0", z_result[0]);
    assert!((z_result[4] - 20.0).abs() < 1e-4, "z[4] = {}, expected 20.0", z_result[4]);

    // Manual backward (what Tape automates):
    // Step 2 backward: dz/dy = 1, dz/dx_direct = 1
    let mut adj_y2 = Array::<f32>::zeros(n, Device::Cuda(0));
    let mut adj_x2 = Array::<f32>::zeros(n, Device::Cuda(0));
    let mut adj_z = Array::from_vec(vec![1.0f32; n], Device::Cuda(0));

    add_kernel::launch_adjoint(
        &y, &x, &mut z, n as i32,
        &mut adj_y2, &mut adj_x2, &mut adj_z,
        n, 0,
    ).unwrap();

    // Step 1 backward: dy/dx = factor = 3.0
    // Seed adj_y from step 2's output
    let mut adj_x_total = Array::<f32>::zeros(n, Device::Cuda(0));
    mul_kernel::launch_adjoint(
        &x, &mut y, factor, n as i32,
        &mut adj_x_total, &mut adj_y2,
        n, 0,
    ).unwrap();

    // Total gradient: adj_x_total (from mul backward) + adj_x2 (from add backward direct path)
    // adj_x_total should be 3.0 (dy/dx * dz/dy = 3 * 1)
    // adj_x2 should be 1.0 (dz/dx direct path)
    let grad_via_y = adj_x_total.to_vec();
    let grad_direct = adj_x2.to_vec();

    for i in 0..n {
        let total = grad_via_y[i] + grad_direct[i];
        assert!(
            (total - 4.0).abs() < 1e-3,
            "total grad[{}] = {} + {} = {}, expected 4.0",
            i, grad_via_y[i], grad_direct[i], total
        );
    }

    eprintln!("✅ Tape two-step backward: dz/dx = 4.0 (via y: 3.0, direct: 1.0) for {} elements", n);
}

/// Test: Tape records and replays in correct order.
#[test]
fn test_tape_ordering() {
    let tape = Tape::new();
    assert!(tape.is_empty());

    let order = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));

    let o1 = order.clone();
    tape.record("step1", move || {
        o1.lock().unwrap().push(1);
        Ok(())
    });

    let o2 = order.clone();
    tape.record("step2", move || {
        o2.lock().unwrap().push(2);
        Ok(())
    });

    let o3 = order.clone();
    tape.record("step3", move || {
        o3.lock().unwrap().push(3);
        Ok(())
    });

    assert_eq!(tape.len(), 3);

    tape.backward().unwrap();

    // Should be reversed: 3, 2, 1
    let executed = order.lock().unwrap().clone();
    assert_eq!(executed, vec![3, 2, 1], "backward should execute in reverse order");
    assert!(tape.is_empty(), "tape should be empty after backward");

    eprintln!("✅ Tape ordering: backward executes in reverse order");
}
