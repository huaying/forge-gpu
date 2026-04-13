//! Advanced autodiff tests — custom adjoint, Jacobian computation.

use forge_macros::kernel;
use forge_runtime::{Array, Device, Tape};
use forge_runtime::jacobian::{compute_jacobian, compute_jacobian_fd, jacobian_row};

// ── Kernel: y = x^3 (via y = x * x * x) ─────────────────
// Auto adjoint gives dy/dx = 3x^2

#[kernel(autodiff)]
fn cube_kernel(input: &Array<f32>, output: &mut Array<f32>, n: i32) {
    let i = thread_id();
    if i < n {
        let x = input[i];
        let x2 = x * x;
        let x3 = x2 * x;
        output[i] = x3;
    }
}

// A hand-written adjoint kernel for y = x^3.
// dy/dx = 3 * x^2, so adj_input[i] += adj_output[i] * 3 * x^2
#[kernel]
fn cube_custom_adjoint(
    input: &Array<f32>,
    output: &Array<f32>,
    adj_output: &Array<f32>,
    adj_input: &mut Array<f32>,
    n: i32,
) {
    let i = thread_id();
    if i < n {
        let x = input[i];
        let grad = 3.0 * x * x;
        adj_input[i] = adj_output[i] * grad;
    }
}

// ── Test 1: Custom adjoint for y = x^3 matches auto-generated ────
#[test]
fn test_custom_adjoint_matches_auto() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let n = 64;
    let input_data: Vec<f32> = (1..=n).map(|i| i as f32 * 0.5).collect();
    let input = Array::from_vec(input_data.clone(), Device::Cuda(0));
    let mut output = Array::<f32>::zeros(n as usize, Device::Cuda(0));

    // Forward pass
    cube_kernel::launch(&input, &mut output, n, n as usize, 0)
        .expect("forward failed");

    // Auto-generated adjoint
    let mut adj_input_auto = Array::<f32>::zeros(n as usize, Device::Cuda(0));
    let mut adj_output_auto = Array::from_vec(vec![1.0f32; n as usize], Device::Cuda(0));
    cube_kernel::launch_adjoint(
        &input, &mut output, n,
        &mut adj_input_auto, &mut adj_output_auto,
        n as usize, 0,
    ).expect("auto adjoint failed");

    // Custom adjoint
    let mut adj_input_custom = Array::<f32>::zeros(n as usize, Device::Cuda(0));
    let adj_output_custom = Array::from_vec(vec![1.0f32; n as usize], Device::Cuda(0));
    cube_custom_adjoint::launch(
        &input, &output, &adj_output_custom, &mut adj_input_custom,
        n, n as usize, 0,
    ).expect("custom adjoint failed");

    let grad_auto = adj_input_auto.to_vec();
    let grad_custom = adj_input_custom.to_vec();

    for i in 0..n as usize {
        let x = input_data[i];
        let expected = 3.0 * x * x;
        assert!(
            (grad_auto[i] - expected).abs() < 1e-1,
            "auto grad[{}] = {}, expected {}",
            i, grad_auto[i], expected
        );
        assert!(
            (grad_custom[i] - expected).abs() < 1e-1,
            "custom grad[{}] = {}, expected {}",
            i, grad_custom[i], expected
        );
        assert!(
            (grad_auto[i] - grad_custom[i]).abs() < 1e-1,
            "auto vs custom grad[{}]: {} vs {}",
            i, grad_auto[i], grad_custom[i]
        );
    }

    eprintln!("✅ Custom adjoint for x^3 matches auto-generated for {} elements", n);
}

// ── Test 2: Tape with register_custom_backward ───────────
#[test]
fn test_tape_custom_backward() {
    let tape = Tape::new();

    let order = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));

    // Record an auto backward
    let o1 = order.clone();
    tape.record("step1", move || {
        o1.lock().unwrap().push("auto_step1");
        Ok(())
    });

    let o2 = order.clone();
    tape.record("step2", move || {
        o2.lock().unwrap().push("auto_step2");
        Ok(())
    });

    // Override step1 with a custom backward
    let o_custom = order.clone();
    tape.register_custom_backward("step1", move || {
        o_custom.lock().unwrap().push("custom_step1");
        Ok(())
    });

    assert_eq!(tape.len(), 2);
    assert!(tape.has_entry("step1"));
    assert!(tape.has_entry("step2"));

    tape.backward().unwrap();

    let executed: Vec<&str> = order.lock().unwrap().clone();
    // Reverse order: step2 then step1 (but step1 is now custom)
    assert_eq!(executed, vec!["auto_step2", "custom_step1"]);

    eprintln!("✅ Tape register_custom_backward: replaced entry correctly");
}

// ── Test 3: Tape register_custom_backward with no existing entry ──
#[test]
fn test_tape_custom_backward_no_existing() {
    let tape = Tape::new();

    let called = std::sync::Arc::new(std::sync::Mutex::new(false));
    let c = called.clone();
    tape.register_custom_backward("new_entry", move || {
        *c.lock().unwrap() = true;
        Ok(())
    });

    assert_eq!(tape.len(), 1);
    assert!(tape.has_entry("new_entry"));

    tape.backward().unwrap();
    assert!(*called.lock().unwrap());

    eprintln!("✅ Tape register_custom_backward: appended when no existing entry");
}

// ── Kernel for Jacobian tests: f(x) = [x0*x1, x0+x1, x0^2] ──
// Jacobian: [[x1, x0], [1, 1], [2*x0, 0]]

#[kernel(autodiff)]
fn multi_output_kernel(input: &Array<f32>, output: &mut Array<f32>, n: i32) {
    let i = thread_id();
    if i < n {
        // We use a single-threaded approach: thread 0 computes all outputs
        // Input has 2 elements, output has 3 elements
        if i == 0 {
            let x0 = input[0];
            let x1 = input[1];
            output[0] = x0 * x1;
        }
        if i == 1 {
            let x0 = input[0];
            let x1 = input[1];
            output[1] = x0 + x1;
        }
        if i == 2 {
            let x0 = input[0];
            let x3 = input[0];
            output[2] = x0 * x3;
        }
    }
}

// ── Test 4: Jacobian via reverse-mode ────────────────────
#[test]
fn test_jacobian_reverse_mode() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let device = Device::Cuda(0);

    // Input: x = [3.0, 5.0]
    let input = Array::from_vec(vec![3.0f32, 5.0], device);
    let output_dim = 3;
    let n = 3i32; // threads needed

    let jac = compute_jacobian(
        |inp, out| {
            multi_output_kernel::launch(inp, out, n, n as usize, 0)
        },
        |inp, out, adj_out, adj_inp| {
            multi_output_kernel::launch_adjoint(
                inp, out, n,
                adj_inp, adj_out,
                n as usize, 0,
            )
        },
        &input,
        output_dim,
        device,
    ).expect("jacobian computation failed");

    let jac_host = jac.to_vec();
    let input_dim = 2;

    // Expected Jacobian at x=[3,5]:
    // Row 0: ∂(x0*x1)/∂x = [x1, x0] = [5, 3]
    // Row 1: ∂(x0+x1)/∂x = [1, 1]
    // Row 2: ∂(x0^2)/∂x  = [2*x0, 0] = [6, 0]

    let row0 = jacobian_row(&jac_host, 0, input_dim);
    let row1 = jacobian_row(&jac_host, 1, input_dim);
    let row2 = jacobian_row(&jac_host, 2, input_dim);

    eprintln!("Jacobian row 0: {:?}", row0);
    eprintln!("Jacobian row 1: {:?}", row1);
    eprintln!("Jacobian row 2: {:?}", row2);

    assert!((row0[0] - 5.0).abs() < 0.5, "J[0][0] = {}, expected 5.0", row0[0]);
    assert!((row0[1] - 3.0).abs() < 0.5, "J[0][1] = {}, expected 3.0", row0[1]);
    assert!((row1[0] - 1.0).abs() < 0.5, "J[1][0] = {}, expected 1.0", row1[0]);
    assert!((row1[1] - 1.0).abs() < 0.5, "J[1][1] = {}, expected 1.0", row1[1]);
    assert!((row2[0] - 6.0).abs() < 0.5, "J[2][0] = {}, expected 6.0", row2[0]);
    assert!((row2[1] - 0.0).abs() < 0.5, "J[2][1] = {}, expected 0.0", row2[1]);

    eprintln!("✅ Jacobian (reverse-mode) correct for f(x) = [x0*x1, x0+x1, x0^2]");
}

// ── Test 5: Jacobian via finite differences ──────────────
#[test]
fn test_jacobian_finite_differences() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let device = Device::Cuda(0);
    let input = Array::from_vec(vec![3.0f32, 5.0], device);
    let output_dim = 3;
    let n = 3i32;
    let eps = 1e-3f32;

    let jac = compute_jacobian_fd(
        |inp, out| {
            multi_output_kernel::launch(inp, out, n, n as usize, 0)
        },
        &input,
        output_dim,
        eps,
        device,
    ).expect("finite difference jacobian failed");

    let jac_host = jac.to_vec();
    let input_dim = 2;

    let row0 = jacobian_row(&jac_host, 0, input_dim);
    let row1 = jacobian_row(&jac_host, 1, input_dim);
    let row2 = jacobian_row(&jac_host, 2, input_dim);

    eprintln!("FD Jacobian row 0: {:?}", row0);
    eprintln!("FD Jacobian row 1: {:?}", row1);
    eprintln!("FD Jacobian row 2: {:?}", row2);

    // Central differences should be accurate to O(eps^2)
    assert!((row0[0] - 5.0).abs() < 0.1, "J_fd[0][0] = {}, expected 5.0", row0[0]);
    assert!((row0[1] - 3.0).abs() < 0.1, "J_fd[0][1] = {}, expected 3.0", row0[1]);
    assert!((row1[0] - 1.0).abs() < 0.1, "J_fd[1][0] = {}, expected 1.0", row1[0]);
    assert!((row1[1] - 1.0).abs() < 0.1, "J_fd[1][1] = {}, expected 1.0", row1[1]);
    assert!((row2[0] - 6.0).abs() < 0.1, "J_fd[2][0] = {}, expected 6.0", row2[0]);
    assert!((row2[1] - 0.0).abs() < 0.1, "J_fd[2][1] = {}, expected 0.0", row2[1]);

    eprintln!("✅ Jacobian (finite differences) correct for f(x) = [x0*x1, x0+x1, x0^2]");
}

// ── Test 6: Jacobian FD matches reverse-mode ─────────────
#[test]
fn test_jacobian_fd_matches_reverse() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let device = Device::Cuda(0);
    let input = Array::from_vec(vec![2.0f32, 7.0], device);
    let output_dim = 3;
    let n = 3i32;

    // Reverse-mode Jacobian
    let jac_rev = compute_jacobian(
        |inp, out| multi_output_kernel::launch(inp, out, n, n as usize, 0),
        |inp, out, adj_out, adj_inp| {
            multi_output_kernel::launch_adjoint(
                inp, out, n,
                adj_inp, adj_out,
                n as usize, 0,
            )
        },
        &input,
        output_dim,
        device,
    ).expect("reverse jacobian failed");

    // Finite-difference Jacobian
    let jac_fd = compute_jacobian_fd(
        |inp, out| multi_output_kernel::launch(inp, out, n, n as usize, 0),
        &input,
        output_dim,
        1e-3,
        device,
    ).expect("fd jacobian failed");

    let rev = jac_rev.to_vec();
    let fd = jac_fd.to_vec();

    for i in 0..rev.len() {
        assert!(
            (rev[i] - fd[i]).abs() < 1.0,
            "Mismatch at {}: reverse={}, fd={}",
            i, rev[i], fd[i]
        );
    }

    eprintln!("✅ Jacobian: reverse-mode and finite-difference results match");
}

// ── Simple Jacobian kernel: y = 2*x (linear) ────────────
// Jacobian should be 2*I (diagonal matrix)

#[kernel(autodiff)]
fn double_kernel(input: &Array<f32>, output: &mut Array<f32>, n: i32) {
    let i = thread_id();
    if i < n {
        let x = input[i];
        output[i] = x + x;
    }
}

#[test]
fn test_jacobian_diagonal() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let device = Device::Cuda(0);
    let n = 4;
    let input = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], device);

    let jac = compute_jacobian_fd(
        |inp, out| double_kernel::launch(inp, out, n as i32, n, 0),
        &input,
        n,
        1e-3,
        device,
    ).expect("jacobian failed");

    let j = jac.to_vec();
    eprintln!("Diagonal Jacobian: {:?}", j);

    // Should be 2*I: J[i][j] = 2 if i==j, 0 otherwise
    for i in 0..n {
        for jj in 0..n {
            let expected = if i == jj { 2.0 } else { 0.0 };
            assert!(
                (j[i * n + jj] - expected).abs() < 0.1,
                "J[{}][{}] = {}, expected {}",
                i, jj, j[i * n + jj], expected
            );
        }
    }

    eprintln!("✅ Jacobian of 2*x = 2*I (diagonal) correct for {} elements", n);
}
