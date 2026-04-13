//! Tests for the Conjugate Gradient solver.

use forge_runtime::{cg_solve, CsrMatrix, Array, Device};

#[test]
fn test_cg_identity_cpu() {
    // Solve I*x = b → x should equal b
    let n = 10;
    let a = CsrMatrix::identity(n);
    let b_data: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
    let b = Array::from_vec(b_data.clone(), Device::Cpu);
    let mut x = Array::<f32>::zeros(n, Device::Cpu);

    let result = cg_solve(&a, &b, &mut x, 100, 1e-6).unwrap();
    assert!(result.converged, "CG did not converge for identity system");
    assert!(result.iterations <= 2, "Identity should converge in ≤2 iters, got {}", result.iterations);

    for i in 0..n {
        assert!(
            (x[i] - b_data[i]).abs() < 1e-4,
            "x[{}] = {}, expected {}", i, x[i], b_data[i]
        );
    }

    eprintln!("✅ CG identity CPU: converged in {} iters, residual={:.2e}",
        result.iterations, result.residual);
}

#[test]
fn test_cg_diagonal_cpu() {
    // Solve diag(2,3,4,...) * x = b
    let n = 5;
    let diag: Vec<f32> = (0..n).map(|i| (i + 2) as f32).collect();
    let a = CsrMatrix::diagonal(n, &diag);
    let b_data = vec![4.0, 9.0, 16.0, 25.0, 36.0]; // diag[i] * expected[i]
    let expected: Vec<f32> = b_data.iter().zip(diag.iter()).map(|(&bi, &di)| bi / di).collect();

    let b = Array::from_vec(b_data, Device::Cpu);
    let mut x = Array::<f32>::zeros(n, Device::Cpu);

    let result = cg_solve(&a, &b, &mut x, 100, 1e-6).unwrap();
    assert!(result.converged, "CG did not converge for diagonal system");

    for i in 0..n {
        assert!(
            (x[i] - expected[i]).abs() < 1e-3,
            "x[{}] = {}, expected {}", i, x[i], expected[i]
        );
    }

    eprintln!("✅ CG diagonal CPU: converged in {} iters", result.iterations);
}

#[test]
fn test_cg_tridiagonal_cpu() {
    // Tridiagonal SPD matrix: 2 on diagonal, -1 on sub/super-diagonals
    let n = 50;
    let mut triplets = Vec::new();
    for i in 0..n {
        triplets.push((i as u32, i as u32, 2.0f32));
        if i > 0 { triplets.push((i as u32, (i - 1) as u32, -1.0)); }
        if i < n - 1 { triplets.push((i as u32, (i + 1) as u32, -1.0)); }
    }
    let a = CsrMatrix::from_triplets(n, n, &triplets);

    // b = [1, 0, 0, ..., 0, 1]
    let mut b_data = vec![0.0f32; n];
    b_data[0] = 1.0;
    b_data[n - 1] = 1.0;

    let b = Array::from_vec(b_data, Device::Cpu);
    let mut x = Array::<f32>::zeros(n, Device::Cpu);

    let result = cg_solve(&a, &b, &mut x, 200, 1e-6).unwrap();
    assert!(result.converged, "CG did not converge for tridiagonal system (residual={:.2e})", result.residual);

    // Verify: Ax should equal b
    let x_vec = x.to_vec();
    let ax = a.spmv(&x_vec);
    let b_orig = b.to_vec();
    for i in 0..n {
        assert!(
            (ax[i] - b_orig[i]).abs() < 1e-3,
            "Ax[{}] = {}, b[{}] = {}", i, ax[i], i, b_orig[i]
        );
    }

    eprintln!("✅ CG tridiagonal CPU: {} iters, residual={:.2e}", result.iterations, result.residual);
}

#[test]
fn test_cg_identity_gpu() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let n = 10;
    let device = Device::Cuda(0);
    let a = CsrMatrix::identity(n);
    let b_data: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
    let b = Array::from_vec(b_data.clone(), device);
    let mut x = Array::<f32>::zeros(n, device);

    let result = cg_solve(&a, &b, &mut x, 100, 1e-6).unwrap();
    assert!(result.converged, "GPU CG did not converge for identity system");

    let x_vec = x.to_vec();
    for i in 0..n {
        assert!(
            (x_vec[i] - b_data[i]).abs() < 1e-3,
            "x[{}] = {}, expected {}", i, x_vec[i], b_data[i]
        );
    }

    eprintln!("✅ CG identity GPU: converged in {} iters, residual={:.2e}",
        result.iterations, result.residual);
}

#[test]
fn test_cg_tridiagonal_gpu() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let n = 100;
    let device = Device::Cuda(0);

    let mut triplets = Vec::new();
    for i in 0..n {
        triplets.push((i as u32, i as u32, 2.0f32));
        if i > 0 { triplets.push((i as u32, (i - 1) as u32, -1.0)); }
        if i < n - 1 { triplets.push((i as u32, (i + 1) as u32, -1.0)); }
    }
    let a = CsrMatrix::from_triplets(n, n, &triplets);

    let mut b_data = vec![0.0f32; n];
    b_data[0] = 1.0;
    b_data[n - 1] = 1.0;

    let b = Array::from_vec(b_data.clone(), device);
    let mut x = Array::<f32>::zeros(n, device);

    let result = cg_solve(&a, &b, &mut x, 500, 1e-4).unwrap();
    assert!(result.converged, "GPU CG did not converge for tridiagonal (residual={:.2e})", result.residual);

    // Verify: Ax ≈ b (on CPU)
    let x_vec = x.to_vec();
    let ax = a.spmv(&x_vec);
    for i in 0..n {
        assert!(
            (ax[i] - b_data[i]).abs() < 1e-2,
            "Ax[{}] = {}, b[{}] = {}", i, ax[i], i, b_data[i]
        );
    }

    eprintln!("✅ CG tridiagonal GPU: {} iters, residual={:.2e}", result.iterations, result.residual);
}

#[test]
fn test_cg_gpu_vs_cpu() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let n = 20;

    // SPD matrix: diagonal dominant
    let mut triplets = Vec::new();
    for i in 0..n {
        triplets.push((i as u32, i as u32, 4.0f32));
        if i > 0 { triplets.push((i as u32, (i - 1) as u32, -1.0)); }
        if i < n - 1 { triplets.push((i as u32, (i + 1) as u32, -1.0)); }
    }
    let a = CsrMatrix::from_triplets(n, n, &triplets);

    let b_data: Vec<f32> = (0..n).map(|i| ((i + 1) as f32) * 0.5).collect();

    // CPU solve
    let b_cpu = Array::from_vec(b_data.clone(), Device::Cpu);
    let mut x_cpu = Array::<f32>::zeros(n, Device::Cpu);
    let res_cpu = cg_solve(&a, &b_cpu, &mut x_cpu, 200, 1e-6).unwrap();
    assert!(res_cpu.converged);

    // GPU solve
    let b_gpu = Array::from_vec(b_data, Device::Cuda(0));
    let mut x_gpu = Array::<f32>::zeros(n, Device::Cuda(0));
    let res_gpu = cg_solve(&a, &b_gpu, &mut x_gpu, 200, 1e-6).unwrap();
    assert!(res_gpu.converged);

    let x_cpu_vec = x_cpu.to_vec();
    let x_gpu_vec = x_gpu.to_vec();

    for i in 0..n {
        assert!(
            (x_cpu_vec[i] - x_gpu_vec[i]).abs() < 1e-2,
            "CPU vs GPU mismatch at {}: {} vs {}", i, x_cpu_vec[i], x_gpu_vec[i]
        );
    }

    eprintln!("✅ CG GPU vs CPU: solutions match (CPU {} iters, GPU {} iters)",
        res_cpu.iterations, res_gpu.iterations);
}
