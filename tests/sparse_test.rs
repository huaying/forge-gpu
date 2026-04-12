//! Tests for CsrMatrix sparse operations.

use forge_runtime::{CsrMatrix, Array, Device};

#[test]
fn test_csr_from_triplets() {
    // 3x3 matrix:
    // [1 0 2]
    // [0 3 0]
    // [4 0 5]
    let mat = CsrMatrix::from_triplets(3, 3, &[
        (0, 0, 1.0), (0, 2, 2.0),
        (1, 1, 3.0),
        (2, 0, 4.0), (2, 2, 5.0),
    ]);

    assert_eq!(mat.rows, 3);
    assert_eq!(mat.cols, 3);
    assert_eq!(mat.nnz, 5);
    assert!((mat.get(0, 0) - 1.0).abs() < 1e-6);
    assert!((mat.get(0, 1) - 0.0).abs() < 1e-6);
    assert!((mat.get(0, 2) - 2.0).abs() < 1e-6);
    assert!((mat.get(1, 1) - 3.0).abs() < 1e-6);
    assert!((mat.get(2, 0) - 4.0).abs() < 1e-6);
    assert!((mat.get(2, 2) - 5.0).abs() < 1e-6);

    eprintln!("✅ CSR from triplets: 3x3, nnz={}", mat.nnz);
}

#[test]
fn test_csr_duplicate_triplets() {
    // Duplicates should be summed
    let mat = CsrMatrix::from_triplets(2, 2, &[
        (0, 0, 1.0), (0, 0, 2.0), (1, 1, 5.0),
    ]);
    assert!((mat.get(0, 0) - 3.0).abs() < 1e-6, "duplicates should sum");
    assert_eq!(mat.nnz, 2);

    eprintln!("✅ CSR duplicate triplets: summed correctly");
}

#[test]
fn test_csr_spmv_cpu() {
    let mat = CsrMatrix::from_triplets(3, 3, &[
        (0, 0, 1.0), (0, 2, 2.0),
        (1, 1, 3.0),
        (2, 0, 4.0), (2, 2, 5.0),
    ]);

    let x = vec![1.0, 2.0, 3.0];
    let y = mat.spmv(&x);

    // [1 0 2] [1]   [1*1 + 2*3] = [7]
    // [0 3 0] [2] = [3*2]       = [6]
    // [4 0 5] [3]   [4*1 + 5*3] = [19]
    assert!((y[0] - 7.0).abs() < 1e-6);
    assert!((y[1] - 6.0).abs() < 1e-6);
    assert!((y[2] - 19.0).abs() < 1e-6);

    eprintln!("✅ CSR SpMV CPU: y = [{}, {}, {}]", y[0], y[1], y[2]);
}

#[test]
fn test_csr_spmv_gpu() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    let mat = CsrMatrix::from_triplets(3, 3, &[
        (0, 0, 1.0), (0, 2, 2.0),
        (1, 1, 3.0),
        (2, 0, 4.0), (2, 2, 5.0),
    ]);

    let x = Array::from_vec(vec![1.0f32, 2.0, 3.0], Device::Cuda(0));
    let mut y = Array::<f32>::zeros(3, Device::Cuda(0));

    mat.spmv_gpu(&x, &mut y, Device::Cuda(0)).expect("GPU SpMV failed");

    let result = y.to_vec();
    assert!((result[0] - 7.0).abs() < 1e-4);
    assert!((result[1] - 6.0).abs() < 1e-4);
    assert!((result[2] - 19.0).abs() < 1e-4);

    eprintln!("✅ CSR SpMV GPU: y = [{}, {}, {}]", result[0], result[1], result[2]);
}

#[test]
fn test_csr_identity() {
    let mat = CsrMatrix::identity(4);
    assert_eq!(mat.nnz, 4);

    let x = vec![2.0, 3.0, 5.0, 7.0];
    let y = mat.spmv(&x);
    assert_eq!(y, x, "identity * x should equal x");

    eprintln!("✅ CSR identity: I * x = x");
}

#[test]
fn test_csr_transpose() {
    let mat = CsrMatrix::from_triplets(2, 3, &[
        (0, 0, 1.0), (0, 2, 2.0),
        (1, 1, 3.0),
    ]);

    let mt = mat.transpose();
    assert_eq!(mt.rows, 3);
    assert_eq!(mt.cols, 2);
    assert!((mt.get(0, 0) - 1.0).abs() < 1e-6);
    assert!((mt.get(2, 0) - 2.0).abs() < 1e-6);
    assert!((mt.get(1, 1) - 3.0).abs() < 1e-6);

    eprintln!("✅ CSR transpose: correct");
}

#[test]
fn test_csr_large_spmv() {
    forge_runtime::cuda::init();
    if forge_runtime::cuda::device_count() == 0 {
        eprintln!("Skipping: no GPU");
        return;
    }

    // 1000x1000 tridiagonal matrix
    let n = 1000;
    let mut triplets = Vec::new();
    for i in 0..n {
        triplets.push((i as u32, i as u32, 2.0f32)); // diagonal
        if i > 0 { triplets.push((i as u32, (i-1) as u32, -1.0)); } // sub-diagonal
        if i < n-1 { triplets.push((i as u32, (i+1) as u32, -1.0)); } // super-diagonal
    }

    let mat = CsrMatrix::from_triplets(n, n, &triplets);
    assert_eq!(mat.rows, n);

    // x = [1, 1, 1, ...]
    let x_data = vec![1.0f32; n];
    let x = Array::from_vec(x_data.clone(), Device::Cuda(0));
    let mut y = Array::<f32>::zeros(n, Device::Cuda(0));

    mat.spmv_gpu(&x, &mut y, Device::Cuda(0)).expect("GPU SpMV failed");

    let result = y.to_vec();
    // For tridiagonal with all 1s: first/last rows = 1, middle rows = 0
    assert!((result[0] - 1.0).abs() < 1e-4, "y[0] = {}", result[0]);
    assert!((result[n-1] - 1.0).abs() < 1e-4, "y[n-1] = {}", result[n-1]);
    for i in 1..n-1 {
        assert!((result[i]).abs() < 1e-4, "y[{}] = {}, expected 0", i, result[i]);
    }

    // Compare with CPU
    let y_cpu = mat.spmv(&x_data);
    for i in 0..n {
        assert!((result[i] - y_cpu[i]).abs() < 1e-3,
            "GPU vs CPU mismatch at {}: {} vs {}", i, result[i], y_cpu[i]);
    }

    eprintln!("✅ CSR large SpMV: 1000x1000 tridiagonal, GPU matches CPU");
}
