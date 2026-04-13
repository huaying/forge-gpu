//! # Conjugate Gradient solver for sparse Ax = b on GPU.
//!
//! Uses CsrMatrix for A, Array<f32> for x and b.
//! GPU-accelerated via custom CUDA kernels for vector ops.

use crate::{Array, CsrMatrix, Device, ForgeError};

/// Result of a conjugate gradient solve.
#[derive(Debug, Clone)]
pub struct CgResult {
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final residual norm (L2).
    pub residual: f32,
    /// Whether the solver converged within tolerance.
    pub converged: bool,
}

/// Conjugate Gradient solver for sparse symmetric positive-definite systems.
///
/// Solves Ax = b where A is a sparse SPD matrix in CSR format.
///
/// # Arguments
/// * `a` — Sparse matrix (CSR format, must be SPD)
/// * `b` — Right-hand side vector
/// * `x` — Initial guess (modified in place to hold solution)
/// * `max_iters` — Maximum number of CG iterations
/// * `tol` — Convergence tolerance on residual norm
///
/// # Returns
/// A `CgResult` with iteration count, final residual, and convergence flag.
pub fn cg_solve(
    a: &CsrMatrix,
    b: &Array<f32>,
    x: &mut Array<f32>,
    max_iters: usize,
    tol: f32,
) -> Result<CgResult, ForgeError> {
    let device = x.device();
    match device {
        Device::Cpu => cg_solve_cpu(a, b, x, max_iters, tol),
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cg_solve_gpu(a, b, x, max_iters, tol),
        #[cfg(not(feature = "cuda"))]
        Device::Cuda(_) => Err(ForgeError::CompilationFailed("CUDA not enabled".to_string())),
    }
}

/// CPU implementation of CG solver.
fn cg_solve_cpu(
    a: &CsrMatrix,
    b: &Array<f32>,
    x: &mut Array<f32>,
    max_iters: usize,
    tol: f32,
) -> Result<CgResult, ForgeError> {
    let n = a.rows;
    assert_eq!(a.cols, n, "A must be square");
    assert_eq!(b.len(), n, "b length must match A rows");
    assert_eq!(x.len(), n, "x length must match A cols");

    let x_vec = x.to_vec();
    let b_vec = b.to_vec();

    // r = b - A*x
    let ax = a.spmv(&x_vec);
    let mut r: Vec<f32> = b_vec.iter().zip(ax.iter()).map(|(&bi, &ai)| bi - ai).collect();
    let mut p = r.clone();
    let mut rsold: f32 = r.iter().map(|&ri| ri * ri).sum();

    let mut x_buf = x_vec;

    for i in 0..max_iters {
        if rsold.sqrt() < tol {
            // Copy result back
            for (j, &val) in x_buf.iter().enumerate() {
                x[j] = val;
            }
            return Ok(CgResult {
                iterations: i,
                residual: rsold.sqrt(),
                converged: true,
            });
        }

        let ap = a.spmv(&p);
        let pap: f32 = p.iter().zip(ap.iter()).map(|(&pi, &api)| pi * api).sum();
        let alpha = rsold / pap;

        for j in 0..n {
            x_buf[j] += alpha * p[j];
            r[j] -= alpha * ap[j];
        }

        let rsnew: f32 = r.iter().map(|&ri| ri * ri).sum();
        if rsnew.sqrt() < tol {
            for (j, &val) in x_buf.iter().enumerate() {
                x[j] = val;
            }
            return Ok(CgResult {
                iterations: i + 1,
                residual: rsnew.sqrt(),
                converged: true,
            });
        }

        let beta = rsnew / rsold;
        for j in 0..n {
            p[j] = r[j] + beta * p[j];
        }
        rsold = rsnew;
    }

    for (j, &val) in x_buf.iter().enumerate() {
        x[j] = val;
    }
    Ok(CgResult {
        iterations: max_iters,
        residual: rsold.sqrt(),
        converged: false,
    })
}

/// GPU implementation of CG solver.
#[cfg(feature = "cuda")]
fn cg_solve_gpu(
    a: &CsrMatrix,
    b: &Array<f32>,
    x: &mut Array<f32>,
    max_iters: usize,
    tol: f32,
) -> Result<CgResult, ForgeError> {
    use crate::cuda;

    let n = a.rows;
    let device = x.device();
    let ordinal = match device {
        Device::Cuda(i) => i,
        _ => 0,
    };

    assert_eq!(a.cols, n, "A must be square");
    assert_eq!(b.len(), n, "b length must match A rows");
    assert_eq!(x.len(), n, "x length must match A cols");

    // Compile individual kernels
    let dot_kernel = crate::CompiledKernel::compile(CG_DOT_SRC, "cg_dot")?;
    let axpy_kernel = crate::CompiledKernel::compile(CG_AXPY_SRC, "cg_axpy")?;
    let copy_kernel = crate::CompiledKernel::compile(CG_COPY_SRC, "cg_copy")?;
    let xpay_kernel = crate::CompiledKernel::compile(CG_XPAY_SRC, "cg_xpay")?;
    let sub_kernel = crate::CompiledKernel::compile(CG_SUB_SRC, "cg_sub")?;

    let dot_func = dot_kernel.get_function(ordinal)?;
    let axpy_func = axpy_kernel.get_function(ordinal)?;
    let copy_func = copy_kernel.get_function(ordinal)?;
    let xpay_func = xpay_kernel.get_function(ordinal)?;
    let sub_func = sub_kernel.get_function(ordinal)?;

    let stream = cuda::default_stream(ordinal);
    let config = cuda::LaunchConfig::for_num_elems(n as u32);
    let n_i32 = n as i32;

    // Allocate temporary vectors on GPU
    let mut r = Array::<f32>::zeros(n, device);
    let mut p = Array::<f32>::zeros(n, device);
    let mut ap = Array::<f32>::zeros(n, device);
    let mut dot_result = Array::<f32>::zeros(1, device);

    // Helper: GPU dot product → returns scalar on host
    let gpu_dot = |a_arr: &Array<f32>, b_arr: &Array<f32>, result: &mut Array<f32>|
        -> Result<f32, ForgeError>
    {
        // Zero the result
        let zero_arr = Array::from_vec(vec![0.0f32], device);
        unsafe {
            use cuda::PushKernelArg;
            let mut builder = stream.launch_builder(&copy_func);
            builder.arg(zero_arr.cuda_slice().unwrap());
            builder.arg(result.cuda_slice_mut().unwrap());
            let one_i32 = 1i32;
            builder.arg(&one_i32);
            builder.launch(cuda::LaunchConfig::for_num_elems(1))
                .map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
        }

        // Dot product with block reduction + atomicAdd
        unsafe {
            use cuda::PushKernelArg;
            let mut builder = stream.launch_builder(&dot_func);
            builder.arg(a_arr.cuda_slice().unwrap());
            builder.arg(b_arr.cuda_slice().unwrap());
            builder.arg(result.cuda_slice_mut().unwrap());
            builder.arg(&n_i32);
            builder.launch(config)
                .map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
        }

        stream.synchronize()
            .map_err(|e| ForgeError::SyncFailed(format!("{:?}", e)))?;
        let val = result.to_vec();
        Ok(val[0])
    };

    // r = b - A*x  (first compute r = A*x, then r = b - r)
    a.spmv_gpu(x, &mut r, device)?;
    unsafe {
        use cuda::PushKernelArg;
        let mut builder = stream.launch_builder(&sub_func);
        builder.arg(b.cuda_slice().unwrap());
        builder.arg(r.cuda_slice_mut().unwrap());
        builder.arg(&n_i32);
        builder.launch(config)
            .map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
    }
    stream.synchronize()
        .map_err(|e| ForgeError::SyncFailed(format!("{:?}", e)))?;

    // p = r
    unsafe {
        use cuda::PushKernelArg;
        let mut builder = stream.launch_builder(&copy_func);
        builder.arg(r.cuda_slice().unwrap());
        builder.arg(p.cuda_slice_mut().unwrap());
        builder.arg(&n_i32);
        builder.launch(config)
            .map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
    }
    stream.synchronize()
        .map_err(|e| ForgeError::SyncFailed(format!("{:?}", e)))?;

    let mut rsold = gpu_dot(&r, &r, &mut dot_result)?;

    for i in 0..max_iters {
        if rsold.sqrt() < tol {
            return Ok(CgResult {
                iterations: i,
                residual: rsold.sqrt(),
                converged: true,
            });
        }

        // ap = A * p (zero ap first, then spmv)
        let zero_ap = Array::<f32>::zeros(n, device);
        unsafe {
            use cuda::PushKernelArg;
            let mut builder = stream.launch_builder(&copy_func);
            builder.arg(zero_ap.cuda_slice().unwrap());
            builder.arg(ap.cuda_slice_mut().unwrap());
            builder.arg(&n_i32);
            builder.launch(config)
                .map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
        }
        stream.synchronize()
            .map_err(|e| ForgeError::SyncFailed(format!("{:?}", e)))?;

        a.spmv_gpu(&p, &mut ap, device)?;

        // alpha = rsold / dot(p, ap)
        let pap = gpu_dot(&p, &ap, &mut dot_result)?;
        let alpha = rsold / pap;

        // x = x + alpha * p
        unsafe {
            use cuda::PushKernelArg;
            let mut builder = stream.launch_builder(&axpy_func);
            builder.arg(&alpha);
            builder.arg(p.cuda_slice().unwrap());
            builder.arg(x.cuda_slice_mut().unwrap());
            builder.arg(&n_i32);
            builder.launch(config)
                .map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
        }

        // r = r - alpha * ap
        let neg_alpha = -alpha;
        unsafe {
            use cuda::PushKernelArg;
            let mut builder = stream.launch_builder(&axpy_func);
            builder.arg(&neg_alpha);
            builder.arg(ap.cuda_slice().unwrap());
            builder.arg(r.cuda_slice_mut().unwrap());
            builder.arg(&n_i32);
            builder.launch(config)
                .map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
        }
        stream.synchronize()
            .map_err(|e| ForgeError::SyncFailed(format!("{:?}", e)))?;

        // rsnew = dot(r, r)
        let rsnew = gpu_dot(&r, &r, &mut dot_result)?;

        if rsnew.sqrt() < tol {
            return Ok(CgResult {
                iterations: i + 1,
                residual: rsnew.sqrt(),
                converged: true,
            });
        }

        // p = r + (rsnew/rsold) * p
        let beta = rsnew / rsold;
        unsafe {
            use cuda::PushKernelArg;
            let mut builder = stream.launch_builder(&xpay_func);
            builder.arg(r.cuda_slice().unwrap());
            builder.arg(&beta);
            builder.arg(p.cuda_slice_mut().unwrap());
            builder.arg(&n_i32);
            builder.launch(config)
                .map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
        }
        stream.synchronize()
            .map_err(|e| ForgeError::SyncFailed(format!("{:?}", e)))?;

        rsold = rsnew;
    }

    Ok(CgResult {
        iterations: max_iters,
        residual: rsold.sqrt(),
        converged: false,
    })
}

// ── Individual CUDA kernel sources for CG ──

#[cfg(feature = "cuda")]
const CG_DOT_SRC: &str = r#"
extern "C" __global__ void cg_dot(
    const float* a, const float* b, float* result, int n
) {
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int idx = i; idx < n; idx += blockDim.x * gridDim.x) {
        sum += a[idx] * b[idx];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(result, sdata[0]);
}
"#;

#[cfg(feature = "cuda")]
const CG_AXPY_SRC: &str = r#"
extern "C" __global__ void cg_axpy(
    float a, const float* x, float* y, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}
"#;

#[cfg(feature = "cuda")]
const CG_COPY_SRC: &str = r#"
extern "C" __global__ void cg_copy(
    const float* src, float* dst, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
"#;

#[cfg(feature = "cuda")]
const CG_XPAY_SRC: &str = r#"
extern "C" __global__ void cg_xpay(
    const float* x, float beta, float* p, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = x[i] + beta * p[i];
}
"#;

#[cfg(feature = "cuda")]
const CG_SUB_SRC: &str = r#"
extern "C" __global__ void cg_sub(
    const float* a, float* r, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) r[i] = a[i] - r[i];
}
"#;
