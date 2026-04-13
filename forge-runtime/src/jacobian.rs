//! # Jacobian computation utilities.
//!
//! Compute Jacobian matrices via reverse-mode autodiff (adjoint) or forward
//! finite differences.
//!
//! ## Reverse-mode Jacobian
//!
//! For a function `f: ℝⁿ → ℝᵐ`, the Jacobian `J[i][j] = ∂f_i/∂x_j` can be
//! computed with `m` backward (adjoint) passes, each seeded with a one-hot
//! vector for output dimension `i`.
//!
//! ## Finite-difference Jacobian
//!
//! A simpler alternative that requires `2n` forward passes using central
//! differences: `J[i][j] ≈ (f_i(x + ε·eⱼ) − f_i(x − ε·eⱼ)) / 2ε`.

use crate::{Array, Device, ForgeError};

/// Compute the Jacobian matrix `J[i][j] = ∂output_i / ∂input_j` using
/// repeated reverse-mode (adjoint) passes.
///
/// For `m` outputs and `n` inputs this performs `m` backward passes.
///
/// Returns a flat `Array<f32>` of size `m × n` in **row-major** order:
/// element `(i, j)` is at index `i * n + j`.
///
/// # Arguments
///
/// * `forward_fn` — runs the forward kernel, filling `output` from `input`.
/// * `adjoint_fn` — runs the adjoint kernel:
///   `(input, output, adj_output, adj_input)` where `adj_output` is the seed
///   and `adj_input` receives the gradient. Both `output` and `adj_output` are
///   passed as `&mut` because many generated adjoint kernels require mutable
///   access to all array parameters.
/// * `input`      — the point at which to evaluate the Jacobian.
/// * `output_dim` — number of output elements (`m`).
/// * `device`     — GPU device to use.
pub fn compute_jacobian<F, A>(
    forward_fn: F,
    adjoint_fn: A,
    input: &Array<f32>,
    output_dim: usize,
    device: Device,
) -> Result<Array<f32>, ForgeError>
where
    F: Fn(&Array<f32>, &mut Array<f32>) -> Result<(), ForgeError>,
    A: Fn(&Array<f32>, &mut Array<f32>, &mut Array<f32>, &mut Array<f32>) -> Result<(), ForgeError>,
{
    let input_dim = input.len();
    let total = output_dim * input_dim;
    let mut jacobian_host = vec![0.0f32; total];

    for i in 0..output_dim {
        // Seed: one-hot vector for output dimension i
        let mut seed_host = vec![0.0f32; output_dim];
        seed_host[i] = 1.0;
        let mut adj_output = Array::from_vec(seed_host, device);

        // Recompute forward (adjoint kernels typically need intermediate values)
        let mut output = Array::<f32>::zeros(output_dim, device);
        forward_fn(input, &mut output)?;

        // Run adjoint pass
        let mut adj_input = Array::<f32>::zeros(input_dim, device);
        adjoint_fn(input, &mut output, &mut adj_output, &mut adj_input)?;

        // adj_input is now the i-th row of the Jacobian
        let row = adj_input.to_vec();
        jacobian_host[i * input_dim..(i + 1) * input_dim].copy_from_slice(&row);
    }

    Ok(Array::from_vec(jacobian_host, device))
}

/// Compute the Jacobian matrix using central finite differences.
///
/// `J[i][j] ≈ (f_i(x + ε·eⱼ) − f_i(x − ε·eⱼ)) / 2ε`
///
/// This requires `2n` forward evaluations and is general-purpose (works even
/// when no adjoint kernel exists), but is less accurate and slower for large
/// input dimensions.
///
/// Returns a flat `Array<f32>` of size `m × n` in row-major order.
pub fn compute_jacobian_fd<F>(
    forward_fn: F,
    input: &Array<f32>,
    output_dim: usize,
    eps: f32,
    device: Device,
) -> Result<Array<f32>, ForgeError>
where
    F: Fn(&Array<f32>, &mut Array<f32>) -> Result<(), ForgeError>,
{
    let input_dim = input.len();
    let total = output_dim * input_dim;
    let mut jacobian_host = vec![0.0f32; total];
    let input_host = input.to_vec();

    for j in 0..input_dim {
        // x + eps * e_j
        let mut plus = input_host.clone();
        plus[j] += eps;
        let x_plus = Array::from_vec(plus, device);
        let mut f_plus = Array::<f32>::zeros(output_dim, device);
        forward_fn(&x_plus, &mut f_plus)?;
        let fp = f_plus.to_vec();

        // x - eps * e_j
        let mut minus = input_host.clone();
        minus[j] -= eps;
        let x_minus = Array::from_vec(minus, device);
        let mut f_minus = Array::<f32>::zeros(output_dim, device);
        forward_fn(&x_minus, &mut f_minus)?;
        let fm = f_minus.to_vec();

        // J[i][j] = (f_plus[i] - f_minus[i]) / (2 * eps)
        let inv_2eps = 1.0 / (2.0 * eps);
        for i in 0..output_dim {
            jacobian_host[i * input_dim + j] = (fp[i] - fm[i]) * inv_2eps;
        }
    }

    Ok(Array::from_vec(jacobian_host, device))
}

/// Helper: extract a sub-matrix row from a flat Jacobian.
///
/// Given a flat row-major Jacobian of size `rows × cols`, returns the `i`-th
/// row as a `Vec<f32>` of length `cols`.
pub fn jacobian_row(jacobian: &[f32], row: usize, cols: usize) -> Vec<f32> {
    let start = row * cols;
    jacobian[start..start + cols].to_vec()
}
