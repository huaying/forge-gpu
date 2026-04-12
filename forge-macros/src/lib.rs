//! # forge-macros
//!
//! Proc macros for Forge GPU compute framework.
//!
//! Provides:
//! - `#[kernel]` — transforms a Rust function into a GPU kernel with host-side launch wrapper
//! - `#[func]` (future) — marks device-callable functions

extern crate proc_macro;

mod cuda_emit;
mod func;
mod kernel;

use proc_macro::TokenStream;

/// Mark a function as a GPU kernel.
///
/// The `#[kernel]` macro transforms a Rust function into:
/// 1. A CUDA C++ kernel string (generated at compile time)
/// 2. A host-side module with a `launch()` function that:
///    - Compiles the CUDA source via nvrtc (cached after first call)
///    - Transfers arguments to GPU
///    - Launches the kernel with proper grid/block configuration
///
/// # Example
///
/// ```rust,ignore
/// use forge_macros::kernel;
///
/// #[kernel]
/// fn add_one(data: &mut Array<f32>, n: i32) {
///     let i = thread_id();
///     if i < n {
///         data[i] += 1.0;
///     }
/// }
///
/// // Generated: add_one::CUDA_SOURCE, add_one::launch(...)
/// ```
///
/// # Supported Constructs
///
/// - Scalar types: `f32`, `f64`, `i32`, `u32`, `bool`
/// - Array parameters: `&Array<T>` (read-only), `&mut Array<T>` (read-write)
/// - `thread_id()` → CUDA thread index
/// - Math ops: `+`, `-`, `*`, `/`, `%`, comparisons
/// - Control flow: `if`/`else`, `for`, `while`
/// - Builtins: `sin`, `cos`, `sqrt`, `abs`, `min`, `max`
#[proc_macro_attribute]
pub fn kernel(_attr: TokenStream, item: TokenStream) -> TokenStream {
    kernel::expand_kernel(item.into())
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

/// Mark a function as a GPU device function callable from kernels.
///
/// The `#[func]` macro transforms a Rust function into a CUDA `__device__` function.
/// The generated module contains a `CUDA_SOURCE` constant that should be passed
/// to `launch_with_funcs()` when launching kernels that reference this function.
///
/// # Example
///
/// ```rust,ignore
/// use forge_macros::func;
///
/// #[func]
/// fn clamp_val(x: f32, lo: f32, hi: f32) -> f32 {
///     if x < lo { return lo; }
///     if x > hi { return hi; }
///     return x;
/// }
///
/// // Generated: clamp_val::CUDA_SOURCE (a __device__ function)
/// ```
#[proc_macro_attribute]
pub fn func(_attr: TokenStream, item: TokenStream) -> TokenStream {
    func::expand_func(item.into())
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}
