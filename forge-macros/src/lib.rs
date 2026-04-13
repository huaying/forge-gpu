//! # forge-macros
//!
//! Proc macros for Forge GPU compute framework.
//!
//! Provides:
//! - `#[kernel]` — transforms a Rust function into a GPU kernel with host-side launch wrapper
//! - `#[func]` (future) — marks device-callable functions

extern crate proc_macro;

mod autodiff;
mod autodiff_parse;
mod cuda_emit;
mod func;
mod kernel;
mod kernel_autodiff;
mod forge_struct;

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
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr_str = attr.to_string();
    if attr_str.contains("autodiff") {
        kernel_autodiff::expand_kernel_autodiff(item.into())
            .unwrap_or_else(|e| e.to_compile_error())
            .into()
    } else {
        kernel::expand_kernel(item.into())
            .unwrap_or_else(|e| e.to_compile_error())
            .into()
    }
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

/// Mark a struct as usable in GPU kernels.
///
/// Generates a CUDA C struct definition with matching fields and
/// optional operator overloads (if all fields are the same scalar type).
///
/// # Example
///
/// ```rust,ignore
/// use forge_macros::forge_struct;
///
/// #[forge_struct]
/// #[derive(Clone, Copy)]
/// pub struct Particle {
///     pub x: f32,
///     pub y: f32,
///     pub z: f32,
///     pub mass: f32,
/// }
///
/// // Generated: Particle_forge_meta::CUDA_STRUCT_DEF (CUDA C struct string)
/// ```
#[proc_macro_attribute]
pub fn forge_struct(_attr: TokenStream, item: TokenStream) -> TokenStream {
    forge_struct::expand_forge_struct(item.into())
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}
