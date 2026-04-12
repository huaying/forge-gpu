//! `#[func]` proc macro expansion.
//!
//! Transforms a Rust function into a CUDA `__device__` function
//! that can be called from `#[kernel]` functions.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{parse2, ItemFn};

use crate::cuda_emit::{generate_device_func_source, parse_param, KernelParam};

/// Main expansion entry point for `#[func]`.
pub fn expand_func(input: TokenStream) -> Result<TokenStream, syn::Error> {
    let func: ItemFn = parse2(input)?;

    let fn_name = &func.sig.ident;
    let fn_name_str = fn_name.to_string();

    // Parse all parameters
    let params: Vec<KernelParam> = func
        .sig
        .inputs
        .iter()
        .map(parse_param)
        .collect::<Result<Vec<_>, _>>()?;

    // Parse return type
    let return_type = match &func.sig.output {
        syn::ReturnType::Default => None,
        syn::ReturnType::Type(_, ty) => Some(ty.as_ref().clone()),
    };

    // Generate CUDA __device__ function source
    let cuda_source = generate_device_func_source(
        &fn_name_str,
        &params,
        &func.block.stmts,
        return_type.as_ref(),
    )?;

    let mod_name = format_ident!("{}", fn_name_str);

    let expanded = quote! {
        /// Generated device function module for `#fn_name_str`.
        pub mod #mod_name {
            /// The generated CUDA C++ source code for this device function.
            ///
            /// Pass this to `launch_with_funcs()` when launching kernels
            /// that call this function.
            pub const CUDA_SOURCE: &str = #cuda_source;

            /// The function name in the CUDA source.
            pub const FUNC_NAME: &str = #fn_name_str;
        }
    };

    Ok(expanded)
}
