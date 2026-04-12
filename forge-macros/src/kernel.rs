//! `#[kernel]` proc macro expansion.
//!
//! Transforms a Rust function into:
//! 1. A constant with the generated CUDA C++ source
//! 2. A module with a `launch()` function that compiles and runs the kernel

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{parse2, ItemFn};

use crate::cuda_emit::{generate_cuda_source, parse_param, KernelParam};

/// Main expansion entry point for `#[kernel]`.
pub fn expand_kernel(input: TokenStream) -> Result<TokenStream, syn::Error> {
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

    // Generate CUDA source
    let cuda_source = generate_cuda_source(&fn_name_str, &params, &func.block.stmts)?;

    // Generate the host-side launch wrapper
    let mod_name = format_ident!("{}", fn_name_str);

    // Build the launch function parameters (Rust side)
    let launch_params = build_launch_params(&params);
    let launch_arg_pushes = build_launch_arg_pushes(&params);

    let expanded = quote! {
        /// Generated kernel module for `#fn_name_str`.
        pub mod #mod_name {
            use super::*;

            /// The generated CUDA C++ source code for this kernel.
            pub const CUDA_SOURCE: &str = #cuda_source;

            /// The kernel function name in the CUDA source.
            pub const KERNEL_NAME: &str = #fn_name_str;

            /// Cached compiled kernel (compiled on first launch).
            static COMPILED: ::std::sync::OnceLock<::forge_runtime::CompiledKernel> =
                ::std::sync::OnceLock::new();

            /// Get or compile the kernel.
            pub fn compiled() -> &'static ::forge_runtime::CompiledKernel {
                COMPILED.get_or_init(|| {
                    ::forge_runtime::CompiledKernel::compile(CUDA_SOURCE, KERNEL_NAME)
                        .expect(&format!("Failed to compile kernel '{}'", KERNEL_NAME))
                })
            }

            /// Launch this kernel on the GPU.
            ///
            /// `dim` is the number of threads (elements) to process.
            /// `ordinal` is the CUDA device ordinal (usually 0).
            pub fn launch(
                #(#launch_params,)*
                dim: usize,
                ordinal: usize,
            ) -> Result<(), String> {
                let kernel = compiled();
                let func = kernel.get_function(ordinal)?;
                let stream = ::forge_runtime::cuda::default_stream(ordinal);
                let config = ::forge_runtime::cuda::LaunchConfig::for_num_elems(dim as u32);

                unsafe {
                    use ::forge_runtime::cuda::PushKernelArg;
                    let mut builder = stream.launch_builder(&func);
                    #(#launch_arg_pushes)*
                    builder.launch(config)
                        .map_err(|e| format!("Kernel launch failed: {:?}", e))?;
                }

                stream.synchronize().map_err(|e| format!("Synchronize failed: {:?}", e))?;
                Ok(())
            }
        }
    };

    Ok(expanded)
}

/// Build the launch function parameter list (Rust types).
fn build_launch_params(params: &[KernelParam]) -> Vec<TokenStream> {
    params
        .iter()
        .map(|p| {
            let name = format_ident!("{}", p.name);
            if p.is_array {
                if p.is_mutable {
                    let elem = elem_type_tokens(&p.elem_type);
                    quote! { #name: &mut ::forge_runtime::Array<#elem> }
                } else {
                    let elem = elem_type_tokens(&p.elem_type);
                    quote! { #name: &::forge_runtime::Array<#elem> }
                }
            } else {
                let ty = scalar_type_tokens(&p.elem_type);
                quote! { #name: #ty }
            }
        })
        .collect()
}

/// Build the `.arg()` calls for the launch builder.
fn build_launch_arg_pushes(params: &[KernelParam]) -> Vec<TokenStream> {
    params
        .iter()
        .map(|p| {
            let name = format_ident!("{}", p.name);
            if p.is_array {
                if p.is_mutable {
                    quote! {
                        builder.arg(
                            #name.cuda_slice_mut()
                                .expect("Array must be on GPU for kernel launch")
                        );
                    }
                } else {
                    quote! {
                        builder.arg(
                            #name.cuda_slice()
                                .expect("Array must be on GPU for kernel launch")
                        );
                    }
                }
            } else {
                quote! {
                    builder.arg(&#name);
                }
            }
        })
        .collect()
}

/// Convert a scalar type name to a token for use in generated code.
fn scalar_type_tokens(type_name: &str) -> TokenStream {
    match type_name {
        "f32" => quote! { f32 },
        "f64" => quote! { f64 },
        "i32" => quote! { i32 },
        "i64" => quote! { i64 },
        "u32" => quote! { u32 },
        "u64" => quote! { u64 },
        "bool" => quote! { bool },
        "usize" => quote! { usize },
        _ => {
            let ident = format_ident!("{}", type_name);
            quote! { #ident }
        }
    }
}

/// Convert an element type name to tokens.
fn elem_type_tokens(type_name: &str) -> TokenStream {
    scalar_type_tokens(type_name)
}
