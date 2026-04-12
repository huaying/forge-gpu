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
    let launch_arg_names = build_launch_arg_names(&params);

    let expanded = quote! {
        /// Generated kernel module for `#fn_name_str`.
        pub mod #mod_name {
            use super::*;

            /// The generated CUDA C++ source code for this kernel.
            pub const CUDA_SOURCE: &str = #cuda_source;

            /// The kernel function name in the CUDA source.
            pub const KERNEL_NAME: &str = #fn_name_str;

            /// Cached compiled kernel (compiled on first launch, no device funcs).
            static COMPILED: ::std::sync::OnceLock<::forge_runtime::CompiledKernel> =
                ::std::sync::OnceLock::new();

            /// Get or compile the kernel (no device functions).
            pub fn compiled() -> &'static ::forge_runtime::CompiledKernel {
                COMPILED.get_or_init(|| {
                    ::forge_runtime::CompiledKernel::compile(CUDA_SOURCE, KERNEL_NAME)
                        .expect(&format!("Failed to compile kernel '{}'", KERNEL_NAME))
                })
            }

            /// Compile the kernel with device function sources prepended.
            ///
            /// This is NOT cached via OnceLock since different func combinations
            /// produce different sources. Callers should cache the result themselves
            /// if calling repeatedly.
            pub fn compile_with_funcs(device_func_sources: &[&str]) -> ::forge_runtime::CompiledKernel {
                let mut full_source = String::new();
                for src in device_func_sources {
                    full_source.push_str(src);
                    full_source.push('\n');
                }
                full_source.push_str(CUDA_SOURCE);
                ::forge_runtime::CompiledKernel::compile(&full_source, KERNEL_NAME)
                    .expect(&format!("Failed to compile kernel '{}' with device funcs", KERNEL_NAME))
            }

            /// Launch this kernel on the GPU.
            ///
            /// `dim` is the number of threads (elements) to process.
            /// `ordinal` is the CUDA device ordinal (usually 0).
            pub fn launch(
                #(#launch_params,)*
                dim: usize,
                ordinal: usize,
            ) -> Result<(), ::forge_runtime::ForgeError> {
                let kernel = compiled();
                launch_with_kernel(kernel, #(#launch_arg_names,)* dim, ordinal)
            }

            /// Launch this kernel with device function sources prepended.
            ///
            /// Pass `&[func_name::CUDA_SOURCE]` for each `#[func]` used.
            pub fn launch_with_funcs(
                #(#launch_params,)*
                dim: usize,
                ordinal: usize,
                device_func_sources: &[&str],
            ) -> Result<(), ::forge_runtime::ForgeError> {
                let kernel = compile_with_funcs(device_func_sources);
                launch_with_kernel(&kernel, #(#launch_arg_names,)* dim, ordinal)
            }

            /// Launch with an explicit LaunchConfig.
            pub fn launch_with_config(
                #(#launch_params,)*
                ordinal: usize,
                config: ::forge_runtime::cuda::LaunchConfig,
            ) -> Result<(), ::forge_runtime::ForgeError> {
                let kernel = compiled();
                let func = kernel.get_function(ordinal)?;
                let stream = ::forge_runtime::cuda::default_stream(ordinal);

                unsafe {
                    use ::forge_runtime::cuda::PushKernelArg;
                    let mut builder = stream.launch_builder(&func);
                    #(#launch_arg_pushes)*
                    builder.launch(config)
                        .map_err(|e| ::forge_runtime::ForgeError::LaunchFailed(format!("{:?}", e)))?;
                }

                stream.synchronize()
                    .map_err(|e| ::forge_runtime::ForgeError::SyncFailed(format!("{:?}", e)))?;
                Ok(())
            }

            /// Launch asynchronously (no synchronize — caller must sync).
            pub fn launch_async(
                #(#launch_params,)*
                dim: usize,
                ordinal: usize,
            ) -> Result<(), ::forge_runtime::ForgeError> {
                let kernel = compiled();
                let func = kernel.get_function(ordinal)?;
                let stream = ::forge_runtime::cuda::default_stream(ordinal);
                let config = ::forge_runtime::cuda::LaunchConfig::for_num_elems(dim as u32);

                unsafe {
                    use ::forge_runtime::cuda::PushKernelArg;
                    let mut builder = stream.launch_builder(&func);
                    #(#launch_arg_pushes)*
                    builder.launch(config)
                        .map_err(|e| ::forge_runtime::ForgeError::LaunchFailed(format!("{:?}", e)))?;
                }

                Ok(())
            }

            fn launch_with_kernel(
                kernel: &::forge_runtime::CompiledKernel,
                #(#launch_params,)*
                dim: usize,
                ordinal: usize,
            ) -> Result<(), ::forge_runtime::ForgeError> {
                let func = kernel.get_function(ordinal)?;
                let stream = ::forge_runtime::cuda::default_stream(ordinal);
                let config = ::forge_runtime::cuda::LaunchConfig::for_num_elems(dim as u32);

                unsafe {
                    use ::forge_runtime::cuda::PushKernelArg;
                    let mut builder = stream.launch_builder(&func);
                    #(#launch_arg_pushes)*
                    builder.launch(config)
                        .map_err(|e| ::forge_runtime::ForgeError::LaunchFailed(format!("{:?}", e)))?;
                }

                stream.synchronize()
                    .map_err(|e| ::forge_runtime::ForgeError::SyncFailed(format!("{:?}", e)))?;
                Ok(())
            }
        }
    };

    Ok(expanded)
}

/// Build the launch function parameter list (Rust types).
pub(crate) fn build_launch_params(params: &[KernelParam]) -> Vec<TokenStream> {
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

/// Build just the argument names for forwarding calls.
pub(crate) fn build_launch_arg_names(params: &[KernelParam]) -> Vec<TokenStream> {
    params
        .iter()
        .map(|p| {
            let name = format_ident!("{}", p.name);
            quote! { #name }
        })
        .collect()
}

/// Build the `.arg()` calls for the launch builder.
pub(crate) fn build_launch_arg_pushes(params: &[KernelParam]) -> Vec<TokenStream> {
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

/// Public wrapper for scalar_type_tokens.
pub(crate) fn scalar_type_tokens_pub(type_name: &str) -> TokenStream {
    scalar_type_tokens(type_name)
}

/// Public wrapper for elem_type_tokens.
pub(crate) fn elem_type_tokens_pub(type_name: &str) -> TokenStream {
    elem_type_tokens(type_name)
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
    match type_name {
        "Vec2f" | "Vec2" => quote! { forge_core::Vec2f },
        "Vec3f" | "Vec3" => quote! { forge_core::Vec3f },
        "Vec4f" | "Vec4" => quote! { forge_core::Vec4f },
        "Vec2d" => quote! { forge_core::Vec2d },
        "Vec3d" => quote! { forge_core::Vec3d },
        "Vec4d" => quote! { forge_core::Vec4d },
        _ => scalar_type_tokens(type_name),
    }
}
