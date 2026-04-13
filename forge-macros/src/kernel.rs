//! `#[kernel]` proc macro expansion.
//!
//! Transforms a Rust function into:
//! 1. A constant with the generated CUDA C++ source
//! 2. A module with a `launch()` function that compiles and runs the kernel

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{parse2, ItemFn, Block};

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

    // Build the CPU fallback body
    let cpu_fallback_body = build_cpu_fallback(&params, &func.block);

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

            /// CPU fallback: runs the kernel body sequentially on CPU arrays.
            ///
            /// Arrays are auto-transferred to CPU if on GPU, and back after.
            /// `thread_id()` is simulated by iterating [0, dim).
            /// Useful for debugging or when no GPU is available.
            pub fn launch_cpu(
                #(#launch_params,)*
                dim: usize,
            ) -> Result<(), ::forge_runtime::ForgeError> {
                #cpu_fallback_body
            }
        }
    };

    Ok(expanded)
}

/// Build the CPU fallback body for a kernel.
///
/// Generates code that:
/// 1. Copies GPU arrays to CPU Vecs
/// 2. Loops over 0..dim, setting thread_id for each iteration
/// 3. Executes the original Rust kernel body
/// 4. Copies mutable arrays back to GPU
fn build_cpu_fallback(params: &[KernelParam], body: &syn::Block) -> TokenStream {
    // Step 1: Generate array copies from GPU to CPU
    let mut setup = Vec::new();
    let mut teardown = Vec::new();

    for p in params {
        let name = format_ident!("{}", p.name);
        if p.is_array {
            if p.is_mutable {
                // Mutable array: copy to local vec, then copy back
                let local_name = format_ident!("_cpu_{}", p.name);
                setup.push(quote! {
                    let mut #local_name: Vec<_> = #name.to_vec();
                });
                teardown.push(quote! {
                    // Write results back to the original array
                    let _new_arr = ::forge_runtime::Array::from_vec(#local_name, #name.device());
                    *#name = _new_arr;
                });
            } else {
                // Read-only array: just copy to local vec
                let local_name = format_ident!("_cpu_{}", p.name);
                setup.push(quote! {
                    let #local_name: Vec<_> = #name.to_vec();
                });
            }
        }
    }

    // Build array rebinds: wrap Vecs in CpuSlice for i32 indexing
    let mut array_rebinds = Vec::new();
    for p in params {
        let name = format_ident!("{}", p.name);
        let local_name = format_ident!("_cpu_{}", p.name);
        if p.is_array {
            if p.is_mutable {
                array_rebinds.push(quote! {
                    let mut #name = _ForgeCpuSliceMut(&mut #local_name[..]);
                });
            } else {
                array_rebinds.push(quote! {
                    let #name = _ForgeCpuSlice(&#local_name[..]);
                });
            }
        }
    }

    let kernel_body = &body.stmts;

    quote! {
        // Wrapper that allows indexing with i32 (like CUDA)
        struct _ForgeCpuSlice<'a, T>(&'a [T]);
        impl<T> ::std::ops::Index<i32> for _ForgeCpuSlice<'_, T> {
            type Output = T;
            #[inline(always)]
            fn index(&self, i: i32) -> &T { &self.0[i as usize] }
        }
        impl<T> ::std::ops::Index<usize> for _ForgeCpuSlice<'_, T> {
            type Output = T;
            #[inline(always)]
            fn index(&self, i: usize) -> &T { &self.0[i] }
        }

        struct _ForgeCpuSliceMut<'a, T>(&'a mut [T]);
        impl<T> ::std::ops::Index<i32> for _ForgeCpuSliceMut<'_, T> {
            type Output = T;
            #[inline(always)]
            fn index(&self, i: i32) -> &T { &self.0[i as usize] }
        }
        impl<T> ::std::ops::Index<usize> for _ForgeCpuSliceMut<'_, T> {
            type Output = T;
            #[inline(always)]
            fn index(&self, i: usize) -> &T { &self.0[i] }
        }
        impl<T> ::std::ops::IndexMut<i32> for _ForgeCpuSliceMut<'_, T> {
            #[inline(always)]
            fn index_mut(&mut self, i: i32) -> &mut T { &mut self.0[i as usize] }
        }
        impl<T> ::std::ops::IndexMut<usize> for _ForgeCpuSliceMut<'_, T> {
            #[inline(always)]
            fn index_mut(&mut self, i: usize) -> &mut T { &mut self.0[i] }
        }

        // CPU builtin functions that mirror CUDA builtins
        #[allow(unused)]
        #[inline(always)]
        fn thread_id() -> i32 {
            _FORGE_CPU_TID.with(|t| *t.borrow())
        }
        #[allow(unused)]
        #[inline(always)]
        fn sin(x: f32) -> f32 { x.sin() }
        #[allow(unused)]
        #[inline(always)]
        fn cos(x: f32) -> f32 { x.cos() }
        #[allow(unused)]
        #[inline(always)]
        fn sqrt(x: f32) -> f32 { x.sqrt() }
        #[allow(unused)]
        #[inline(always)]
        fn abs(x: f32) -> f32 { x.abs() }
        #[allow(unused)]
        #[inline(always)]
        fn exp(x: f32) -> f32 { x.exp() }
        #[allow(unused)]
        #[inline(always)]
        fn log(x: f32) -> f32 { x.ln() }
        #[allow(unused)]
        #[inline(always)]
        fn pow(x: f32, y: f32) -> f32 { x.powf(y) }
        #[allow(unused)]
        #[inline(always)]
        fn min(x: f32, y: f32) -> f32 { x.min(y) }
        #[allow(unused)]
        #[inline(always)]
        fn max(x: f32, y: f32) -> f32 { x.max(y) }
        #[allow(unused)]
        fn atomic_add(arr: &mut _ForgeCpuSliceMut<'_, f32>, idx: i32, val: f32) {
            arr[idx] += val;
        }

        ::std::thread_local! {
            static _FORGE_CPU_TID: ::std::cell::RefCell<i32> = ::std::cell::RefCell::new(0);
        }

        // Copy arrays to CPU
        #(#setup)*

        // Rebind array names to indexed wrappers for the kernel body
        {
            #(#array_rebinds)*

            for _forge_i in 0..dim {
                _FORGE_CPU_TID.with(|t| *t.borrow_mut() = _forge_i as i32);
                // Execute kernel body
                #(#kernel_body)*
            }
        }

        // Copy mutable arrays back
        #(#teardown)*

        Ok(())
    }
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
