//! `#[kernel(autodiff)]` expansion.
//!
//! Generates both a forward kernel and its adjoint (backward) kernel.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{parse2, ItemFn};
use std::collections::HashSet;

use crate::cuda_emit::{generate_cuda_source, generate_struct_preamble, parse_param, KernelParam};
use crate::autodiff::generate_adjoint_body;
use crate::autodiff_parse::{parse_stmts_to_ops, parse_stmts_to_ops_with_types};
use crate::kernel::{build_launch_params, build_launch_arg_names, build_launch_arg_pushes};

/// Generate both forward and adjoint kernels from `#[kernel(autodiff)]`.
pub fn expand_kernel_autodiff(input: TokenStream) -> Result<TokenStream, syn::Error> {
    let func: ItemFn = parse2(input)?;

    let fn_name = &func.sig.ident;
    let fn_name_str = fn_name.to_string();
    let adj_name_str = format!("{}_adjoint", fn_name_str);

    // Parse parameters
    let params: Vec<KernelParam> = func
        .sig
        .inputs
        .iter()
        .map(parse_param)
        .collect::<Result<Vec<_>, _>>()?;

    // Generate forward CUDA source (same as regular #[kernel])
    let forward_cuda = generate_cuda_source(&fn_name_str, &params, &func.block.stmts)?;

    // Classify params for autodiff
    let output_arrays: HashSet<String> = params.iter()
        .filter(|p| p.is_array && p.is_mutable)
        .map(|p| p.name.clone())
        .collect();
    let input_arrays: HashSet<String> = params.iter()
        .filter(|p| p.is_array && !p.is_mutable)
        .map(|p| p.name.clone())
        .collect();

    // Parse forward body into IR with type information
    let mut param_type_map = std::collections::HashMap::new();
    for p in &params {
        let cuda_type = if p.is_array {
            // For arrays, the "type" in context is the element type
            crate::cuda_emit::type_to_cuda(&p.elem_type).unwrap_or("float".to_string())
        } else {
            crate::cuda_emit::type_to_cuda(&p.elem_type).unwrap_or("float".to_string())
        };
        param_type_map.insert(p.name.clone(), cuda_type);
    }
    let forward_ops = parse_stmts_to_ops_with_types(&func.block.stmts, &param_type_map);

    // Generate adjoint body
    let adj_lines = generate_adjoint_body(&forward_ops, &output_arrays, &input_arrays);

    // Build adjoint kernel CUDA source
    let mut adj_cuda = String::new();

    // Preamble (struct definitions if needed)
    let preamble = generate_struct_preamble(&params, &func.block.stmts);
    adj_cuda.push_str(&preamble);

    // Adjoint kernel signature: same params + adj_ version of each array
    adj_cuda.push_str("extern \"C\" __global__ void ");
    adj_cuda.push_str(&adj_name_str);
    adj_cuda.push('(');

    let mut adj_param_strs = Vec::new();
    // Original params
    for p in &params {
        adj_param_strs.push(format!("{} {}", p.cuda_type, p.name));
    }
    // Adjoint params: adj_ for each array
    for p in &params {
        if p.is_array {
            // Adjoint arrays: strip "const " from type if present, ensure it's a mutable pointer
            let base_type = p.cuda_type.trim_start_matches("const ").trim();
            // base_type is like "float*" — use directly
            adj_param_strs.push(format!("{} adj_{}", base_type, p.name));
        }
    }
    adj_cuda.push_str(&adj_param_strs.join(", "));
    adj_cuda.push_str(") {\n");

    // Thread ID — already declared at top
    adj_cuda.push_str("    int tid = blockIdx.x * blockDim.x + threadIdx.x;\n\n");

    // Declare adj_ locals for all SSA vars and named vars (deduplicated)
    let mut declared_adj: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Mark 'tid' as already declared so it doesn't get redeclared
    declared_adj.insert("tid".to_string());
    declared_adj.insert("fwd_tid".to_string());

    // Declare adj_ for scalar params
    for p in &params {
        if !p.is_array {
            declared_adj.insert(p.name.clone());
            let cuda_type = crate::cuda_emit::type_to_cuda(&p.elem_type).unwrap_or("float".to_string());
            adj_cuda.push_str(&format!("    {} adj_{} = 0.0f;\n", cuda_type, p.name));
        }
    }

    declare_adj_vars_recursive(&forward_ops, &mut declared_adj, &mut adj_cuda);

    // Re-run forward to compute intermediate values needed by adjoint
    // Declare ALL forward variables at top scope so backward can see them
    adj_cuda.push_str("\n    // === Forward variables ===\n");
    declare_forward_vars_recursive(&forward_ops, &mut declared_adj, &mut adj_cuda);

    adj_cuda.push_str("\n    // === Forward pass (recompute) ===\n");
    for op in &forward_ops {
        let fwd_line = forward_op_to_cuda_assign_only(op);
        adj_cuda.push_str(&format!("    {}\n", fwd_line));
    }

    adj_cuda.push_str("\n    // === Backward pass ===\n");
    for line in &adj_lines {
        adj_cuda.push_str(line);
        adj_cuda.push('\n');
    }

    adj_cuda.push_str("}\n");

    // Now generate the Rust module with both kernels
    let mod_name = format_ident!("{}", fn_name_str);
    let _adj_mod_name = format_ident!("{}_adjoint", fn_name_str);

    // Build launch params for forward
    let launch_params = build_launch_params(&params);
    let launch_arg_pushes = build_launch_arg_pushes(&params);
    let _launch_arg_names = build_launch_arg_names(&params);

    // Build adjoint launch params (original + adj_ arrays)
    let mut adj_launch_params_tokens = Vec::new();
    let mut adj_launch_arg_pushes_tokens = Vec::new();

    // Original params
    for p in &params {
        let name = format_ident!("{}", p.name);
        if p.is_array {
            let elem = crate::kernel::elem_type_tokens_pub(&p.elem_type);
            if p.is_mutable {
                adj_launch_params_tokens.push(quote! { #name: &mut ::forge_runtime::Array<#elem> });
            } else {
                adj_launch_params_tokens.push(quote! { #name: &::forge_runtime::Array<#elem> });
            }
            if p.is_mutable {
                adj_launch_arg_pushes_tokens.push(quote! {
                    builder.arg(#name.cuda_slice_mut().expect("Array must be on GPU"));
                });
            } else {
                adj_launch_arg_pushes_tokens.push(quote! {
                    builder.arg(#name.cuda_slice().expect("Array must be on GPU"));
                });
            }
        } else {
            let ty = crate::kernel::scalar_type_tokens_pub(&p.elem_type);
            adj_launch_params_tokens.push(quote! { #name: #ty });
            adj_launch_arg_pushes_tokens.push(quote! { builder.arg(&#name); });
        }
    }
    // Adjoint array params
    for p in &params {
        if p.is_array {
            let adj_name = format_ident!("adj_{}", p.name);
            let elem = crate::kernel::elem_type_tokens_pub(&p.elem_type);
            adj_launch_params_tokens.push(quote! { #adj_name: &mut ::forge_runtime::Array<#elem> });
            adj_launch_arg_pushes_tokens.push(quote! {
                builder.arg(#adj_name.cuda_slice_mut().expect("Array must be on GPU"));
            });
        }
    }

    let expanded = quote! {
        /// Generated forward + adjoint kernel module.
        pub mod #mod_name {
            use super::*;

            /// Forward kernel CUDA source.
            pub const CUDA_SOURCE: &str = #forward_cuda;
            pub const KERNEL_NAME: &str = #fn_name_str;

            /// Adjoint (backward) kernel CUDA source.
            pub const ADJOINT_CUDA_SOURCE: &str = #adj_cuda;
            pub const ADJOINT_KERNEL_NAME: &str = #adj_name_str;

            static COMPILED: ::std::sync::OnceLock<::forge_runtime::CompiledKernel> =
                ::std::sync::OnceLock::new();

            static ADJOINT_COMPILED: ::std::sync::OnceLock<::forge_runtime::CompiledKernel> =
                ::std::sync::OnceLock::new();

            pub fn compiled() -> &'static ::forge_runtime::CompiledKernel {
                COMPILED.get_or_init(|| {
                    ::forge_runtime::CompiledKernel::compile(CUDA_SOURCE, KERNEL_NAME)
                        .expect(&format!("Failed to compile forward kernel '{}'", KERNEL_NAME))
                })
            }

            pub fn adjoint_compiled() -> &'static ::forge_runtime::CompiledKernel {
                ADJOINT_COMPILED.get_or_init(|| {
                    ::forge_runtime::CompiledKernel::compile(ADJOINT_CUDA_SOURCE, ADJOINT_KERNEL_NAME)
                        .expect(&format!("Failed to compile adjoint kernel '{}'", ADJOINT_KERNEL_NAME))
                })
            }

            /// Launch the forward kernel.
            pub fn launch(
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
                stream.synchronize()
                    .map_err(|e| ::forge_runtime::ForgeError::SyncFailed(format!("{:?}", e)))?;
                Ok(())
            }

            /// Launch the adjoint (backward) kernel.
            ///
            /// Pass the same arrays as forward, plus `adj_*` arrays for each array param.
            /// Output arrays' `adj_*` are inputs (seeded with loss gradients).
            /// Input arrays' `adj_*` are outputs (accumulated gradients).
            pub fn launch_adjoint(
                #(#adj_launch_params_tokens,)*
                dim: usize,
                ordinal: usize,
            ) -> Result<(), ::forge_runtime::ForgeError> {
                let kernel = adjoint_compiled();
                let func = kernel.get_function(ordinal)?;
                let stream = ::forge_runtime::cuda::default_stream(ordinal);
                let config = ::forge_runtime::cuda::LaunchConfig::for_num_elems(dim as u32);
                unsafe {
                    use ::forge_runtime::cuda::PushKernelArg;
                    let mut builder = stream.launch_builder(&func);
                    #(#adj_launch_arg_pushes_tokens)*
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


/// Recursively declare adj_ variables for all forward ops (including those inside if blocks).
fn declare_adj_vars_recursive(
    ops: &[crate::autodiff::ForwardOp],
    declared: &mut std::collections::HashSet<String>,
    cuda: &mut String,
) {
    use crate::autodiff::ForwardOp;
    for op in ops {
        match op {
            ForwardOp::IfBlock { then_ops, else_ops, .. } => {
                declare_adj_vars_recursive(then_ops, declared, cuda);
                declare_adj_vars_recursive(else_ops, declared, cuda);
            }
            _ => {
                if let Some(var) = get_forward_var(op) {
                    if declared.insert(var.to_string()) {
                        let ty = get_forward_type(op);
                        if ty == "bool" || ty == "int" {
                            // No adj for bools or ints
                        } else if ty.starts_with("forge_vec") {
                            cuda.push_str(&format!("    {} adj_{} = {{}};\n", ty, var));
                        } else {
                            cuda.push_str(&format!("    float adj_{} = 0.0f;\n", var));
                        }
                    }
                }
            }
        }
    }
}

fn get_forward_var(op: &crate::autodiff::ForwardOp) -> Option<&str> {
    use crate::autodiff::ForwardOp;
    match op {
        ForwardOp::ThreadId { var } | ForwardOp::Literal { var, .. } |
        ForwardOp::BinOp { var, .. } | ForwardOp::UnaryFunc { var, .. } |
        ForwardOp::ArrayRead { var, .. } | ForwardOp::FieldAccess { var, .. } |
        ForwardOp::Assign { var, .. } | ForwardOp::VecConstruct { var, .. } => Some(var),
        _ => None,
    }
}

fn get_forward_type(op: &crate::autodiff::ForwardOp) -> String {
    use crate::autodiff::ForwardOp;
    match op {
        ForwardOp::ThreadId { .. } => "int".to_string(),
        ForwardOp::Literal { cuda_type, .. } => cuda_type.clone(),
        ForwardOp::BinOp { result_type, .. } => result_type.clone(),
        ForwardOp::UnaryFunc { result_type, .. } => result_type.clone(),
        ForwardOp::ArrayRead { elem_type, .. } => elem_type.clone(),
        ForwardOp::FieldAccess { .. } => "float".to_string(),
        ForwardOp::Assign { value_type, .. } => value_type.clone(),
        ForwardOp::VecConstruct { vec_type, .. } => vec_type.clone(),
        _ => "float".to_string(),
    }
}

/// Declare forward variables at top scope (so backward can access them).
fn declare_forward_vars_recursive(
    ops: &[crate::autodiff::ForwardOp],
    declared: &mut std::collections::HashSet<String>,
    cuda: &mut String,
) {
    use crate::autodiff::ForwardOp;
    for op in ops {
        match op {
            ForwardOp::IfBlock { then_ops, else_ops, .. } => {
                declare_forward_vars_recursive(then_ops, declared, cuda);
                declare_forward_vars_recursive(else_ops, declared, cuda);
            }
            ForwardOp::ThreadId { var } => {
                if declared.insert(format!("fwd_{}", var)) {
                    cuda.push_str(&format!("    int {} = 0;\n", var));
                }
            }
            ForwardOp::Literal { var, cuda_type, .. } => {
                if declared.insert(format!("fwd_{}", var)) {
                    cuda.push_str(&format!("    {} {} = {{}};\n", cuda_type, var));
                }
            }
            ForwardOp::BinOp { var, result_type, .. } => {
                if declared.insert(format!("fwd_{}", var)) {
                    if result_type.starts_with("forge_vec") {
                        cuda.push_str(&format!("    {} {} = {{}};\n", result_type, var));
                    } else if result_type == "int" {
                        cuda.push_str(&format!("    int {} = 0;\n", var));
                    } else {
                        cuda.push_str(&format!("    float {} = 0.0f;\n", var));
                    }
                }
            }
            ForwardOp::UnaryFunc { var, result_type, .. } => {
                if declared.insert(format!("fwd_{}", var)) {
                    cuda.push_str(&format!("    float {} = 0.0f;\n", var));
                }
            }
            ForwardOp::ArrayRead { var, elem_type, .. } => {
                if declared.insert(format!("fwd_{}", var)) {
                    if elem_type.starts_with("forge_vec") {
                        cuda.push_str(&format!("    {} {} = {{}};\n", elem_type, var));
                    } else {
                        cuda.push_str(&format!("    float {} = 0.0f;\n", var));
                    }
                }
            }
            ForwardOp::FieldAccess { var, .. } => {
                if declared.insert(format!("fwd_{}", var)) {
                    cuda.push_str(&format!("    float {} = 0.0f;\n", var));
                }
            }
            ForwardOp::Assign { var, value_type, .. } => {
                if declared.insert(format!("fwd_{}", var)) {
                    if value_type == "int" {
                        cuda.push_str(&format!("    int {} = 0;\n", var));
                    } else if value_type.starts_with("forge_vec") {
                        cuda.push_str(&format!("    {} {} = {{}};\n", value_type, var));
                    } else {
                        cuda.push_str(&format!("    float {} = 0.0f;\n", var));
                    }
                }
            }
            ForwardOp::VecConstruct { var, vec_type, .. } => {
                if declared.insert(format!("fwd_{}", var)) {
                    cuda.push_str(&format!("    {} {} = {{}};\n", vec_type, var));
                }
            }
            _ => {}
        }
    }
}

/// Generate CUDA for a forward op using ASSIGNMENT (not declaration).
fn forward_op_to_cuda_assign_only(op: &crate::autodiff::ForwardOp) -> String {
    use crate::autodiff::ForwardOp;
    match op {
        ForwardOp::ThreadId { var } => format!("{} = blockIdx.x * blockDim.x + threadIdx.x;", var),
        ForwardOp::Literal { var, value, .. } => format!("{} = {};", var, value),
        ForwardOp::BinOp { var, left, op, right, .. } => {
            format!("{} = ({} {} {});", var, left, op.as_str(), right)
        }
        ForwardOp::UnaryFunc { var, func, arg, .. } => {
            format!("{} = {}({});", var, func, arg)
        }
        ForwardOp::ArrayRead { var, array, index, .. } => {
            format!("{} = {}[{}];", var, array, index)
        }
        ForwardOp::ArrayWrite { array, index, value, .. } => {
            format!("{}[{}] = {};", array, index, value)
        }
        ForwardOp::ArrayCompound { array, index, op, value, .. } => {
            format!("{}[{}] {}= {};", array, index, op.as_str(), value)
        }
        ForwardOp::FieldAccess { var, base, field, .. } => {
            format!("{} = {}.{};", var, base, field)
        }
        ForwardOp::Assign { var, value, .. } => {
            format!("{} = {};", var, value)
        }
        ForwardOp::VecConstruct { var, vec_type, args } => {
            format!("{} = {}{{{}}};", var, vec_type, args.join(", "))
        }
        ForwardOp::IfBlock { cond, then_ops, else_ops } => {
            let mut s = format!("if ({}) {{\n", cond);
            for op in then_ops {
                s.push_str(&format!("        {}\n", forward_op_to_cuda_assign_only(op)));
            }
            if !else_ops.is_empty() {
                s.push_str("    } else {\n");
                for op in else_ops {
                    s.push_str(&format!("        {}\n", forward_op_to_cuda_assign_only(op)));
                }
            }
            s.push_str("    }");
            s
        }
    }
}
