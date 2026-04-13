//! Rust AST → CUDA C++ code emitter.
//!
//! Translates parsed Rust function bodies into CUDA C++ kernel source strings.

use proc_macro2::Span;
use syn::{
    BinOp, Expr, ExprAssign, ExprBinary, ExprBlock, ExprCall, ExprCast, ExprField, ExprIf,
    ExprIndex, ExprLit, ExprMethodCall, ExprParen, ExprPath, ExprReturn, ExprUnary, FnArg,
    Lit, Local, Pat, PatIdent, PatType, Stmt, Type, TypePath, TypeReference,
    UnOp,
};

/// Information about a kernel parameter for code generation.
#[derive(Debug, Clone)]
pub struct KernelParam {
    pub name: String,
    pub cuda_type: String,
    pub is_array: bool,
    pub is_mutable: bool,
    pub elem_type: String,
}

/// Parse a function parameter into a KernelParam.
pub fn parse_param(arg: &FnArg) -> Result<KernelParam, syn::Error> {
    match arg {
        FnArg::Typed(PatType { pat, ty, .. }) => {
            let name = match pat.as_ref() {
                Pat::Ident(PatIdent { ident, .. }) => ident.to_string(),
                _ => {
                    return Err(syn::Error::new_spanned(pat, "expected simple identifier"));
                }
            };
            parse_type(&name, ty)
        }
        FnArg::Receiver(_) => Err(syn::Error::new(
            Span::call_site(),
            "#[kernel] functions cannot have self parameter",
        )),
    }
}

/// Parse a type into CUDA type info.
fn parse_type(name: &str, ty: &Type) -> Result<KernelParam, syn::Error> {
    match ty {
        // &Array<T> or &mut Array<T>
        Type::Reference(TypeReference {
            mutability, elem, ..
        }) => {
            let is_mutable = mutability.is_some();
            // Check if it's Array<T>
            if let Type::Path(TypePath { path, .. }) = elem.as_ref() {
                let seg = path.segments.last().ok_or_else(|| {
                    syn::Error::new_spanned(path, "empty type path")
                })?;
                if seg.ident == "Array" {
                    // Extract the element type from Array<T>
                    if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                        if let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first() {
                            let elem_type = type_to_string(inner_ty);
                            let cuda_elem = type_to_cuda(&elem_type)
                                .map_err(|_| syn::Error::new_spanned(
                                    inner_ty,
                                    format!("unsupported Array element type '{}'", elem_type),
                                ))?;
                            let cuda_type = if is_mutable {
                                format!("{}*", cuda_elem)
                            } else {
                                format!("const {}*", cuda_elem)
                            };
                            return Ok(KernelParam {
                                name: name.to_string(),
                                cuda_type,
                                is_array: true,
                                is_mutable,
                                elem_type: elem_type.clone(),
                            });
                        }
                    }
                    return Err(syn::Error::new_spanned(
                        &seg.arguments,
                        "Array must have a type parameter, e.g. Array<f32>",
                    ));
                }
            }
            Err(syn::Error::new_spanned(
                ty,
                "reference parameters must be &Array<T> or &mut Array<T>",
            ))
        }
        // Scalar types: f32, i32, etc.
        Type::Path(TypePath { .. }) => {
            let type_name = type_to_string(ty);
            let cuda_type = scalar_to_cuda(&type_name)?;
            Ok(KernelParam {
                name: name.to_string(),
                cuda_type: cuda_type.to_string(),
                is_array: false,
                is_mutable: false,
                elem_type: type_name,
            })
        }
        _ => Err(syn::Error::new_spanned(
            ty,
            "unsupported parameter type in #[kernel]",
        )),
    }
}

/// Map a Rust scalar type name to its CUDA equivalent.
fn scalar_to_cuda(type_name: &str) -> Result<&'static str, syn::Error> {
    match type_name {
        // Floating point
        "f16" | "half" => Ok("__half"),
        "f32" => Ok("float"),
        "f64" => Ok("double"),
        // Signed integers
        "i8" => Ok("signed char"),
        "i16" => Ok("short"),
        "i32" => Ok("int"),
        "i64" => Ok("long long"),
        // Unsigned integers
        "u8" => Ok("unsigned char"),
        "u16" => Ok("unsigned short"),
        "u32" => Ok("unsigned int"),
        "u64" => Ok("unsigned long long"),
        // Other
        "bool" => Ok("bool"),
        "usize" => Ok("unsigned long long"),
        "isize" => Ok("long long"),
        other => Err(syn::Error::new(
            Span::call_site(),
            format!("unsupported scalar type '{}' in #[kernel]", other),
        )),
    }
}

/// Information about a Forge vector/matrix type for CUDA codegen.
#[derive(Debug, Clone)]
pub struct ForgeStructInfo {
    /// CUDA struct name (e.g., "forge_vec3f")
    pub cuda_name: &'static str,
    /// Scalar element type in CUDA (e.g., "float")
    pub scalar_cuda: &'static str,
    /// Field names
    pub fields: &'static [&'static str],
    /// Number of components
    pub components: usize,
}

/// Look up a known Forge type (Vec2f, Vec3f, etc.).
pub fn forge_type_info(type_name: &str) -> Option<ForgeStructInfo> {
    match type_name {
        "Vec2f" | "Vec2" => Some(ForgeStructInfo {
            cuda_name: "forge_vec2f",
            scalar_cuda: "float",
            fields: &["x", "y"],
            components: 2,
        }),
        "Vec3f" | "Vec3" => Some(ForgeStructInfo {
            cuda_name: "forge_vec3f",
            scalar_cuda: "float",
            fields: &["x", "y", "z"],
            components: 3,
        }),
        "Vec4f" | "Vec4" => Some(ForgeStructInfo {
            cuda_name: "forge_vec4f",
            scalar_cuda: "float",
            fields: &["x", "y", "z", "w"],
            components: 4,
        }),
        "Vec2d" => Some(ForgeStructInfo {
            cuda_name: "forge_vec2d",
            scalar_cuda: "double",
            fields: &["x", "y"],
            components: 2,
        }),
        "Vec3d" => Some(ForgeStructInfo {
            cuda_name: "forge_vec3d",
            scalar_cuda: "double",
            fields: &["x", "y", "z"],
            components: 3,
        }),
        "Vec4d" => Some(ForgeStructInfo {
            cuda_name: "forge_vec4d",
            scalar_cuda: "double",
            fields: &["x", "y", "z", "w"],
            components: 4,
        }),
        _ => None,
    }
}

/// Map any Forge type name (scalar or struct) to its CUDA equivalent.
/// Returns the CUDA type name as a string.
pub fn type_to_cuda(type_name: &str) -> Result<String, syn::Error> {
    // Try scalar first
    if let Ok(s) = scalar_to_cuda(type_name) {
        return Ok(s.to_string());
    }
    // Try forge struct types
    if let Some(info) = forge_type_info(type_name) {
        return Ok(info.cuda_name.to_string());
    }
    Err(syn::Error::new(
        Span::call_site(),
        format!("unsupported type '{}' in #[kernel]", type_name),
    ))
}

/// Generate CUDA struct definitions for all Forge types used in a kernel.
/// Returns a string to prepend to the kernel source.
pub fn generate_struct_preamble(params: &[KernelParam], body_stmts: &[Stmt]) -> String {
    let mut needed: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

    // Check params for struct types
    for p in params {
        if forge_type_info(&p.elem_type).is_some() {
            needed.insert(p.elem_type.clone());
        }
    }

    // Scan body for struct constructors like Vec3f::new(...)
    scan_stmts_for_types(body_stmts, &mut needed);

    let mut preamble = String::new();
    for type_name in &needed {
        if let Some(info) = forge_type_info(type_name) {
            preamble.push_str(&format!("struct {} {{\n", info.cuda_name));
            for &field in info.fields {
                preamble.push_str(&format!("    {} {};\n", info.scalar_cuda, field));
            }
            preamble.push_str("};\n\n");

            // Generate operator overloads
            let n = info.cuda_name;
            let s = info.scalar_cuda;

            // vec + vec
            preamble.push_str(&format!(
                "__device__ {n} operator+({n} a, {n} b) {{ return {n}{{{}}}; }}\n",
                info.fields.iter().map(|f| format!("a.{f} + b.{f}")).collect::<Vec<_>>().join(", ")
            ));
            // vec - vec
            preamble.push_str(&format!(
                "__device__ {n} operator-({n} a, {n} b) {{ return {n}{{{}}}; }}\n",
                info.fields.iter().map(|f| format!("a.{f} - b.{f}")).collect::<Vec<_>>().join(", ")
            ));
            // vec * scalar
            preamble.push_str(&format!(
                "__device__ {n} operator*({n} a, {s} b) {{ return {n}{{{}}}; }}\n",
                info.fields.iter().map(|f| format!("a.{f} * b")).collect::<Vec<_>>().join(", ")
            ));
            // scalar * vec
            preamble.push_str(&format!(
                "__device__ {n} operator*({s} a, {n} b) {{ return {n}{{{}}}; }}\n",
                info.fields.iter().map(|f| format!("a * b.{f}")).collect::<Vec<_>>().join(", ")
            ));
            // vec / scalar
            preamble.push_str(&format!(
                "__device__ {n} operator/({n} a, {s} b) {{ return {n}{{{}}}; }}\n",
                info.fields.iter().map(|f| format!("a.{f} / b")).collect::<Vec<_>>().join(", ")
            ));
            // unary negation
            preamble.push_str(&format!(
                "__device__ {n} operator-({n} a) {{ return {n}{{{}}}; }}\n",
                info.fields.iter().map(|f| format!("-a.{f}")).collect::<Vec<_>>().join(", ")
            ));
            preamble.push('\n');
        }
    }

    preamble
}

/// Scan statements for Forge type references (e.g., Vec3f::new(...)).
fn scan_stmts_for_types(stmts: &[Stmt], found: &mut std::collections::BTreeSet<String>) {
    for stmt in stmts {
        match stmt {
            Stmt::Expr(expr, _) => scan_expr_for_types(expr, found),
            Stmt::Local(local) => {
                if let Some(init) = &local.init {
                    scan_expr_for_types(&init.expr, found);
                }
            }
            _ => {}
        }
    }
}

fn scan_expr_for_types(expr: &Expr, found: &mut std::collections::BTreeSet<String>) {
    match expr {
        Expr::Call(ExprCall { func, args, .. }) => {
            // Check for Type::new(...) patterns
            if let Expr::Path(ExprPath { path, .. }) = func.as_ref() {
                if path.segments.len() == 2 {
                    let type_name = path.segments[0].ident.to_string();
                    if forge_type_info(&type_name).is_some() {
                        found.insert(type_name);
                    }
                }
            }
            for arg in args {
                scan_expr_for_types(arg, found);
            }
        }
        Expr::Binary(ExprBinary { left, right, .. }) => {
            scan_expr_for_types(left, found);
            scan_expr_for_types(right, found);
        }
        Expr::Unary(ExprUnary { expr, .. }) => scan_expr_for_types(expr, found),
        Expr::If(ExprIf { cond, then_branch, else_branch, .. }) => {
            scan_expr_for_types(cond, found);
            scan_stmts_for_types(&then_branch.stmts, found);
            if let Some((_, else_expr)) = else_branch {
                scan_expr_for_types(else_expr, found);
            }
        }
        Expr::Block(ExprBlock { block, .. }) => {
            scan_stmts_for_types(&block.stmts, found);
        }
        Expr::Assign(ExprAssign { left, right, .. }) => {
            scan_expr_for_types(left, found);
            scan_expr_for_types(right, found);
        }
        Expr::Index(ExprIndex { expr, index, .. }) => {
            scan_expr_for_types(expr, found);
            scan_expr_for_types(index, found);
        }
        Expr::Field(ExprField { base, .. }) => scan_expr_for_types(base, found),
        Expr::Paren(ExprParen { expr, .. }) => scan_expr_for_types(expr, found),
        Expr::MethodCall(ExprMethodCall { receiver, args, .. }) => {
            scan_expr_for_types(receiver, found);
            for arg in args {
                scan_expr_for_types(arg, found);
            }
        }
        Expr::ForLoop(for_loop) => {
            scan_expr_for_types(&for_loop.expr, found);
            scan_stmts_for_types(&for_loop.body.stmts, found);
        }
        Expr::While(while_loop) => {
            scan_expr_for_types(&while_loop.cond, found);
            scan_stmts_for_types(&while_loop.body.stmts, found);
        }
        Expr::Return(ExprReturn { expr, .. }) => {
            if let Some(e) = expr { scan_expr_for_types(e, found); }
        }
        Expr::Cast(ExprCast { expr, .. }) => scan_expr_for_types(expr, found),
        _ => {}
    }
}

/// Convert a syn::Type to a string representation.
fn type_to_string(ty: &Type) -> String {
    match ty {
        Type::Path(TypePath { path, .. }) => {
            path.segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect::<Vec<_>>()
                .join("::")
        }
        _ => quote::quote!(#ty).to_string(),
    }
}

/// Infer the CUDA type for a local variable from its initializer expression.
/// Returns None if we can't infer (caller should default to "auto" or error).
fn infer_cuda_type_from_expr(expr: &Expr) -> Option<&'static str> {
    match expr {
        Expr::Call(ExprCall { func, .. }) => {
            if let Expr::Path(ExprPath { path, .. }) = func.as_ref() {
                // Check for Vec3f::new(...) etc.
                if path.segments.len() == 2 {
                    let type_name = path.segments[0].ident.to_string();
                    if let Some(info) = forge_type_info(&type_name) {
                        return Some(info.cuda_name);
                    }
                }
                // Simple function calls
                let fname = path_to_string(path);
                match fname.as_str() {
                    "thread_id" => Some("int"),
                    _ => None,
                }
            } else {
                None
            }
        }
        Expr::Lit(ExprLit { lit, .. }) => match lit {
            Lit::Float(f) => {
                let s = f.to_string();
                if s.ends_with("f64") {
                    Some("double")
                } else {
                    Some("float")
                }
            }
            Lit::Int(i) => {
                let s = i.to_string();
                if s.ends_with("u32") {
                    Some("unsigned int")
                } else if s.ends_with("i64") {
                    Some("long long")
                } else if s.ends_with("u64") {
                    Some("unsigned long long")
                } else {
                    Some("int")
                }
            }
            Lit::Bool(_) => Some("bool"),
            _ => None,
        },
        Expr::Binary(ExprBinary { left, .. }) => infer_cuda_type_from_expr(left),
        Expr::Cast(ExprCast { ty, .. }) => {
            let type_str = type_to_string(ty);
            match scalar_to_cuda(&type_str) {
                Ok(ct) => Some(ct),
                Err(_) => None,
            }
        }
        _ => None,
    }
}

/// Generate a CUDA `__device__` function source.
pub fn generate_device_func_source(
    func_name: &str,
    params: &[KernelParam],
    body_stmts: &[Stmt],
    return_type: Option<&Type>,
) -> Result<String, syn::Error> {
    let mut cuda = String::new();

    // Return type
    let ret_str = match return_type {
        Some(ty) => {
            let type_str = type_to_string(ty);
            scalar_to_cuda(&type_str)
                .map(|s| s.to_string())
                .unwrap_or_else(|_| "void".to_string())
        }
        None => "void".to_string(),
    };

    cuda.push_str("__device__ ");
    cuda.push_str(&ret_str);
    cuda.push(' ');
    cuda.push_str(func_name);
    cuda.push('(');

    let param_strs: Vec<String> = params
        .iter()
        .map(|p| format!("{} {}", p.cuda_type, p.name))
        .collect();
    cuda.push_str(&param_strs.join(", "));

    cuda.push_str(") {\n");

    for stmt in body_stmts {
        let line = emit_stmt(stmt, 1)?;
        cuda.push_str(&line);
    }

    cuda.push_str("}\n");

    Ok(cuda)
}

/// Generate the complete CUDA C++ kernel source.
/// Device utility functions included in every kernel.
const FORGE_DEVICE_UTILS: &str = r#"
// ── Forge PRNG (xorshift32) ──
__device__ unsigned int forge_rand_init(unsigned int seed) {
    // Wang hash for better initial distribution
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);
    return seed;
}

__device__ unsigned int forge_randi(unsigned int* state) {
    unsigned int x = *state;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    *state = x;
    return x;
}

__device__ float forge_randf(unsigned int* state) {
    return (float)forge_randi(state) / 4294967295.0f;
}

__device__ float forge_randf_range(unsigned int* state, float lo, float hi) {
    return lo + forge_randf(state) * (hi - lo);
}
"#;

pub fn generate_cuda_source(
    kernel_name: &str,
    params: &[KernelParam],
    body_stmts: &[Stmt],
) -> Result<String, syn::Error> {
    let mut cuda = String::new();

    // Generate struct definitions for any Forge types used
    let preamble = generate_struct_preamble(params, body_stmts);
    cuda.push_str(&preamble);

    // Utility device functions (random, etc.)
    cuda.push_str(FORGE_DEVICE_UTILS);

    // Kernel signature
    cuda.push_str("extern \"C\" __global__ void ");
    cuda.push_str(kernel_name);
    cuda.push('(');

    let param_strs: Vec<String> = params
        .iter()
        .map(|p| format!("{} {}", p.cuda_type, p.name))
        .collect();
    cuda.push_str(&param_strs.join(", "));

    cuda.push_str(") {\n");

    // Body
    for stmt in body_stmts {
        let line = emit_stmt(stmt, 1)?;
        cuda.push_str(&line);
    }

    cuda.push_str("}\n");

    Ok(cuda)
}

/// Emit a statement as CUDA C++.
fn emit_stmt(stmt: &Stmt, indent: usize) -> Result<String, syn::Error> {
    let pad = "    ".repeat(indent);
    match stmt {
        Stmt::Local(local) => emit_local(local, &pad),
        Stmt::Expr(expr, semi) => {
            let s = emit_expr(expr)?;
            if semi.is_some() {
                Ok(format!("{}{};\n", pad, s))
            } else {
                // Expression without semicolon (e.g., last expression in block)
                Ok(format!("{}{}\n", pad, s))
            }
        }
        Stmt::Item(_) => Err(syn::Error::new(
            Span::call_site(),
            "item declarations not supported inside #[kernel]",
        )),
        Stmt::Macro(m) => Err(syn::Error::new_spanned(
            &m.mac,
            "macros not supported inside #[kernel]",
        )),
    }
}

/// Emit a let binding.
fn emit_local(local: &Local, pad: &str) -> Result<String, syn::Error> {
    let name = match &local.pat {
        Pat::Ident(PatIdent { ident, .. }) => ident.to_string(),
        Pat::Type(PatType { pat, ty, .. }) => {
            // let x: Type = ...
            let name = match pat.as_ref() {
                Pat::Ident(PatIdent { ident, .. }) => ident.to_string(),
                _ => {
                    return Err(syn::Error::new_spanned(pat, "unsupported pattern"));
                }
            };
            let type_str = type_to_string(ty);
            let cuda_type = scalar_to_cuda(&type_str)
                .unwrap_or_else(|_| "auto");
            if let Some(init) = &local.init {
                let val = emit_expr(&init.expr)?;
                return Ok(format!("{}{} {} = {};\n", pad, cuda_type, name, val));
            } else {
                return Ok(format!("{}{} {};\n", pad, cuda_type, name));
            }
        }
        _ => {
            return Err(syn::Error::new_spanned(
                &local.pat,
                "unsupported pattern in let binding",
            ));
        }
    };

    if let Some(init) = &local.init {
        let val = emit_expr(&init.expr)?;
        // Try to infer type from the expression
        let cuda_type = infer_cuda_type_from_expr(&init.expr).unwrap_or("auto");
        Ok(format!("{}{} {} = {};\n", pad, cuda_type, name, val))
    } else {
        // Uninitialized — need type annotation (handled by Pat::Type above)
        Err(syn::Error::new_spanned(
            &local.pat,
            "let binding without initializer needs a type annotation",
        ))
    }
}

/// Emit an expression as CUDA C++.
fn emit_expr(expr: &Expr) -> Result<String, syn::Error> {
    match expr {
        // Literals
        Expr::Lit(ExprLit { lit, .. }) => emit_lit(lit),

        // Variable references
        Expr::Path(ExprPath { path, .. }) => Ok(path_to_string(path)),

        // Binary operations: a + b, a < b, etc.
        Expr::Binary(ExprBinary {
            left, op, right, ..
        }) => {
            let l = emit_expr(left)?;
            let r = emit_expr(right)?;
            let op_str = binop_to_cuda(op);
            Ok(format!("({} {} {})", l, op_str, r))
        }

        // Unary operations: -x, !x
        Expr::Unary(ExprUnary { op, expr, .. }) => {
            let e = emit_expr(expr)?;
            let op_str = match op {
                UnOp::Neg(_) => "-",
                UnOp::Not(_) => "!",
                UnOp::Deref(_) => "*",
                _ => {
                    return Err(syn::Error::new_spanned(expr, "unsupported unary operator"));
                }
            };
            Ok(format!("({}{})", op_str, e))
        }

        // Function calls: thread_id(), sin(x), Vec3f::new(x,y,z), etc.
        Expr::Call(ExprCall { func, args, .. }) => {
            let fname = emit_expr(func)?;
            // Special case: thread_id() expands to an expression, not a function call
            if fname == "thread_id" {
                return Ok("(blockIdx.x * blockDim.x + threadIdx.x)".to_string());
            }
            // Special case: Vec3f::new(x, y, z) → forge_vec3f{x, y, z}
            if let Expr::Path(ExprPath { path, .. }) = func.as_ref() {
                if path.segments.len() == 2 {
                    let type_name = path.segments[0].ident.to_string();
                    let method = path.segments[1].ident.to_string();
                    if let Some(info) = forge_type_info(&type_name) {
                        if method == "new" {
                            let arg_strs: Result<Vec<String>, _> = args.iter().map(emit_expr).collect();
                            return Ok(format!("{}{{{}}}",
                                info.cuda_name,
                                arg_strs?.join(", ")
                            ));
                        } else if method == "zero" || method == "zeros" {
                            let zeros = vec!["0.0f"; info.components].join(", ");
                            return Ok(format!("{}{{{}}}", info.cuda_name, zeros));
                        } else if method == "splat" {
                            if let Some(arg) = args.first() {
                                let val = emit_expr(arg)?;
                                let vals = vec![val.as_str(); info.components].join(", ");
                                return Ok(format!("{}{{{}}}", info.cuda_name, vals));
                            }
                        }
                    }
                }
            }
            let cuda_fname = builtin_to_cuda(&fname);
            let arg_strs: Result<Vec<String>, _> = args.iter().map(emit_expr).collect();
            Ok(format!("{}({})", cuda_fname, arg_strs?.join(", ")))
        }

        // Method calls: x.abs(), etc.
        Expr::MethodCall(ExprMethodCall {
            receiver,
            method,
            args,
            ..
        }) => {
            let recv = emit_expr(receiver)?;
            let method_name = method.to_string();
            // Map common methods
            match method_name.as_str() {
                "abs" => Ok(format!("fabsf({})", recv)),
                "sqrt" => Ok(format!("sqrtf({})", recv)),
                "sin" => Ok(format!("sinf({})", recv)),
                "cos" => Ok(format!("cosf({})", recv)),
                "min" => {
                    if let Some(arg) = args.first() {
                        let a = emit_expr(arg)?;
                        Ok(format!("fminf({}, {})", recv, a))
                    } else {
                        Err(syn::Error::new_spanned(method, "min requires an argument"))
                    }
                }
                "max" => {
                    if let Some(arg) = args.first() {
                        let a = emit_expr(arg)?;
                        Ok(format!("fmaxf({}, {})", recv, a))
                    } else {
                        Err(syn::Error::new_spanned(method, "max requires an argument"))
                    }
                }
                _ => {
                    let arg_strs: Result<Vec<String>, _> = args.iter().map(emit_expr).collect();
                    Ok(format!("{}.{}({})", recv, method_name, arg_strs?.join(", ")))
                }
            }
        }

        // Array indexing: data[i]
        Expr::Index(ExprIndex { expr, index, .. }) => {
            let arr = emit_expr(expr)?;
            let idx = emit_expr(index)?;
            Ok(format!("{}[{}]", arr, idx))
        }

        // Field access: v.x, v.y
        Expr::Field(ExprField { base, member, .. }) => {
            let b = emit_expr(base)?;
            Ok(format!("{}.{}", b, member.to_token_stream()))
        }

        // Assignment: x = ...
        Expr::Assign(ExprAssign { left, right, .. }) => {
            let l = emit_expr(left)?;
            let r = emit_expr(right)?;
            Ok(format!("{} = {}", l, r))
        }

        // Parenthesized: (expr)
        Expr::Paren(ExprParen { expr, .. }) => {
            let e = emit_expr(expr)?;
            Ok(format!("({})", e))
        }

        // If expression
        Expr::If(ExprIf {
            cond,
            then_branch,
            else_branch,
            ..
        }) => emit_if(cond, then_branch, else_branch.as_ref().map(|(_, e)| e.as_ref())),

        // Block expression
        Expr::Block(ExprBlock { block, .. }) => {
            let mut s = String::new();
            s.push_str("{\n");
            for stmt in &block.stmts {
                s.push_str(&emit_stmt(stmt, 2)?);
            }
            s.push_str("    }");
            Ok(s)
        }

        // Return
        Expr::Return(ExprReturn { expr, .. }) => {
            if let Some(e) = expr {
                let val = emit_expr(e)?;
                Ok(format!("return {}", val))
            } else {
                Ok("return".to_string())
            }
        }

        // Cast: x as f32
        Expr::Cast(ExprCast { expr, ty, .. }) => {
            let e = emit_expr(expr)?;
            let type_str = type_to_string(ty);
            let cuda_type = scalar_to_cuda(&type_str)
                .map_err(|_| syn::Error::new_spanned(ty, "unsupported cast type"))?;
            Ok(format!("(({}){})", cuda_type, e))
        }

        // For loop (only range-based for now)
        Expr::ForLoop(for_loop) => emit_for_loop(for_loop),

        // While loop
        Expr::While(while_loop) => {
            let cond = emit_expr(&while_loop.cond)?;
            let mut s = format!("while ({}) {{\n", cond);
            for stmt in &while_loop.body.stmts {
                s.push_str(&emit_stmt(stmt, 2)?);
            }
            s.push_str("    }");
            Ok(s)
        }

        _ => Err(syn::Error::new_spanned(
            expr,
            format!(
                "unsupported expression in #[kernel]: {}",
                quote::quote!(#expr)
            ),
        )),
    }
}

/// Emit an if/else as CUDA.
fn emit_if(
    cond: &Expr,
    then_block: &syn::Block,
    else_branch: Option<&Expr>,
) -> Result<String, syn::Error> {
    let cond_str = emit_expr(cond)?;
    let mut s = format!("if ({}) {{\n", cond_str);
    for stmt in &then_block.stmts {
        s.push_str(&emit_stmt(stmt, 2)?);
    }
    s.push_str("    }");

    if let Some(else_expr) = else_branch {
        match else_expr {
            Expr::Block(ExprBlock { block, .. }) => {
                s.push_str(" else {\n");
                for stmt in &block.stmts {
                    s.push_str(&emit_stmt(stmt, 2)?);
                }
                s.push_str("    }");
            }
            Expr::If(ExprIf {
                cond,
                then_branch,
                else_branch,
                ..
            }) => {
                s.push_str(" else ");
                s.push_str(&emit_if(
                    cond,
                    then_branch,
                    else_branch.as_ref().map(|(_, e)| e.as_ref()),
                )?);
            }
            _ => {
                return Err(syn::Error::new_spanned(else_expr, "unsupported else branch"));
            }
        }
    }

    Ok(s)
}

/// Emit a for loop (range-based).
fn emit_for_loop(for_loop: &syn::ExprForLoop) -> Result<String, syn::Error> {
    let var = match &*for_loop.pat {
        Pat::Ident(PatIdent { ident, .. }) => ident.to_string(),
        _ => {
            return Err(syn::Error::new_spanned(
                &for_loop.pat,
                "only simple identifiers supported in for loops",
            ));
        }
    };

    // Try to parse range: start..end
    let iter_expr = &for_loop.expr;
    if let Expr::Range(range) = iter_expr.as_ref() {
        let start = range
            .start
            .as_ref()
            .map(|e| emit_expr(e))
            .transpose()?
            .unwrap_or_else(|| "0".to_string());
        let end = range
            .end
            .as_ref()
            .map(|e| emit_expr(e))
            .ok_or_else(|| {
                syn::Error::new_spanned(iter_expr, "for loop range must have an end bound")
            })??;

        let mut s = format!(
            "for (int {} = {}; {} < {}; {}++) {{\n",
            var, start, var, end, var
        );
        for stmt in &for_loop.body.stmts {
            s.push_str(&emit_stmt(stmt, 2)?);
        }
        s.push_str("    }");
        return Ok(s);
    }

    Err(syn::Error::new_spanned(
        iter_expr,
        "only range expressions (start..end) supported in #[kernel] for loops",
    ))
}

/// Convert a binary operator to CUDA C++.
fn binop_to_cuda(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add(_) => "+",
        BinOp::Sub(_) => "-",
        BinOp::Mul(_) => "*",
        BinOp::Div(_) => "/",
        BinOp::Rem(_) => "%",
        BinOp::And(_) => "&&",
        BinOp::Or(_) => "||",
        BinOp::BitXor(_) => "^",
        BinOp::BitAnd(_) => "&",
        BinOp::BitOr(_) => "|",
        BinOp::Shl(_) => "<<",
        BinOp::Shr(_) => ">>",
        BinOp::Eq(_) => "==",
        BinOp::Lt(_) => "<",
        BinOp::Le(_) => "<=",
        BinOp::Ne(_) => "!=",
        BinOp::Ge(_) => ">=",
        BinOp::Gt(_) => ">",
        BinOp::AddAssign(_) => "+=",
        BinOp::SubAssign(_) => "-=",
        BinOp::MulAssign(_) => "*=",
        BinOp::DivAssign(_) => "/=",
        BinOp::RemAssign(_) => "%=",
        BinOp::BitXorAssign(_) => "^=",
        BinOp::BitAndAssign(_) => "&=",
        BinOp::BitOrAssign(_) => "|=",
        BinOp::ShlAssign(_) => "<<=",
        BinOp::ShrAssign(_) => ">>=",
        _ => "/* unsupported op */",
    }
}

/// Public version of builtin_to_cuda for use by autodiff module.
pub fn builtin_to_cuda_pub(name: &str) -> String {
    builtin_to_cuda(name)
}

/// Map Rust builtin function names to CUDA equivalents.
fn builtin_to_cuda(name: &str) -> String {
    match name {
        "thread_id" => "blockIdx.x * blockDim.x + threadIdx.x".to_string(),
        // Trig
        "sin" | "sinf" => "sinf".to_string(),
        "cos" | "cosf" => "cosf".to_string(),
        "tan" | "tanf" => "tanf".to_string(),
        "asin" | "asinf" => "asinf".to_string(),
        "acos" | "acosf" => "acosf".to_string(),
        "atan" | "atanf" => "atanf".to_string(),
        "atan2" => "atan2f".to_string(),
        "sinh" => "sinhf".to_string(),
        "cosh" => "coshf".to_string(),
        "tanh" => "tanhf".to_string(),
        // Exponential / Log
        "exp" => "expf".to_string(),
        "exp2" => "exp2f".to_string(),
        "log" => "logf".to_string(),
        "log2" => "log2f".to_string(),
        "log10" => "log10f".to_string(),
        "pow" => "powf".to_string(),
        "sqrt" | "sqrtf" => "sqrtf".to_string(),
        "rsqrt" => "rsqrtf".to_string(),
        "cbrt" => "cbrtf".to_string(),
        // Rounding
        "abs" | "fabsf" => "fabsf".to_string(),
        "floor" => "floorf".to_string(),
        "ceil" => "ceilf".to_string(),
        "round" => "roundf".to_string(),
        "trunc" => "truncf".to_string(),
        "frac" | "fract" => "fractf".to_string(),  // CUDA: fractf or x - floorf(x)
        // Min/Max/Clamp
        "min" | "fminf" => "fminf".to_string(),
        "max" | "fmaxf" => "fmaxf".to_string(),
        "clamp" => "fminf(fmaxf".to_string(), // NOTE: handled specially below
        // Special
        "erf" => "erff".to_string(),
        "erfc" => "erfcf".to_string(),
        "sign" | "copysign" => "copysignf".to_string(),
        "isnan" => "isnan".to_string(),
        "isinf" => "isinf".to_string(),
        "isfinite" => "isfinite".to_string(),
        // Atomics
        "atomic_add" => "atomicAdd".to_string(),
        "atomic_min" => "atomicMin".to_string(),
        "atomic_max" => "atomicMax".to_string(),
        "atomic_cas" | "atomic_compare_exchange" => "atomicCAS".to_string(),
        "atomic_exch" | "atomic_exchange" => "atomicExch".to_string(),
        "atomic_sub" => "atomicSub".to_string(),
        "atomic_and" => "atomicAnd".to_string(),
        "atomic_or" => "atomicOr".to_string(),
        "atomic_xor" => "atomicXor".to_string(),
        // Warp primitives
        "warp_shfl" | "shfl_sync" => "__shfl_sync".to_string(),
        "warp_shfl_down" | "shfl_down_sync" => "__shfl_down_sync".to_string(),
        "warp_shfl_up" | "shfl_up_sync" => "__shfl_up_sync".to_string(),
        "warp_shfl_xor" | "shfl_xor_sync" => "__shfl_xor_sync".to_string(),
        "warp_ballot" | "ballot_sync" => "__ballot_sync".to_string(),
        "warp_all" | "all_sync" => "__all_sync".to_string(),
        "warp_any" | "any_sync" => "__any_sync".to_string(),
        // Random (xorshift-based, state is u32)
        "rand_init" => "forge_rand_init".to_string(),
        "rand_f32" | "randf" => "forge_randf".to_string(),
        "rand_u32" | "randi" => "forge_randi".to_string(),
        // Sync
        "syncthreads" | "sync_threads" => "__syncthreads".to_string(),
        "threadfence" | "thread_fence" => "__threadfence".to_string(),
        // Thread indices (1D-3D)
        "block_id" | "block_idx" => "blockIdx.x".to_string(),
        "block_id_y" | "block_idx_y" => "blockIdx.y".to_string(),
        "block_id_z" | "block_idx_z" => "blockIdx.z".to_string(),
        "block_dim" | "block_size" => "blockDim.x".to_string(),
        "block_dim_y" => "blockDim.y".to_string(),
        "block_dim_z" => "blockDim.z".to_string(),
        "thread_idx" | "local_thread_id" => "threadIdx.x".to_string(),
        "thread_idx_y" => "threadIdx.y".to_string(),
        "thread_idx_z" => "threadIdx.z".to_string(),
        "grid_dim" | "grid_size" => "gridDim.x".to_string(),
        "grid_dim_y" => "gridDim.y".to_string(),
        "grid_dim_z" => "gridDim.z".to_string(),
        // Global thread IDs (convenience)
        "tid_x" => "(blockIdx.x * blockDim.x + threadIdx.x)".to_string(),
        "tid_y" => "(blockIdx.y * blockDim.y + threadIdx.y)".to_string(),
        "tid_z" => "(blockIdx.z * blockDim.z + threadIdx.z)".to_string(),
        // Passthrough
        other => other.to_string(),
    }
}

/// Convert a syn::Path to a string.
fn path_to_string(path: &syn::Path) -> String {
    path.segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}

/// Emit a literal value.
fn emit_lit(lit: &Lit) -> Result<String, syn::Error> {
    match lit {
        Lit::Int(i) => {
            let s = i.to_string();
            // Strip Rust suffixes for CUDA
            let cleaned = s
                .trim_end_matches("i32")
                .trim_end_matches("i64")
                .trim_end_matches("u32")
                .trim_end_matches("u64")
                .trim_end_matches("usize")
                .trim_end_matches("isize");
            Ok(cleaned.to_string())
        }
        Lit::Float(f) => {
            let s = f.to_string();
            let cleaned = s.trim_end_matches("f32").trim_end_matches("f64");
            // Ensure float literal has 'f' suffix for CUDA float
            if s.contains("f64") {
                Ok(cleaned.to_string()) // double literal, no suffix
            } else if cleaned.contains('.') {
                Ok(format!("{}f", cleaned))
            } else {
                Ok(format!("{}.0f", cleaned))
            }
        }
        Lit::Bool(b) => Ok(if b.value { "true" } else { "false" }.to_string()),
        _ => Err(syn::Error::new_spanned(lit, "unsupported literal type")),
    }
}

// Use quote's ToTokens for Field member
use quote::ToTokens;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_to_cuda() {
        assert_eq!(scalar_to_cuda("f32").unwrap(), "float");
        assert_eq!(scalar_to_cuda("f64").unwrap(), "double");
        assert_eq!(scalar_to_cuda("f16").unwrap(), "__half");
        assert_eq!(scalar_to_cuda("i8").unwrap(), "signed char");
        assert_eq!(scalar_to_cuda("i16").unwrap(), "short");
        assert_eq!(scalar_to_cuda("i32").unwrap(), "int");
        assert_eq!(scalar_to_cuda("u8").unwrap(), "unsigned char");
        assert_eq!(scalar_to_cuda("u16").unwrap(), "unsigned short");
        assert_eq!(scalar_to_cuda("u32").unwrap(), "unsigned int");
        assert_eq!(scalar_to_cuda("bool").unwrap(), "bool");
    }

    #[test]
    fn test_builtin_mapping() {
        assert_eq!(builtin_to_cuda("sin"), "sinf");
        assert_eq!(builtin_to_cuda("cos"), "cosf");
        assert_eq!(builtin_to_cuda("sqrt"), "sqrtf");
        assert_eq!(builtin_to_cuda("thread_id"), "blockIdx.x * blockDim.x + threadIdx.x");
        // Atomics
        assert_eq!(builtin_to_cuda("atomic_add"), "atomicAdd");
        assert_eq!(builtin_to_cuda("atomic_min"), "atomicMin");
        assert_eq!(builtin_to_cuda("atomic_max"), "atomicMax");
        assert_eq!(builtin_to_cuda("atomic_cas"), "atomicCAS");
        // Math
        assert_eq!(builtin_to_cuda("erf"), "erff");
        assert_eq!(builtin_to_cuda("cbrt"), "cbrtf");
        assert_eq!(builtin_to_cuda("rsqrt"), "rsqrtf");
        assert_eq!(builtin_to_cuda("tanh"), "tanhf");
        // Warp
        assert_eq!(builtin_to_cuda("warp_shfl_down"), "__shfl_down_sync");
        assert_eq!(builtin_to_cuda("syncthreads"), "__syncthreads");
    }

    #[test]
    fn test_binop_to_cuda() {
        assert_eq!(binop_to_cuda(&BinOp::Add(syn::token::Plus::default())), "+");
        assert_eq!(binop_to_cuda(&BinOp::Lt(syn::token::Lt::default())), "<");
        assert_eq!(binop_to_cuda(&BinOp::AddAssign(syn::token::PlusEq::default())), "+=");
    }
}
