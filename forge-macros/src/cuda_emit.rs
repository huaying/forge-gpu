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
                            let cuda_elem = scalar_to_cuda(&elem_type)?;
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
        "f32" => Ok("float"),
        "f64" => Ok("double"),
        "i32" => Ok("int"),
        "i64" => Ok("long long"),
        "u32" => Ok("unsigned int"),
        "u64" => Ok("unsigned long long"),
        "bool" => Ok("bool"),
        "usize" => Ok("unsigned long long"),
        other => Err(syn::Error::new(
            Span::call_site(),
            format!("unsupported scalar type '{}' in #[kernel]", other),
        )),
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
pub fn generate_cuda_source(
    kernel_name: &str,
    params: &[KernelParam],
    body_stmts: &[Stmt],
) -> Result<String, syn::Error> {
    let mut cuda = String::new();

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

        // Function calls: thread_id(), sin(x), etc.
        Expr::Call(ExprCall { func, args, .. }) => {
            let fname = emit_expr(func)?;
            // Special case: thread_id() expands to an expression, not a function call
            if fname == "thread_id" {
                return Ok("(blockIdx.x * blockDim.x + threadIdx.x)".to_string());
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

/// Map Rust builtin function names to CUDA equivalents.
fn builtin_to_cuda(name: &str) -> String {
    match name {
        "thread_id" => "blockIdx.x * blockDim.x + threadIdx.x".to_string(),
        "sin" | "sinf" => "sinf".to_string(),
        "cos" | "cosf" => "cosf".to_string(),
        "sqrt" | "sqrtf" => "sqrtf".to_string(),
        "abs" | "fabsf" => "fabsf".to_string(),
        "min" | "fminf" => "fminf".to_string(),
        "max" | "fmaxf" => "fmaxf".to_string(),
        "floor" => "floorf".to_string(),
        "ceil" => "ceilf".to_string(),
        "round" => "roundf".to_string(),
        "exp" => "expf".to_string(),
        "log" => "logf".to_string(),
        "pow" => "powf".to_string(),
        "atan2" => "atan2f".to_string(),
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
        assert_eq!(scalar_to_cuda("i32").unwrap(), "int");
        assert_eq!(scalar_to_cuda("u32").unwrap(), "unsigned int");
        assert_eq!(scalar_to_cuda("bool").unwrap(), "bool");
    }

    #[test]
    fn test_builtin_mapping() {
        assert_eq!(builtin_to_cuda("sin"), "sinf");
        assert_eq!(builtin_to_cuda("cos"), "cosf");
        assert_eq!(builtin_to_cuda("sqrt"), "sqrtf");
        assert_eq!(builtin_to_cuda("thread_id"), "blockIdx.x * blockDim.x + threadIdx.x");
    }

    #[test]
    fn test_binop_to_cuda() {
        assert_eq!(binop_to_cuda(&BinOp::Add(syn::token::Plus::default())), "+");
        assert_eq!(binop_to_cuda(&BinOp::Lt(syn::token::Lt::default())), "<");
        assert_eq!(binop_to_cuda(&BinOp::AddAssign(syn::token::PlusEq::default())), "+=");
    }
}
