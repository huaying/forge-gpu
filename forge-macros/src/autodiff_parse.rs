//! Parse Rust kernel AST into ForwardOp IR for autodiff.
//!
//! This module walks the kernel body statements and produces
//! a flat list of ForwardOp operations suitable for adjoint generation.

use syn::{
    BinOp, Expr, ExprAssign, ExprBinary, ExprCall, ExprField, ExprIf,
    ExprIndex, ExprLit, ExprMethodCall, ExprParen, ExprPath, ExprUnary,
    Lit, Local, Pat, PatIdent, PatType, Stmt, UnOp,
};
use quote::ToTokens;

use crate::autodiff::{BinOpKind, ForwardOp};
use crate::cuda_emit::builtin_to_cuda_pub;

/// Counter for generating unique SSA variable names.
struct SsaCounter {
    next: usize,
}

impl SsaCounter {
    fn new() -> Self {
        Self { next: 0 }
    }

    fn fresh(&mut self, prefix: &str) -> String {
        let name = format!("{}_{}", prefix, self.next);
        self.next += 1;
        name
    }
}

/// Parse a list of statements into ForwardOp IR.
pub fn parse_stmts_to_ops(stmts: &[Stmt]) -> Vec<ForwardOp> {
    let mut ops = Vec::new();
    let mut ctr = SsaCounter::new();
    for stmt in stmts {
        parse_stmt(stmt, &mut ops, &mut ctr);
    }
    ops
}

fn parse_stmt(stmt: &Stmt, ops: &mut Vec<ForwardOp>, ctr: &mut SsaCounter) {
    match stmt {
        Stmt::Local(local) => parse_local(local, ops, ctr),
        Stmt::Expr(expr, _semi) => {
            parse_expr_as_stmt(expr, ops, ctr);
        }
        _ => {}
    }
}

fn parse_local(local: &Local, ops: &mut Vec<ForwardOp>, ctr: &mut SsaCounter) {
    let var_name = match &local.pat {
        Pat::Ident(PatIdent { ident, .. }) => ident.to_string(),
        Pat::Type(PatType { pat, .. }) => {
            if let Pat::Ident(PatIdent { ident, .. }) = pat.as_ref() {
                ident.to_string()
            } else {
                return;
            }
        }
        _ => return,
    };

    if let Some(init) = &local.init {
        let result = parse_expr_to_value(&init.expr, ops, ctr);
        match result {
            ExprValue::Var(v) if v == var_name => {
                // Self-assignment, skip
            }
            ExprValue::Var(v) => {
                ops.push(ForwardOp::Assign { var: var_name, value: v });
            }
            ExprValue::ThreadId => {
                ops.push(ForwardOp::ThreadId { var: var_name });
            }
            ExprValue::Literal(val, ty) => {
                ops.push(ForwardOp::Literal { var: var_name, value: val, cuda_type: ty });
            }
            ExprValue::Op(op) => {
                // Replace the var name in the op
                let op = replace_op_var(op, &var_name);
                ops.push(op);
            }
        }
    }
}

/// Represents the result of parsing an expression.
enum ExprValue {
    Var(String),
    ThreadId,
    Literal(String, String),
    Op(ForwardOp),
}

fn parse_expr_to_value(expr: &Expr, ops: &mut Vec<ForwardOp>, ctr: &mut SsaCounter) -> ExprValue {
    match expr {
        Expr::Path(ExprPath { path, .. }) => {
            let name = path.segments.iter().map(|s| s.ident.to_string()).collect::<Vec<_>>().join("::");
            ExprValue::Var(name)
        }

        Expr::Lit(ExprLit { lit, .. }) => {
            let (val, ty) = match lit {
                Lit::Float(f) => (f.to_string().trim_end_matches("f32").trim_end_matches("f64").to_string(), "float".to_string()),
                Lit::Int(i) => (i.to_string().trim_end_matches("i32").trim_end_matches("u32").to_string(), "int".to_string()),
                Lit::Bool(b) => (if b.value { "true" } else { "false" }.to_string(), "bool".to_string()),
                _ => ("0".to_string(), "float".to_string()),
            };
            ExprValue::Literal(val, ty)
        }

        Expr::Call(ExprCall { func, args, .. }) => {
            if let Expr::Path(ExprPath { path, .. }) = func.as_ref() {
                let fname = path.segments.iter().map(|s| s.ident.to_string()).collect::<Vec<_>>().join("::");
                if fname == "thread_id" {
                    return ExprValue::ThreadId;
                }
                // Unary builtin: sin(x), cos(x), etc.
                if args.len() == 1 {
                    let arg_val = parse_expr_to_value(&args[0], ops, ctr);
                    let arg_name = ensure_named(arg_val, ops, ctr, "arg");
                    let cuda_func = builtin_to_cuda_pub(&fname);
                    let tmp = ctr.fresh("t");
                    return ExprValue::Op(ForwardOp::UnaryFunc {
                        var: tmp,
                        func: cuda_func,
                        arg: arg_name,
                    });
                }
            }
            // Fallback: treat as opaque
            let tmp = ctr.fresh("t");
            ExprValue::Op(ForwardOp::Literal { var: tmp, value: "/* complex expr */".to_string(), cuda_type: "float".to_string() })
        }

        Expr::Binary(ExprBinary { left, op, right, .. }) => {
            let left_val = parse_expr_to_value(left, ops, ctr);
            let right_val = parse_expr_to_value(right, ops, ctr);
            let left_name = ensure_named(left_val, ops, ctr, "l");
            let right_name = ensure_named(right_val, ops, ctr, "r");

            if let Some(kind) = binop_to_kind(op) {
                let tmp = ctr.fresh("t");
                ExprValue::Op(ForwardOp::BinOp {
                    var: tmp,
                    left: left_name,
                    op: kind,
                    right: right_name,
                })
            } else {
                // Comparison or logical op — format as CUDA string (no gradient)
                let op_str = match op {
                    BinOp::Lt(_) => "<",
                    BinOp::Le(_) => "<=",
                    BinOp::Gt(_) => ">",
                    BinOp::Ge(_) => ">=",
                    BinOp::Eq(_) => "==",
                    BinOp::Ne(_) => "!=",
                    BinOp::And(_) => "&&",
                    BinOp::Or(_) => "||",
                    _ => "?",
                };
                // Return as a "variable" that's really a condition expression
                let tmp = ctr.fresh("cmp");
                let cond_expr = format!("({} {} {})", left_name, op_str, right_name);
                ExprValue::Op(ForwardOp::Literal {
                    var: tmp,
                    value: cond_expr,
                    cuda_type: "bool".to_string(),
                })
            }
        }

        Expr::Unary(ExprUnary { op: UnOp::Neg(_), expr, .. }) => {
            let val = parse_expr_to_value(expr, ops, ctr);
            let name = ensure_named(val, ops, ctr, "neg");
            let tmp = ctr.fresh("t");
            // -x = 0 - x
            let zero = ctr.fresh("zero");
            ops.push(ForwardOp::Literal { var: zero.clone(), value: "0.0f".to_string(), cuda_type: "float".to_string() });
            ExprValue::Op(ForwardOp::BinOp {
                var: tmp,
                left: zero,
                op: BinOpKind::Sub,
                right: name,
            })
        }

        Expr::Index(ExprIndex { expr, index, .. }) => {
            let arr = parse_expr_to_value(expr, ops, ctr);
            let idx = parse_expr_to_value(index, ops, ctr);
            let arr_name = ensure_named(arr, ops, ctr, "arr");
            let idx_name = ensure_named(idx, ops, ctr, "idx");
            let tmp = ctr.fresh("t");
            ExprValue::Op(ForwardOp::ArrayRead {
                var: tmp,
                array: arr_name,
                index: idx_name,
            })
        }

        Expr::Field(ExprField { base, member, .. }) => {
            let base_val = parse_expr_to_value(base, ops, ctr);
            let base_name = ensure_named(base_val, ops, ctr, "base");
            let tmp = ctr.fresh("t");
            ExprValue::Op(ForwardOp::FieldAccess {
                var: tmp,
                base: base_name,
                field: member.to_token_stream().to_string(),
            })
        }

        Expr::Paren(ExprParen { expr, .. }) => {
            parse_expr_to_value(expr, ops, ctr)
        }

        Expr::MethodCall(ExprMethodCall { receiver, method, args, .. }) => {
            let recv_val = parse_expr_to_value(receiver, ops, ctr);
            let recv_name = ensure_named(recv_val, ops, ctr, "recv");
            let method_name = method.to_string();

            // Map method calls to CUDA builtins
            let cuda_func = match method_name.as_str() {
                "abs" => "fabsf",
                "sqrt" => "sqrtf",
                "sin" => "sinf",
                "cos" => "cosf",
                _ => {
                    // For min/max with args, treat as binary
                    if (method_name == "min" || method_name == "max") && args.len() == 1 {
                        let arg_val = parse_expr_to_value(&args[0], ops, ctr);
                        let arg_name = ensure_named(arg_val, ops, ctr, "arg");
                        let tmp = ctr.fresh("t");
                        let func = if method_name == "min" { "fminf" } else { "fmaxf" };
                        return ExprValue::Op(ForwardOp::UnaryFunc {
                            var: tmp,
                            func: func.to_string(),
                            arg: format!("{}, {}", recv_name, arg_name), // not ideal but works
                        });
                    }
                    return ExprValue::Var(recv_name);
                }
            };

            let tmp = ctr.fresh("t");
            ExprValue::Op(ForwardOp::UnaryFunc {
                var: tmp,
                func: cuda_func.to_string(),
                arg: recv_name,
            })
        }

        _ => {
            let tmp = ctr.fresh("t");
            ExprValue::Op(ForwardOp::Literal {
                var: tmp,
                value: "/* unsupported expr */".to_string(),
                cuda_type: "float".to_string(),
            })
        }
    }
}

/// Handle expression-level statements (assignments, compound assignments, if, etc.)
fn parse_expr_as_stmt(expr: &Expr, ops: &mut Vec<ForwardOp>, ctr: &mut SsaCounter) {
    match expr {
        Expr::Assign(ExprAssign { left, right, .. }) => {
            // arr[i] = val or var = val
            if let Expr::Index(ExprIndex { expr: arr_expr, index, .. }) = left.as_ref() {
                let arr_val = parse_expr_to_value(arr_expr, ops, ctr);
                let idx_val = parse_expr_to_value(index, ops, ctr);
                let val = parse_expr_to_value(right, ops, ctr);
                let arr_name = ensure_named(arr_val, ops, ctr, "arr");
                let idx_name = ensure_named(idx_val, ops, ctr, "idx");
                let val_name = ensure_named(val, ops, ctr, "val");
                ops.push(ForwardOp::ArrayWrite {
                    array: arr_name,
                    index: idx_name,
                    value: val_name,
                });
            } else {
                let lhs = parse_expr_to_value(left, ops, ctr);
                let rhs = parse_expr_to_value(right, ops, ctr);
                let lhs_name = ensure_named(lhs, ops, ctr, "lhs");
                let rhs_name = ensure_named(rhs, ops, ctr, "rhs");
                ops.push(ForwardOp::Assign { var: lhs_name, value: rhs_name });
            }
        }

        Expr::Binary(ExprBinary { left, op, right, .. }) => {
            // Compound assignment: arr[i] += val
            if let Some(kind) = compound_op_to_kind(op) {
                if let Expr::Index(ExprIndex { expr: arr_expr, index, .. }) = left.as_ref() {
                    let arr_val = parse_expr_to_value(arr_expr, ops, ctr);
                    let idx_val = parse_expr_to_value(index, ops, ctr);
                    let val = parse_expr_to_value(right, ops, ctr);
                    let arr_name = ensure_named(arr_val, ops, ctr, "arr");
                    let idx_name = ensure_named(idx_val, ops, ctr, "idx");
                    let val_name = ensure_named(val, ops, ctr, "val");
                    ops.push(ForwardOp::ArrayCompound {
                        array: arr_name,
                        index: idx_name,
                        op: kind,
                        value: val_name,
                    });
                    return;
                }
            }
            // Regular binary expression used as statement
            let _ = parse_expr_to_value(expr, ops, ctr);
        }

        Expr::If(ExprIf { cond, then_branch, else_branch, .. }) => {
            let cond_val = parse_expr_to_value(cond, ops, ctr);
            let cond_name = ensure_named(cond_val, ops, ctr, "cond");
            let then_ops = parse_stmts_to_ops(&then_branch.stmts);
            let else_ops = if let Some((_, else_expr)) = else_branch {
                if let Expr::Block(block) = else_expr.as_ref() {
                    parse_stmts_to_ops(&block.block.stmts)
                } else {
                    vec![]
                }
            } else {
                vec![]
            };
            ops.push(ForwardOp::IfBlock {
                cond: cond_name,
                then_ops,
                else_ops,
            });
        }

        _ => {
            let _ = parse_expr_to_value(expr, ops, ctr);
        }
    }
}

/// Ensure an ExprValue has a named variable. If it's an Op, push it and return the var name.
fn ensure_named(val: ExprValue, ops: &mut Vec<ForwardOp>, ctr: &mut SsaCounter, hint: &str) -> String {
    match val {
        ExprValue::Var(name) => name,
        ExprValue::ThreadId => {
            let name = ctr.fresh(hint);
            ops.push(ForwardOp::ThreadId { var: name.clone() });
            name
        }
        ExprValue::Literal(v, ty) => {
            let name = ctr.fresh(hint);
            ops.push(ForwardOp::Literal { var: name.clone(), value: v, cuda_type: ty });
            name
        }
        ExprValue::Op(op) => {
            let name = get_op_var(&op).to_string();
            ops.push(op);
            name
        }
    }
}

fn get_op_var(op: &ForwardOp) -> &str {
    match op {
        ForwardOp::ThreadId { var } => var,
        ForwardOp::Literal { var, .. } => var,
        ForwardOp::BinOp { var, .. } => var,
        ForwardOp::UnaryFunc { var, .. } => var,
        ForwardOp::ArrayRead { var, .. } => var,
        ForwardOp::FieldAccess { var, .. } => var,
        ForwardOp::Assign { var, .. } => var,
        ForwardOp::ArrayWrite { .. } => "",
        ForwardOp::ArrayCompound { .. } => "",
        ForwardOp::IfBlock { .. } => "",
    }
}

fn replace_op_var(op: ForwardOp, new_var: &str) -> ForwardOp {
    match op {
        ForwardOp::BinOp { left, op, right, .. } => ForwardOp::BinOp { var: new_var.to_string(), left, op, right },
        ForwardOp::UnaryFunc { func, arg, .. } => ForwardOp::UnaryFunc { var: new_var.to_string(), func, arg },
        ForwardOp::ArrayRead { array, index, .. } => ForwardOp::ArrayRead { var: new_var.to_string(), array, index },
        ForwardOp::FieldAccess { base, field, .. } => ForwardOp::FieldAccess { var: new_var.to_string(), base, field },
        ForwardOp::Literal { value, cuda_type, .. } => ForwardOp::Literal { var: new_var.to_string(), value, cuda_type },
        ForwardOp::ThreadId { .. } => ForwardOp::ThreadId { var: new_var.to_string() },
        other => other,
    }
}

fn binop_to_kind(op: &BinOp) -> Option<BinOpKind> {
    match op {
        BinOp::Add(_) => Some(BinOpKind::Add),
        BinOp::Sub(_) => Some(BinOpKind::Sub),
        BinOp::Mul(_) => Some(BinOpKind::Mul),
        BinOp::Div(_) => Some(BinOpKind::Div),
        BinOp::Rem(_) => Some(BinOpKind::Rem),
        _ => None, // comparisons etc. don't have arithmetic gradients
    }
}

fn compound_op_to_kind(op: &BinOp) -> Option<BinOpKind> {
    match op {
        BinOp::AddAssign(_) => Some(BinOpKind::Add),
        BinOp::SubAssign(_) => Some(BinOpKind::Sub),
        BinOp::MulAssign(_) => Some(BinOpKind::Mul),
        BinOp::DivAssign(_) => Some(BinOpKind::Div),
        _ => None,
    }
}
