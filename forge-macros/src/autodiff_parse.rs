//! Parse Rust kernel AST into ForwardOp IR for autodiff.

use syn::{
    BinOp, Expr, ExprAssign, ExprBinary, ExprCall, ExprField, ExprIf,
    ExprIndex, ExprLit, ExprMethodCall, ExprParen, ExprPath, ExprUnary,
    Lit, Local, Pat, PatIdent, PatType, Stmt, UnOp,
};
use quote::ToTokens;

use crate::autodiff::{BinOpKind, ForwardOp};
use crate::cuda_emit::{builtin_to_cuda_pub, forge_type_info};

struct SsaCounter { next: usize }
impl SsaCounter {
    fn new() -> Self { Self { next: 0 } }
    fn fresh(&mut self, prefix: &str) -> String {
        let n = format!("{}_{}", prefix, self.next);
        self.next += 1;
        n
    }
}

struct TypeCtx { types: std::collections::HashMap<String, String> }
impl TypeCtx {
    fn new() -> Self { Self { types: std::collections::HashMap::new() } }
    fn set(&mut self, var: &str, t: &str) { self.types.insert(var.to_string(), t.to_string()); }
    fn get(&self, var: &str) -> String { self.types.get(var).cloned().unwrap_or("float".to_string()) }
}

pub fn parse_stmts_to_ops(stmts: &[Stmt]) -> Vec<ForwardOp> {
    let mut ops = Vec::new();
    let mut ctr = SsaCounter::new();
    let mut ctx = TypeCtx::new();
    for s in stmts { parse_stmt(s, &mut ops, &mut ctr, &mut ctx); }
    ops
}

pub fn parse_stmts_to_ops_with_types(
    stmts: &[Stmt], param_types: &std::collections::HashMap<String, String>,
) -> Vec<ForwardOp> {
    let mut ops = Vec::new();
    let mut ctr = SsaCounter::new();
    let mut ctx = TypeCtx::new();
    for (k, v) in param_types { ctx.set(k, v); }
    for s in stmts { parse_stmt(s, &mut ops, &mut ctr, &mut ctx); }
    ops
}

fn parse_stmt(stmt: &Stmt, ops: &mut Vec<ForwardOp>, ctr: &mut SsaCounter, ctx: &mut TypeCtx) {
    match stmt {
        Stmt::Local(local) => parse_local(local, ops, ctr, ctx),
        Stmt::Expr(expr, _) => { parse_expr_as_stmt(expr, ops, ctr, ctx); }
        _ => {}
    }
}

fn parse_local(local: &Local, ops: &mut Vec<ForwardOp>, ctr: &mut SsaCounter, ctx: &mut TypeCtx) {
    let var_name = match &local.pat {
        Pat::Ident(PatIdent { ident, .. }) => ident.to_string(),
        Pat::Type(PatType { pat, .. }) => {
            if let Pat::Ident(PatIdent { ident, .. }) = pat.as_ref() { ident.to_string() } else { return; }
        }
        _ => return,
    };

    if let Some(init) = &local.init {
        let result = parse_expr_to_value(&init.expr, ops, ctr, ctx);
        let (name, ty) = ensure_named_typed(result, ops, ctr, ctx, "init");
        if name != var_name {
            ctx.set(&var_name, &ty);
            ops.push(ForwardOp::Assign { var: var_name, value: name, value_type: ty });
        } else {
            ctx.set(&var_name, &ty);
        }
    }
}

enum ExprValue {
    Var(String, String), // name, type
    ThreadId,
    Literal(String, String), // value, cuda_type
    Op(ForwardOp, String), // op, result_type
}

fn result_type_of(op: &ForwardOp) -> String {
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

fn parse_expr_to_value(expr: &Expr, ops: &mut Vec<ForwardOp>, ctr: &mut SsaCounter, ctx: &mut TypeCtx) -> ExprValue {
    match expr {
        Expr::Path(ExprPath { path, .. }) => {
            let name = path.segments.iter().map(|s| s.ident.to_string()).collect::<Vec<_>>().join("::");
            let ty = ctx.get(&name);
            ExprValue::Var(name, ty)
        }

        Expr::Lit(ExprLit { lit, .. }) => {
            let (val, ty) = match lit {
                Lit::Float(f) => {
                    let s = f.to_string();
                    let v = s.trim_end_matches("f32").trim_end_matches("f64").to_string();
                    // Ensure float literals have 'f' suffix
                    if v.contains('.') { (format!("{}f", v.trim_end_matches('f')), "float".to_string()) }
                    else { (format!("{}.0f", v), "float".to_string()) }
                }
                Lit::Int(i) => (i.base10_digits().to_string(), "int".to_string()),
                Lit::Bool(b) => (if b.value { "true" } else { "false" }.to_string(), "bool".to_string()),
                _ => ("0".to_string(), "float".to_string()),
            };
            ExprValue::Literal(val, ty)
        }

        Expr::Call(ExprCall { func, args, .. }) => {
            if let Expr::Path(ExprPath { path, .. }) = func.as_ref() {
                // Check for Vec3f::new(x, y, z)
                if path.segments.len() == 2 {
                    let type_name = path.segments[0].ident.to_string();
                    let method = path.segments[1].ident.to_string();
                    if let Some(info) = forge_type_info(&type_name) {
                        if method == "new" {
                            let mut arg_names = Vec::new();
                            for a in args {
                                let v = parse_expr_to_value(a, ops, ctr, ctx);
                                let (n, _) = ensure_named_typed(v, ops, ctr, ctx, "va");
                                arg_names.push(n);
                            }
                            let tmp = ctr.fresh("v");
                            let vt = info.cuda_name.to_string();
                            ctx.set(&tmp, &vt);
                            return ExprValue::Op(
                                ForwardOp::VecConstruct { var: tmp.clone(), vec_type: vt.clone(), args: arg_names },
                                vt,
                            );
                        } else if method == "zero" {
                            let vt = info.cuda_name.to_string();
                            let zeros: Vec<String> = (0..info.components).map(|_| "0.0f".to_string()).collect();
                            let tmp = ctr.fresh("v");
                            ctx.set(&tmp, &vt);
                            return ExprValue::Op(
                                ForwardOp::VecConstruct { var: tmp.clone(), vec_type: vt.clone(), args: zeros },
                                vt,
                            );
                        }
                    }
                }

                let fname = path.segments.iter().map(|s| s.ident.to_string()).collect::<Vec<_>>().join("::");
                if fname == "thread_id" { return ExprValue::ThreadId; }

                if args.len() == 1 {
                    let v = parse_expr_to_value(&args[0], ops, ctr, ctx);
                    let (arg_name, arg_type) = ensure_named_typed(v, ops, ctr, ctx, "arg");
                    let cuda_func = builtin_to_cuda_pub(&fname);
                    let tmp = ctr.fresh("t");
                    let rt = "float".to_string();
                    ctx.set(&tmp, &rt);
                    return ExprValue::Op(
                        ForwardOp::UnaryFunc { var: tmp, func: cuda_func, arg: arg_name, result_type: rt.clone(), arg_type },
                        rt,
                    );
                }
            }
            ExprValue::Literal("0".to_string(), "float".to_string())
        }

        Expr::Binary(ExprBinary { left, op, right, .. }) => {
            let lv = parse_expr_to_value(left, ops, ctr, ctx);
            let rv = parse_expr_to_value(right, ops, ctr, ctx);
            let (ln, lt) = ensure_named_typed(lv, ops, ctr, ctx, "l");
            let (rn, rt) = ensure_named_typed(rv, ops, ctr, ctx, "r");

            if let Some(kind) = binop_to_kind(op) {
                let tmp = ctr.fresh("t");
                // Infer result type
                let res_type = if lt.starts_with("forge_vec") || rt.starts_with("forge_vec") {
                    if lt.starts_with("forge_vec") { lt.clone() } else { rt.clone() }
                } else if lt == "int" && rt == "int" {
                    "int".to_string()
                } else if lt == "int" || rt == "int" {
                    // int op float → float; int op int → int
                    if lt == "float" || rt == "float" { "float".to_string() } else { "int".to_string() }
                } else {
                    "float".to_string()
                };
                ctx.set(&tmp, &res_type);
                ExprValue::Op(
                    ForwardOp::BinOp { var: tmp, left: ln, op: kind, right: rn,
                        result_type: res_type.clone(), left_type: lt, right_type: rt },
                    res_type,
                )
            } else {
                // Comparison
                let op_str = match op {
                    BinOp::Lt(_) => "<", BinOp::Le(_) => "<=", BinOp::Gt(_) => ">",
                    BinOp::Ge(_) => ">=", BinOp::Eq(_) => "==", BinOp::Ne(_) => "!=",
                    BinOp::And(_) => "&&", BinOp::Or(_) => "||", _ => "?",
                };
                let tmp = ctr.fresh("cmp");
                let cond_expr = format!("({} {} {})", ln, op_str, rn);
                ctx.set(&tmp, "bool");
                ExprValue::Op(
                    ForwardOp::Literal { var: tmp, value: cond_expr, cuda_type: "bool".to_string() },
                    "bool".to_string(),
                )
            }
        }

        Expr::Unary(ExprUnary { op: UnOp::Neg(_), expr, .. }) => {
            let v = parse_expr_to_value(expr, ops, ctr, ctx);
            let (name, ty) = ensure_named_typed(v, ops, ctr, ctx, "neg");
            let tmp = ctr.fresh("t");
            let zero = ctr.fresh("zero");
            let zero_val = if ty.starts_with("forge_vec") {
                format!("{}{{0.0f, 0.0f, 0.0f}}", ty)
            } else {
                "0.0f".to_string()
            };
            ops.push(ForwardOp::Literal { var: zero.clone(), value: zero_val, cuda_type: ty.clone() });
            ctx.set(&zero, &ty);
            ctx.set(&tmp, &ty);
            let ty_clone = ty.clone();
            ExprValue::Op(
                ForwardOp::BinOp { var: tmp, left: zero, op: BinOpKind::Sub, right: name,
                    result_type: ty.clone(), left_type: ty.clone(), right_type: ty },
                ty_clone,
            )
        }

        Expr::Index(ExprIndex { expr, index, .. }) => {
            let av = parse_expr_to_value(expr, ops, ctr, ctx);
            let iv = parse_expr_to_value(index, ops, ctr, ctx);
            let (arr_name, arr_type) = ensure_named_typed(av, ops, ctr, ctx, "arr");
            let (idx_name, _) = ensure_named_typed(iv, ops, ctr, ctx, "idx");
            let tmp = ctr.fresh("t");
            // Element type: look up from array's known type, default to array's type
            let elem_type = ctx.get(&arr_name);
            ctx.set(&tmp, &elem_type);
            ExprValue::Op(
                ForwardOp::ArrayRead { var: tmp, array: arr_name, index: idx_name, elem_type: elem_type.clone() },
                elem_type,
            )
        }

        Expr::Field(ExprField { base, member, .. }) => {
            let bv = parse_expr_to_value(base, ops, ctr, ctx);
            let (base_name, base_type) = ensure_named_typed(bv, ops, ctr, ctx, "base");
            let tmp = ctr.fresh("t");
            ctx.set(&tmp, "float"); // field access on vec → scalar
            ExprValue::Op(
                ForwardOp::FieldAccess { var: tmp, base: base_name, field: member.to_token_stream().to_string(), base_type },
                "float".to_string(),
            )
        }

        Expr::Paren(ExprParen { expr, .. }) => parse_expr_to_value(expr, ops, ctr, ctx),

        Expr::MethodCall(ExprMethodCall { receiver, method, args, .. }) => {
            let rv = parse_expr_to_value(receiver, ops, ctr, ctx);
            let (recv_name, recv_type) = ensure_named_typed(rv, ops, ctr, ctx, "recv");
            let mname = method.to_string();
            let cuda_func = match mname.as_str() {
                "abs" => "fabsf", "sqrt" => "sqrtf", "sin" => "sinf", "cos" => "cosf",
                _ => {
                    return ExprValue::Var(recv_name, recv_type);
                }
            };
            let tmp = ctr.fresh("t");
            ctx.set(&tmp, "float");
            ExprValue::Op(
                ForwardOp::UnaryFunc { var: tmp, func: cuda_func.to_string(), arg: recv_name,
                    result_type: "float".to_string(), arg_type: recv_type },
                "float".to_string(),
            )
        }

        _ => ExprValue::Literal("0".to_string(), "float".to_string()),
    }
}

fn parse_expr_as_stmt(expr: &Expr, ops: &mut Vec<ForwardOp>, ctr: &mut SsaCounter, ctx: &mut TypeCtx) {
    match expr {
        Expr::Assign(ExprAssign { left, right, .. }) => {
            if let Expr::Index(ExprIndex { expr: arr_expr, index, .. }) = left.as_ref() {
                let av = parse_expr_to_value(arr_expr, ops, ctr, ctx);
                let iv = parse_expr_to_value(index, ops, ctr, ctx);
                let val = parse_expr_to_value(right, ops, ctr, ctx);
                let (arr_name, _) = ensure_named_typed(av, ops, ctr, ctx, "arr");
                let (idx_name, _) = ensure_named_typed(iv, ops, ctr, ctx, "idx");
                let (val_name, val_type) = ensure_named_typed(val, ops, ctr, ctx, "val");
                ops.push(ForwardOp::ArrayWrite { array: arr_name, index: idx_name, value: val_name, elem_type: val_type });
            } else {
                let lhs = parse_expr_to_value(left, ops, ctr, ctx);
                let rhs = parse_expr_to_value(right, ops, ctr, ctx);
                let (ln, _) = ensure_named_typed(lhs, ops, ctr, ctx, "lhs");
                let (rn, rt) = ensure_named_typed(rhs, ops, ctr, ctx, "rhs");
                ctx.set(&ln, &rt);
                ops.push(ForwardOp::Assign { var: ln, value: rn, value_type: rt });
            }
        }

        Expr::Binary(ExprBinary { left, op, right, .. }) => {
            if let Some(kind) = compound_op_to_kind(op) {
                if let Expr::Index(ExprIndex { expr: arr_expr, index, .. }) = left.as_ref() {
                    let av = parse_expr_to_value(arr_expr, ops, ctr, ctx);
                    let iv = parse_expr_to_value(index, ops, ctr, ctx);
                    let val = parse_expr_to_value(right, ops, ctr, ctx);
                    let (an, _) = ensure_named_typed(av, ops, ctr, ctx, "arr");
                    let (in_, _) = ensure_named_typed(iv, ops, ctr, ctx, "idx");
                    let (vn, vt) = ensure_named_typed(val, ops, ctr, ctx, "val");
                    ops.push(ForwardOp::ArrayCompound { array: an, index: in_, op: kind, value: vn, elem_type: vt });
                    return;
                }
            }
            let _ = parse_expr_to_value(expr, ops, ctr, ctx);
        }

        Expr::If(ExprIf { cond, then_branch, else_branch, .. }) => {
            let cv = parse_expr_to_value(cond, ops, ctr, ctx);
            let (cn, _) = ensure_named_typed(cv, ops, ctr, ctx, "cond");
            let then_ops = {
                let mut o = Vec::new();
                let mut c = SsaCounter { next: ctr.next };
                for s in &then_branch.stmts { parse_stmt(s, &mut o, &mut c, ctx); }
                ctr.next = c.next;
                o
            };
            let else_ops = if let Some((_, else_expr)) = else_branch {
                if let Expr::Block(block) = else_expr.as_ref() {
                    let mut o = Vec::new();
                    let mut c = SsaCounter { next: ctr.next };
                    for s in &block.block.stmts { parse_stmt(s, &mut o, &mut c, ctx); }
                    ctr.next = c.next;
                    o
                } else { vec![] }
            } else { vec![] };
            ops.push(ForwardOp::IfBlock { cond: cn, then_ops, else_ops });
        }

        _ => { let _ = parse_expr_to_value(expr, ops, ctr, ctx); }
    }
}

fn ensure_named_typed(val: ExprValue, ops: &mut Vec<ForwardOp>, ctr: &mut SsaCounter, ctx: &mut TypeCtx, hint: &str) -> (String, String) {
    match val {
        ExprValue::Var(n, t) => (n, t),
        ExprValue::ThreadId => {
            let n = ctr.fresh(hint);
            ctx.set(&n, "int");
            ops.push(ForwardOp::ThreadId { var: n.clone() });
            (n, "int".to_string())
        }
        ExprValue::Literal(v, t) => {
            let n = ctr.fresh(hint);
            ctx.set(&n, &t);
            ops.push(ForwardOp::Literal { var: n.clone(), value: v, cuda_type: t.clone() });
            (n, t)
        }
        ExprValue::Op(op, t) => {
            let n = get_op_var(&op).to_string();
            ctx.set(&n, &t);
            ops.push(op);
            (n, t)
        }
    }
}

fn get_op_var(op: &ForwardOp) -> &str {
    match op {
        ForwardOp::ThreadId { var } | ForwardOp::Literal { var, .. } |
        ForwardOp::BinOp { var, .. } | ForwardOp::UnaryFunc { var, .. } |
        ForwardOp::ArrayRead { var, .. } | ForwardOp::FieldAccess { var, .. } |
        ForwardOp::Assign { var, .. } | ForwardOp::VecConstruct { var, .. } => var,
        _ => "",
    }
}

fn binop_to_kind(op: &BinOp) -> Option<BinOpKind> {
    match op {
        BinOp::Add(_) => Some(BinOpKind::Add),
        BinOp::Sub(_) => Some(BinOpKind::Sub),
        BinOp::Mul(_) => Some(BinOpKind::Mul),
        BinOp::Div(_) => Some(BinOpKind::Div),
        BinOp::Rem(_) => Some(BinOpKind::Rem),
        _ => None,
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
