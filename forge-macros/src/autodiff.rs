//! Automatic differentiation — reverse-mode adjoint kernel generation.
//!
//! Given a forward kernel's AST, generates the corresponding adjoint (backward) kernel
//! that computes gradients via the chain rule.
//!
//! ## How it works
//!
//! For each forward operation, we know the adjoint rule:
//!
//! | Forward         | Adjoint (reverse)                    |
//! |-----------------|--------------------------------------|
//! | c = a + b       | adj_a += adj_c; adj_b += adj_c       |
//! | c = a - b       | adj_a += adj_c; adj_b -= adj_c       |
//! | c = a * b       | adj_a += adj_c * b; adj_b += adj_c * a |
//! | c = a / b       | adj_a += adj_c / b; adj_b -= adj_c * a / (b*b) |
//! | c = sin(a)      | adj_a += adj_c * cos(a)              |
//! | c = cos(a)      | adj_a -= adj_c * sin(a)              |
//! | c = sqrt(a)     | adj_a += adj_c * 0.5 / sqrt(a)       |
//! | c = a[i]        | adj_a[i] += adj_c                    |
//! | a[i] = c        | adj_c += adj_a[i]; adj_a[i] = 0      |
//!
//! The adjoint kernel processes statements in reverse order.

use std::collections::HashSet;

/// An operation in the forward kernel, in SSA-like form.
#[derive(Debug, Clone)]
pub enum ForwardOp {
    /// let var = thread_id()
    ThreadId { var: String },
    /// let var = literal
    Literal { var: String, value: String, cuda_type: String },
    /// let var = a + b  (or -, *, /)
    BinOp { var: String, left: String, op: BinOpKind, right: String,
            /// Type of the result (e.g., "float", "forge_vec3f")
            result_type: String,
            /// Type of left operand
            left_type: String,
            /// Type of right operand
            right_type: String,
    },
    /// let var = func(arg)  — unary builtin
    UnaryFunc { var: String, func: String, arg: String,
                result_type: String, arg_type: String },
    /// let var = arr[idx]  — array read
    ArrayRead { var: String, array: String, index: String,
                elem_type: String },
    /// arr[idx] = val  — array write
    ArrayWrite { array: String, index: String, value: String,
                 elem_type: String },
    /// arr[idx] op= val  — compound assignment (+=, -=, *=)
    ArrayCompound { array: String, index: String, op: BinOpKind, value: String,
                    elem_type: String },
    /// let var = a.field  — field access (for Vec3f etc.)
    FieldAccess { var: String, base: String, field: String,
                  base_type: String },
    /// if cond { ... }  
    IfBlock { cond: String, then_ops: Vec<ForwardOp>, else_ops: Vec<ForwardOp> },
    /// var = expr  — simple assignment
    Assign { var: String, value: String, value_type: String },
    /// let var = Vec3f::new(x, y, z)  — vec constructor
    VecConstruct { var: String, vec_type: String, args: Vec<String> },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

impl BinOpKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            BinOpKind::Add => "+",
            BinOpKind::Sub => "-",
            BinOpKind::Mul => "*",
            BinOpKind::Div => "/",
            BinOpKind::Rem => "%",
        }
    }
}

/// Helper: is a type a vec type?
fn is_vec_type(t: &str) -> bool {
    t.starts_with("forge_vec")
}

/// Helper: zero literal for a type
fn zero_for_type(t: &str) -> String {
    if is_vec_type(t) {
        format!("{}{{0.0f, 0.0f, 0.0f}}", t)
    } else {
        "0.0f".to_string()
    }
}

/// Helper: generate "adj_a += adj_c" for vec or scalar
fn adj_add(adj_target: &str, adj_source: &str, _target_type: &str) -> String {
    // Works for both scalar and vec because we defined operator+ for vec types
    format!("    {} = {} + {};", adj_target, adj_target, adj_source)
}

/// Helper: generate "adj_a -= adj_c" for vec or scalar  
fn adj_sub(adj_target: &str, adj_source: &str, _target_type: &str) -> String {
    format!("    {} = {} - {};", adj_target, adj_target, adj_source)
}

/// Generate adjoint CUDA statements for a list of forward operations.
pub fn generate_adjoint_body(
    forward_ops: &[ForwardOp],
    output_arrays: &HashSet<String>,
    input_arrays: &HashSet<String>,
) -> Vec<String> {
    let mut lines = Vec::new();

    for op in forward_ops.iter().rev() {
        match op {
            ForwardOp::ThreadId { .. } | ForwardOp::Literal { .. } => {}

            ForwardOp::BinOp { var, left, op, right, result_type, left_type, right_type } => {
                let adj_var = format!("adj_{}", var);
                let adj_left = format!("adj_{}", left);
                let adj_right = format!("adj_{}", right);

                match op {
                    BinOpKind::Add => {
                        // c = a + b  →  adj_a += adj_c; adj_b += adj_c
                        lines.push(adj_add(&adj_left, &adj_var, left_type));
                        lines.push(adj_add(&adj_right, &adj_var, right_type));
                    }
                    BinOpKind::Sub => {
                        lines.push(adj_add(&adj_left, &adj_var, left_type));
                        lines.push(adj_sub(&adj_right, &adj_var, right_type));
                    }
                    BinOpKind::Mul => {
                        if is_vec_type(left_type) && !is_vec_type(right_type) {
                            // vec * scalar: adj_vec += adj_result * scalar; adj_scalar += dot(adj_result, vec)
                            lines.push(format!("    {} = {} + {} * {};", adj_left, adj_left, adj_var, right));
                            // For scalar adjoint from vec: need dot product
                            // dot(adj_result, left_vec)
                            if is_vec_type(result_type) {
                                let dot_expr = dot_product(&adj_var, left, result_type);
                                lines.push(format!("    {} = {} + {};", adj_right, adj_right, dot_expr));
                            } else {
                                lines.push(format!("    {} += {} * {};", adj_right, adj_var, left));
                            }
                        } else if !is_vec_type(left_type) && is_vec_type(right_type) {
                            // scalar * vec
                            if is_vec_type(result_type) {
                                let dot_expr = dot_product(&adj_var, right, result_type);
                                lines.push(format!("    {} = {} + {};", adj_left, adj_left, dot_expr));
                            } else {
                                lines.push(format!("    {} += {} * {};", adj_left, adj_var, right));
                            }
                            lines.push(format!("    {} = {} + {} * {};", adj_right, adj_right, adj_var, left));
                        } else {
                            // scalar * scalar (original behavior)
                            lines.push(format!("    {} += {} * {};", adj_left, adj_var, right));
                            lines.push(format!("    {} += {} * {};", adj_right, adj_var, left));
                        }
                    }
                    BinOpKind::Div => {
                        if !is_vec_type(left_type) && !is_vec_type(right_type) {
                            lines.push(format!("    {} += {} / {};", adj_left, adj_var, right));
                            lines.push(format!("    {} -= {} * {} / ({} * {});", adj_right, adj_var, left, right, right));
                        } else {
                            // vec / scalar
                            lines.push(format!("    {} = {} + {} / {};", adj_left, adj_left, adj_var, right));
                            if is_vec_type(result_type) {
                                let dot_expr = dot_product(&adj_var, left, result_type);
                                lines.push(format!("    {} -= {} / ({} * {});", adj_right, dot_expr, right, right));
                            }
                        }
                    }
                    BinOpKind::Rem => {}
                }
                lines.push(format!("    {} = {};", adj_var, zero_for_type(result_type)));
            }

            ForwardOp::UnaryFunc { var, func, arg, result_type, arg_type } => {
                let adj_var = format!("adj_{}", var);
                let adj_arg = format!("adj_{}", arg);

                match func.as_str() {
                    "sinf" => lines.push(format!("    {} += {} * cosf({});", adj_arg, adj_var, arg)),
                    "cosf" => lines.push(format!("    {} -= {} * sinf({});", adj_arg, adj_var, arg)),
                    "sqrtf" => lines.push(format!("    {} += {} * 0.5f / sqrtf({});", adj_arg, adj_var, arg)),
                    "expf" => lines.push(format!("    {} += {} * expf({});", adj_arg, adj_var, arg)),
                    "logf" => lines.push(format!("    {} += {} / {};", adj_arg, adj_var, arg)),
                    "fabsf" => lines.push(format!("    {} += {} * (({} >= 0.0f) ? 1.0f : -1.0f);", adj_arg, adj_var, arg)),
                    _ => lines.push(format!("    // TODO: adjoint for {}({})", func, arg)),
                }
                lines.push(format!("    {} = {};", adj_var, zero_for_type(result_type)));
            }

            ForwardOp::ArrayRead { var, array, index, elem_type } => {
                let adj_var = format!("adj_{}", var);
                if input_arrays.contains(array) || output_arrays.contains(array) {
                    lines.push(format!("    adj_{}[{}] = adj_{}[{}] + {};", array, index, array, index, adj_var));
                }
                lines.push(format!("    {} = {};", adj_var, zero_for_type(elem_type)));
            }

            ForwardOp::ArrayWrite { array, index, value, elem_type } => {
                let adj_value = format!("adj_{}", value);
                if output_arrays.contains(array) {
                    lines.push(format!("    {} = {} + adj_{}[{}];", adj_value, adj_value, array, index));
                    lines.push(format!("    adj_{}[{}] = {};", array, index, zero_for_type(elem_type)));
                }
            }

            ForwardOp::ArrayCompound { array, index, op, value, elem_type } => {
                let adj_value = format!("adj_{}", value);
                if output_arrays.contains(array) {
                    match op {
                        BinOpKind::Add => {
                            lines.push(format!("    {} = {} + adj_{}[{}];", adj_value, adj_value, array, index));
                        }
                        BinOpKind::Sub => {
                            lines.push(format!("    {} = {} - adj_{}[{}];", adj_value, adj_value, array, index));
                        }
                        _ => {
                            lines.push(format!("    // TODO: compound adjoint for {:?}", op));
                        }
                    }
                }
            }

            ForwardOp::Assign { var, value, value_type } => {
                // Skip adjoint for int assignments (thread_id etc.)
                if value_type == "int" || value_type == "bool" {
                    continue;
                }
                let adj_var = format!("adj_{}", var);
                let adj_value = format!("adj_{}", value);
                lines.push(adj_add(&adj_value, &adj_var, value_type));
                lines.push(format!("    {} = {};", adj_var, zero_for_type(value_type)));
            }

            ForwardOp::FieldAccess { var, base, field, base_type } => {
                // var = base.x  (scalar from vec)
                // adj_base.x += adj_var
                let adj_var = format!("adj_{}", var);
                let adj_base = format!("adj_{}", base);
                lines.push(format!("    {}.{} += {};", adj_base, field, adj_var));
                lines.push(format!("    {} = 0.0f;", adj_var));
            }

            ForwardOp::VecConstruct { var, vec_type, args } => {
                // var = Vec3f::new(a, b, c)
                // adj_a += adj_var.x; adj_b += adj_var.y; adj_c += adj_var.z
                let adj_var = format!("adj_{}", var);
                let fields = match args.len() {
                    2 => vec!["x", "y"],
                    3 => vec!["x", "y", "z"],
                    4 => vec!["x", "y", "z", "w"],
                    _ => vec![],
                };
                for (i, arg) in args.iter().enumerate() {
                    if i < fields.len() {
                        let adj_arg = format!("adj_{}", arg);
                        lines.push(format!("    {} += {}.{};", adj_arg, adj_var, fields[i]));
                    }
                }
                lines.push(format!("    {} = {};", adj_var, zero_for_type(vec_type)));
            }

            ForwardOp::IfBlock { cond, then_ops, else_ops } => {
                let then_adj = generate_adjoint_body(then_ops, output_arrays, input_arrays);
                let else_adj = generate_adjoint_body(else_ops, output_arrays, input_arrays);
                lines.push(format!("    if ({}) {{", cond));
                lines.extend(then_adj.iter().map(|l| format!("    {}", l)));
                if !else_adj.is_empty() {
                    lines.push("    } else {".to_string());
                    lines.extend(else_adj.iter().map(|l| format!("    {}", l)));
                }
                lines.push("    }".to_string());
            }
        }
    }

    lines
}

/// Generate a dot product expression for a vec type
fn dot_product(a: &str, b: &str, vec_type: &str) -> String {
    if vec_type.contains("vec2") {
        format!("({}.x * {}.x + {}.y * {}.y)", a, b, a, b)
    } else if vec_type.contains("vec4") {
        format!("({}.x * {}.x + {}.y * {}.y + {}.z * {}.z + {}.w * {}.w)", a, b, a, b, a, b, a, b)
    } else {
        // Default: vec3
        format!("({}.x * {}.x + {}.y * {}.y + {}.z * {}.z)", a, b, a, b, a, b)
    }
}

/// Information about kernel parameters classified for autodiff.
#[derive(Debug)]
pub struct AutodiffParams {
    /// Input arrays (read-only in forward) — get adj_ output arrays
    pub inputs: Vec<String>,
    /// Output arrays (written in forward) — get adj_ input arrays
    pub outputs: Vec<String>,
    /// Scalar params — get adj_ scalars (accumulated)
    pub scalars: Vec<String>,
    /// Local variables found in forward body
    pub locals: HashSet<String>,
}
