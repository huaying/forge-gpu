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
    BinOp { var: String, left: String, op: BinOpKind, right: String },
    /// let var = func(arg)  — unary builtin
    UnaryFunc { var: String, func: String, arg: String },
    /// let var = arr[idx]  — array read
    ArrayRead { var: String, array: String, index: String },
    /// arr[idx] = val  — array write
    ArrayWrite { array: String, index: String, value: String },
    /// arr[idx] op= val  — compound assignment (+=, -=, *=)
    ArrayCompound { array: String, index: String, op: BinOpKind, value: String },
    /// let var = a.field  — field access (for Vec3f etc.)
    FieldAccess { var: String, base: String, field: String },
    /// if cond { ... }  — we skip control flow in Phase 1
    IfBlock { cond: String, then_ops: Vec<ForwardOp>, else_ops: Vec<ForwardOp> },
    /// var = expr  — simple scalar assignment
    Assign { var: String, value: String },
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

/// Generate adjoint CUDA statements for a list of forward operations.
///
/// `output_arrays` — names of arrays that are outputs (their adj_ arrays are inputs).
/// `input_arrays` — names of arrays that are inputs (their adj_ arrays are outputs).
///
/// Returns a list of CUDA source lines for the adjoint kernel body.
pub fn generate_adjoint_body(
    forward_ops: &[ForwardOp],
    output_arrays: &HashSet<String>,
    input_arrays: &HashSet<String>,
) -> Vec<String> {
    let mut lines = Vec::new();

    // Process operations in reverse order
    for op in forward_ops.iter().rev() {
        match op {
            ForwardOp::ThreadId { .. } | ForwardOp::Literal { .. } => {
                // No adjoint contribution
            }

            ForwardOp::BinOp { var, left, op, right } => {
                let adj_var = format!("adj_{}", var);
                let adj_left = format!("adj_{}", left);
                let adj_right = format!("adj_{}", right);

                match op {
                    BinOpKind::Add => {
                        // c = a + b  →  adj_a += adj_c; adj_b += adj_c
                        lines.push(format!("    {} += {};", adj_left, adj_var));
                        lines.push(format!("    {} += {};", adj_right, adj_var));
                    }
                    BinOpKind::Sub => {
                        // c = a - b  →  adj_a += adj_c; adj_b -= adj_c
                        lines.push(format!("    {} += {};", adj_left, adj_var));
                        lines.push(format!("    {} -= {};", adj_right, adj_var));
                    }
                    BinOpKind::Mul => {
                        // c = a * b  →  adj_a += adj_c * b; adj_b += adj_c * a
                        lines.push(format!("    {} += {} * {};", adj_left, adj_var, right));
                        lines.push(format!("    {} += {} * {};", adj_right, adj_var, left));
                    }
                    BinOpKind::Div => {
                        // c = a / b  →  adj_a += adj_c / b; adj_b -= adj_c * a / (b*b)
                        lines.push(format!("    {} += {} / {};", adj_left, adj_var, right));
                        lines.push(format!("    {} -= {} * {} / ({} * {});", adj_right, adj_var, left, right, right));
                    }
                    BinOpKind::Rem => {
                        // Modulo has no useful gradient
                    }
                }
                // Zero out adj_var after propagation
                lines.push(format!("    {} = 0.0f;", adj_var));
            }

            ForwardOp::UnaryFunc { var, func, arg } => {
                let adj_var = format!("adj_{}", var);
                let adj_arg = format!("adj_{}", arg);

                match func.as_str() {
                    "sinf" => {
                        // c = sin(a)  →  adj_a += adj_c * cos(a)
                        lines.push(format!("    {} += {} * cosf({});", adj_arg, adj_var, arg));
                    }
                    "cosf" => {
                        // c = cos(a)  →  adj_a -= adj_c * sin(a)
                        lines.push(format!("    {} -= {} * sinf({});", adj_arg, adj_var, arg));
                    }
                    "sqrtf" => {
                        // c = sqrt(a)  →  adj_a += adj_c * 0.5 / sqrt(a)
                        lines.push(format!("    {} += {} * 0.5f / sqrtf({});", adj_arg, adj_var, arg));
                    }
                    "expf" => {
                        // c = exp(a)  →  adj_a += adj_c * exp(a)
                        lines.push(format!("    {} += {} * expf({});", adj_arg, adj_var, arg));
                    }
                    "logf" => {
                        // c = log(a)  →  adj_a += adj_c / a
                        lines.push(format!("    {} += {} / {};", adj_arg, adj_var, arg));
                    }
                    "fabsf" => {
                        // c = abs(a)  →  adj_a += adj_c * sign(a)
                        lines.push(format!("    {} += {} * (({} >= 0.0f) ? 1.0f : -1.0f);", adj_arg, adj_var, arg));
                    }
                    _ => {
                        lines.push(format!("    // TODO: adjoint for {}({})", func, arg));
                    }
                }
                lines.push(format!("    {} = 0.0f;", adj_var));
            }

            ForwardOp::ArrayRead { var, array, index } => {
                // let var = arr[i]  →  adj_arr[i] += adj_var
                let adj_var = format!("adj_{}", var);
                if input_arrays.contains(array) || output_arrays.contains(array) {
                    lines.push(format!("    adj_{}[{}] += {};", array, index, adj_var));
                }
                lines.push(format!("    {} = 0.0f;", adj_var));
            }

            ForwardOp::ArrayWrite { array, index, value } => {
                // arr[i] = val  →  adj_val += adj_arr[i]; adj_arr[i] = 0
                let adj_value = format!("adj_{}", value);
                if output_arrays.contains(array) {
                    lines.push(format!("    {} += adj_{}[{}];", adj_value, array, index));
                    lines.push(format!("    adj_{}[{}] = 0.0f;", array, index));
                }
            }

            ForwardOp::ArrayCompound { array, index, op, value } => {
                // arr[i] += val  is same as arr[i] = arr[i] + val
                let adj_value = format!("adj_{}", value);
                if output_arrays.contains(array) {
                    match op {
                        BinOpKind::Add => {
                            lines.push(format!("    {} += adj_{}[{}];", adj_value, array, index));
                        }
                        BinOpKind::Sub => {
                            lines.push(format!("    {} -= adj_{}[{}];", adj_value, array, index));
                        }
                        BinOpKind::Mul => {
                            // arr[i] *= val → adj_val += adj_arr[i] * arr[i]; but we need the old value
                            lines.push(format!("    // WARNING: *= adjoint needs forward value"));
                            lines.push(format!("    {} += adj_{}[{}] * {}[{}];", adj_value, array, index, array, index));
                        }
                        _ => {
                            lines.push(format!("    // TODO: compound adjoint for {:?}", op));
                        }
                    }
                }
            }

            ForwardOp::Assign { var, value } => {
                let adj_var = format!("adj_{}", var);
                let adj_value = format!("adj_{}", value);
                lines.push(format!("    {} += {};", adj_value, adj_var));
                lines.push(format!("    {} = 0.0f;", adj_var));
            }

            ForwardOp::FieldAccess { .. } => {
                // Phase 2: Vec3f field adjoint
                lines.push("    // TODO: field access adjoint (Phase 2)".to_string());
            }

            ForwardOp::IfBlock { cond, then_ops, else_ops } => {
                // Re-evaluate condition, apply adjoint to the correct branch
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
