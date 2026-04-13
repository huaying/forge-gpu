//! Expression language parser and CUDA codegen for custom forces.
//!
//! Parses simple expressions like:
//!   "vel.y += sin(pos.x * 3.14) * 0.1"
//!   "vel = vel * 0.99"
//!   "pos.y = max(pos.y, 0.0)"
//!
//! Supported:
//!   - Fields: pos.x, pos.y, pos.z, vel.x, vel.y, vel.z, density
//!   - Operators: +, -, *, /, %, +=, -=, *=, =
//!   - Builtins: sin, cos, sqrt, abs, min, max, floor, ceil, exp, log, pow, atan2
//!   - Constants: pi, dt (timestep), n (particle count)
//!   - Literals: 1.0, -3.14, 1e-6
//!   - Parentheses for grouping
//!   - Multiple statements separated by ;

/// Generates a CUDA kernel source string from an expression.
///
/// The kernel operates per-particle: each thread processes one particle.
/// Fields are passed as separate float arrays (pos_x, pos_y, pos_z, vel_x, etc.)
pub fn compile_expr_to_cuda(expr: &str, kernel_name: &str) -> String {
    // Split into statements
    let stmts: Vec<&str> = expr.split(';').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();

    // Analyze which fields are read/written
    let full_expr = expr;
    let reads_pos = full_expr.contains("pos.");
    let reads_vel = full_expr.contains("vel.");
    let reads_density = full_expr.contains("density");
    let writes_pos = has_write(full_expr, "pos.");
    let writes_vel = has_write(full_expr, "vel.");

    // Build kernel
    let mut src = String::new();
    src.push_str(&format!(
        "extern \"C\" __global__ void {}(\n", kernel_name
    ));

    // Always include pos and vel (simplifies codegen)
    src.push_str("    float* pos_x, float* pos_y, float* pos_z,\n");
    src.push_str("    float* vel_x, float* vel_y, float* vel_z,\n");
    if reads_density {
        src.push_str("    const float* density_arr,\n");
    }
    src.push_str("    float dt, int n\n");
    src.push_str(") {\n");
    src.push_str("    int i = blockIdx.x * blockDim.x + threadIdx.x;\n");
    src.push_str("    if (i >= n) return;\n\n");

    // Load fields into local variables
    src.push_str("    // Load\n");
    src.push_str("    float px = pos_x[i], py = pos_y[i], pz = pos_z[i];\n");
    src.push_str("    float vx = vel_x[i], vy = vel_y[i], vz = vel_z[i];\n");
    if reads_density {
        src.push_str("    float dens = density_arr[i];\n");
    }
    src.push_str("    const float pi = 3.14159265358979323846f;\n");
    src.push_str("\n");

    // Translate statements
    src.push_str("    // User expressions\n");
    for stmt in &stmts {
        let cuda_stmt = translate_statement(stmt);
        src.push_str(&format!("    {};\n", cuda_stmt));
    }

    // Write back modified fields
    src.push_str("\n    // Store\n");
    if writes_pos || true {
        // Always write back (simpler, user might modify via complex expr)
        src.push_str("    pos_x[i] = px; pos_y[i] = py; pos_z[i] = pz;\n");
    }
    if writes_vel || true {
        src.push_str("    vel_x[i] = vx; vel_y[i] = vy; vel_z[i] = vz;\n");
    }

    src.push_str("}\n");
    src
}

/// Check if an expression writes to a field prefix (e.g., "pos." or "vel.")
fn has_write(expr: &str, field_prefix: &str) -> bool {
    // Look for patterns like "pos.x =" or "pos.x +=" etc.
    for part in expr.split(field_prefix) {
        if part.is_empty() { continue; }
        // After "pos." we expect "x", "y", or "z" then whitespace then an assignment op
        let rest = part.trim_start();
        if rest.starts_with('x') || rest.starts_with('y') || rest.starts_with('z') {
            let after_field = &rest[1..].trim_start();
            if after_field.starts_with("+=") || after_field.starts_with("-=")
                || after_field.starts_with("*=") || after_field.starts_with("/=")
                || after_field.starts_with('=')
            {
                return true;
            }
        }
    }
    false
}

/// Translate a single expression statement to CUDA.
/// Maps field names: pos.x → px, vel.y → vy, density → dens
/// Maps builtins: sin → sinf, cos → cosf, etc.
fn translate_statement(stmt: &str) -> String {
    let mut result = stmt.to_string();

    // Field mappings (order matters — longer first to avoid partial matches)
    result = result.replace("pos.x", "px");
    result = result.replace("pos.y", "py");
    result = result.replace("pos.z", "pz");
    result = result.replace("vel.x", "vx");
    result = result.replace("vel.y", "vy");
    result = result.replace("vel.z", "vz");
    result = result.replace("density", "dens");

    // Math builtins → CUDA float versions
    // Be careful not to replace inside other words
    let builtins = [
        ("sqrt", "sqrtf"),
        ("sin", "sinf"),
        ("cos", "cosf"),
        ("abs", "fabsf"),
        ("floor", "floorf"),
        ("ceil", "ceilf"),
        ("exp", "expf"),
        ("log", "logf"),
        ("pow", "powf"),
        ("atan2", "atan2f"),
        ("min", "fminf"),
        ("max", "fmaxf"),
        ("round", "roundf"),
        ("tan", "tanf"),
        ("asin", "asinf"),
        ("acos", "acosf"),
    ];

    for (from, to) in &builtins {
        // Replace "sin(" with "sinf(" — but not "asin(" with "asinf("
        // We do replacements from the result string using a word-boundary approach
        result = replace_builtin(&result, from, to);
    }

    result
}

/// Replace a builtin function name with its CUDA equivalent,
/// ensuring we don't replace partial matches (e.g., "sin" in "asin").
fn replace_builtin(s: &str, from: &str, to: &str) -> String {
    let mut result = String::with_capacity(s.len() + 16);
    let bytes = s.as_bytes();
    let from_bytes = from.as_bytes();
    let from_len = from_bytes.len();
    let mut i = 0;

    while i < bytes.len() {
        if i + from_len <= bytes.len() && &bytes[i..i + from_len] == from_bytes {
            // Check character before: must not be alphanumeric or underscore
            let before_ok = if i == 0 {
                true
            } else {
                let c = bytes[i - 1] as char;
                !c.is_alphanumeric() && c != '_'
            };
            // Check character after: must be '(' for a function call
            let after_ok = if i + from_len < bytes.len() {
                bytes[i + from_len] == b'('
            } else {
                false
            };

            if before_ok && after_ok {
                result.push_str(to);
                i += from_len;
                continue;
            }
        }
        result.push(bytes[i] as char);
        i += 1;
    }

    result
}

/// Returns which fields the expression reads (for kernel arg building).
pub struct ExprFields {
    pub reads_density: bool,
}

pub fn analyze_expr(expr: &str) -> ExprFields {
    ExprFields {
        reads_density: expr.contains("density"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_gravity() {
        let cuda = compile_expr_to_cuda("vel.y += -9.81 * dt", "test_gravity");
        assert!(cuda.contains("vy += -9.81 * dt"));
        assert!(cuda.contains("extern \"C\" __global__"));
    }

    #[test]
    fn test_sin_builtin() {
        let cuda = compile_expr_to_cuda("vel.y += sin(pos.x * 3.14) * 0.1", "test_sin");
        assert!(cuda.contains("sinf(px * 3.14)"));
        assert!(!cuda.contains("asinf")); // should not corrupt asin
    }

    #[test]
    fn test_multiple_statements() {
        let cuda = compile_expr_to_cuda("vel.x *= 0.99; vel.y *= 0.99; vel.z *= 0.99", "test_multi");
        assert!(cuda.contains("vx *= 0.99"));
        assert!(cuda.contains("vy *= 0.99"));
        assert!(cuda.contains("vz *= 0.99"));
    }

    #[test]
    fn test_complex_expr() {
        let cuda = compile_expr_to_cuda(
            "vel.x += -pos.x * 0.1; vel.y += -pos.y * 0.1 + sin(pos.x) * 0.05",
            "test_complex",
        );
        assert!(cuda.contains("vx += -px * 0.1"));
        assert!(cuda.contains("sinf(px)"));
    }

    #[test]
    fn test_density_field() {
        let cuda = compile_expr_to_cuda("vel.y += density * 0.001", "test_density");
        assert!(cuda.contains("const float* density_arr"));
        assert!(cuda.contains("dens * 0.001"));
    }

    #[test]
    fn test_min_max() {
        let cuda = compile_expr_to_cuda("pos.y = max(pos.y, 0.0)", "test_minmax");
        assert!(cuda.contains("fmaxf(py, 0.0)"));
    }

    #[test]
    fn test_no_false_builtin_replace() {
        let cuda = compile_expr_to_cuda("vel.y += asin(pos.x)", "test_asin");
        assert!(cuda.contains("asinf(px)"));
        // Make sure "sin" inside "asin" wasn't double-replaced
        assert!(!cuda.contains("asinff"));
    }
}
