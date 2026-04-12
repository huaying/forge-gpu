//! # forge-codegen
//!
//! Code generation for Forge GPU compute framework.
//!
//! This crate is responsible for:
//! - `#[kernel]` proc macro — transforms Rust functions into GPU kernels
//! - `#[func]` proc macro — marks functions as callable from GPU kernels
//! - Generating CUDA C++, PTX, or SPIR-V from Rust kernel definitions
//! - Generating adjoint (reverse-mode) kernels for automatic differentiation
//!
//! ## Architecture
//!
//! ```text
//! Rust fn with #[kernel]
//!   → proc macro parses TokenStream
//!   → validates types against forge-core type catalog
//!   → emits CUDA C++ source string
//!   → compiles via nvcc/ptx-compiler (build-time or JIT)
//!   → generates host-side launch wrapper
//! ```
//!
//! ## Status
//!
//! 🚧 Under development — proc macro infrastructure and CUDA C++ emitter
//! are the first targets.

/// Placeholder for the kernel code generation pipeline.
///
/// In the full implementation, this module will:
/// 1. Accept a parsed Rust function AST
/// 2. Map Forge types to CUDA types
/// 3. Emit a CUDA C++ kernel string
/// 4. Handle control flow (if/else, for, while)
/// 5. Map builtin functions (sin, cos, dot, cross, etc.)
pub mod cuda {
    /// Map a Forge scalar type to its CUDA C++ equivalent.
    pub fn scalar_type_to_cuda(type_name: &str) -> &str {
        match type_name {
            "f32" => "float",
            "f64" => "double",
            "i32" => "int",
            "i64" => "long long",
            "u32" => "unsigned int",
            "u64" => "unsigned long long",
            "bool" => "bool",
            _ => panic!("Unknown scalar type: {}", type_name),
        }
    }

    /// Map a Forge vector type to its CUDA C++ equivalent.
    pub fn vec_type_to_cuda(type_name: &str) -> &str {
        match type_name {
            "Vec2f" | "Vec2<f32>" => "wp::vec2f",
            "Vec3f" | "Vec3<f32>" => "wp::vec3f",
            "Vec4f" | "Vec4<f32>" => "wp::vec4f",
            "Vec3d" | "Vec3<f64>" => "wp::vec3d",
            _ => panic!("Unknown vector type: {}", type_name),
        }
    }
}

/// Placeholder for the autodiff adjoint generation.
///
/// The full implementation will:
/// 1. Parse the forward kernel
/// 2. Build an SSA (Static Single Assignment) graph
/// 3. Generate the reverse-mode adjoint by traversing the graph backward
/// 4. Emit the adjoint kernel as CUDA C++
pub mod autodiff {
    /// Marker for functions that should have adjoints generated.
    pub struct AdjointRequest {
        pub kernel_name: String,
        pub generate_forward: bool,
        pub generate_backward: bool,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_mapping() {
        assert_eq!(cuda::scalar_type_to_cuda("f32"), "float");
        assert_eq!(cuda::scalar_type_to_cuda("f64"), "double");
        assert_eq!(cuda::scalar_type_to_cuda("i32"), "int");
    }

    #[test]
    fn test_vec_mapping() {
        assert_eq!(cuda::vec_type_to_cuda("Vec3f"), "wp::vec3f");
        assert_eq!(cuda::vec_type_to_cuda("Vec4f"), "wp::vec4f");
    }
}
