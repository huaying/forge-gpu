//! # forge-core
//!
//! Core type system for Forge GPU compute framework.
//!
//! Provides:
//! - Scalar types matching GPU precision (f16, f32, f64, integer types)
//! - Vector types: `Vec2<T>`, `Vec3<T>`, `Vec4<T>`
//! - Matrix types: `Mat22<T>`, `Mat33<T>`, `Mat44<T>`
//! - Quaternion: `Quat<T>`
//! - Spatial types: `Transform<T>`, `SpatialVector<T>`
//!
//! All types are `#[repr(C)]` for GPU compatibility and implement
//! standard math operations via operator overloading.

mod scalar;
mod vec;
mod mat;
mod quat;

pub use scalar::*;
pub use vec::*;
pub use mat::*;
pub use quat::*;

// Type aliases for common configurations
/// 2D vector of f32
pub type Vec2f = Vec2<f32>;
/// 3D vector of f32
pub type Vec3f = Vec3<f32>;
/// 4D vector of f32
pub type Vec4f = Vec4<f32>;
/// 2D vector of f64
pub type Vec2d = Vec2<f64>;
/// 3D vector of f64
pub type Vec3d = Vec3<f64>;
/// 4D vector of f64
pub type Vec4d = Vec4<f64>;

/// 2x2 matrix of f32
pub type Mat22f = Mat22<f32>;
/// 3x3 matrix of f32
pub type Mat33f = Mat33<f32>;
/// 4x4 matrix of f32
pub type Mat44f = Mat44<f32>;

/// Quaternion of f32
pub type Quatf = Quat<f32>;
/// Quaternion of f64
pub type Quatd = Quat<f64>;
