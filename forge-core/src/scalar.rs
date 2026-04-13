//! Scalar type traits for GPU-compatible numeric types.

use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Trait for types that can be used as scalars in Forge GPU computations.
///
/// All scalar types must be Copy, have a known size, and support basic
/// arithmetic. This trait is sealed — only Forge-provided types implement it.
pub trait Scalar:
    Copy
    + Clone
    + fmt::Debug
    + fmt::Display
    + PartialEq
    + PartialOrd
    + Default
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Send
    + Sync
    + 'static
{
    /// Zero value for this scalar type.
    const ZERO: Self;
    /// One value for this scalar type.
    const ONE: Self;
    /// The name of this type as it appears in generated GPU code.
    const GPU_TYPE_NAME: &'static str;
}

impl Scalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const GPU_TYPE_NAME: &'static str = "float";
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const GPU_TYPE_NAME: &'static str = "double";
}

impl Scalar for i32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const GPU_TYPE_NAME: &'static str = "int";
}

impl Scalar for i64 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const GPU_TYPE_NAME: &'static str = "long long";
}

// Note: u32 and u64 don't implement Scalar because vectors require Neg.
// Use i32/i64 for integer GPU computations, or wrap unsigned types manually.

/// Float-specific operations available in GPU kernels.
pub trait Float: Scalar {
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn abs(self) -> Self;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn clamp(self, lo: Self, hi: Self) -> Self;
    fn lerp(self, other: Self, t: Self) -> Self;
    fn acos_safe(self) -> Self;
    fn asin_safe(self) -> Self;
    fn atan2(self, other: Self) -> Self;
}

impl Float for f32 {
    #[inline] fn sqrt(self) -> Self { f32::sqrt(self) }
    #[inline] fn sin(self) -> Self { f32::sin(self) }
    #[inline] fn cos(self) -> Self { f32::cos(self) }
    #[inline] fn abs(self) -> Self { f32::abs(self) }
    #[inline] fn min(self, other: Self) -> Self { f32::min(self, other) }
    #[inline] fn max(self, other: Self) -> Self { f32::max(self, other) }
    #[inline] fn clamp(self, lo: Self, hi: Self) -> Self { f32::clamp(self, lo, hi) }
    #[inline] fn lerp(self, other: Self, t: Self) -> Self { self + (other - self) * t }
    #[inline] fn acos_safe(self) -> Self { f32::acos(self.clamp(-1.0, 1.0)) }
    #[inline] fn asin_safe(self) -> Self { f32::asin(self.clamp(-1.0, 1.0)) }
    #[inline] fn atan2(self, other: Self) -> Self { f32::atan2(self, other) }
}

impl Float for f64 {
    #[inline] fn sqrt(self) -> Self { f64::sqrt(self) }
    #[inline] fn sin(self) -> Self { f64::sin(self) }
    #[inline] fn cos(self) -> Self { f64::cos(self) }
    #[inline] fn abs(self) -> Self { f64::abs(self) }
    #[inline] fn min(self, other: Self) -> Self { f64::min(self, other) }
    #[inline] fn max(self, other: Self) -> Self { f64::max(self, other) }
    #[inline] fn clamp(self, lo: Self, hi: Self) -> Self { f64::clamp(self, lo, hi) }
    #[inline] fn lerp(self, other: Self, t: Self) -> Self { self + (other - self) * t }
    #[inline] fn acos_safe(self) -> Self { f64::acos(self.clamp(-1.0, 1.0)) }
    #[inline] fn asin_safe(self) -> Self { f64::asin(self.clamp(-1.0, 1.0)) }
    #[inline] fn atan2(self, other: Self) -> Self { f64::atan2(self, other) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_constants() {
        assert_eq!(f32::ZERO, 0.0);
        assert_eq!(f32::ONE, 1.0);
        assert_eq!(i32::ZERO, 0);
        assert_eq!(i32::ONE, 1);
    }

    #[test]
    fn test_float_ops() {
        assert!((2.0f32.sqrt() - std::f32::consts::SQRT_2).abs() < 1e-6);
        assert!((0.5f32.lerp(1.0, 0.5) - 0.75).abs() < 1e-6);
    }
}
