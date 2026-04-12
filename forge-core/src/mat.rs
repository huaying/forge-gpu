//! Matrix types for GPU computation.
//!
//! `Mat22<T>`, `Mat33<T>`, `Mat44<T>` — column-major storage,
//! `#[repr(C)]` for GPU compatibility.

use crate::scalar::Scalar;
use crate::vec::{Vec2, Vec3, Vec4};
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

// ---------------------------------------------------------------------------
// Mat22
// ---------------------------------------------------------------------------

/// A 2x2 matrix (column-major).
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Default)]
pub struct Mat22<T: Scalar> {
    /// Columns of the matrix.
    pub cols: [Vec2<T>; 2],
}

impl<T: Scalar> Mat22<T> {
    #[inline]
    pub fn new(c0r0: T, c0r1: T, c1r0: T, c1r1: T) -> Self {
        Self {
            cols: [Vec2::new(c0r0, c0r1), Vec2::new(c1r0, c1r1)],
        }
    }

    #[inline]
    pub fn identity() -> Self {
        Self::new(T::ONE, T::ZERO, T::ZERO, T::ONE)
    }

    #[inline]
    pub fn zero() -> Self {
        Self::new(T::ZERO, T::ZERO, T::ZERO, T::ZERO)
    }

    #[inline]
    pub fn determinant(self) -> T {
        self.cols[0].x * self.cols[1].y - self.cols[1].x * self.cols[0].y
    }

    #[inline]
    pub fn transpose(self) -> Self {
        Self::new(
            self.cols[0].x, self.cols[1].x,
            self.cols[0].y, self.cols[1].y,
        )
    }
}

impl<T: Scalar> fmt::Debug for Mat22<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mat22([{}, {}], [{}, {}])",
            self.cols[0].x, self.cols[1].x,
            self.cols[0].y, self.cols[1].y)
    }
}

impl<T: Scalar> Add for Mat22<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self { cols: [self.cols[0] + rhs.cols[0], self.cols[1] + rhs.cols[1]] }
    }
}

impl<T: Scalar> Sub for Mat22<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self { cols: [self.cols[0] - rhs.cols[0], self.cols[1] - rhs.cols[1]] }
    }
}

// Mat22 * Vec2
impl<T: Scalar> Mul<Vec2<T>> for Mat22<T> {
    type Output = Vec2<T>;
    #[inline]
    fn mul(self, v: Vec2<T>) -> Vec2<T> {
        self.cols[0] * v.x + self.cols[1] * v.y
    }
}

// ---------------------------------------------------------------------------
// Mat33
// ---------------------------------------------------------------------------

/// A 3x3 matrix (column-major).
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Default)]
pub struct Mat33<T: Scalar> {
    pub cols: [Vec3<T>; 3],
}

impl<T: Scalar> Mat33<T> {
    #[inline]
    pub fn from_cols(c0: Vec3<T>, c1: Vec3<T>, c2: Vec3<T>) -> Self {
        Self { cols: [c0, c1, c2] }
    }

    #[inline]
    pub fn identity() -> Self {
        Self::from_cols(
            Vec3::new(T::ONE, T::ZERO, T::ZERO),
            Vec3::new(T::ZERO, T::ONE, T::ZERO),
            Vec3::new(T::ZERO, T::ZERO, T::ONE),
        )
    }

    #[inline]
    pub fn zero() -> Self {
        Self::from_cols(Vec3::zero(), Vec3::zero(), Vec3::zero())
    }

    #[inline]
    pub fn transpose(self) -> Self {
        Self::from_cols(
            Vec3::new(self.cols[0].x, self.cols[1].x, self.cols[2].x),
            Vec3::new(self.cols[0].y, self.cols[1].y, self.cols[2].y),
            Vec3::new(self.cols[0].z, self.cols[1].z, self.cols[2].z),
        )
    }

    #[inline]
    pub fn determinant(self) -> T {
        let c0 = self.cols[0];
        let c1 = self.cols[1];
        let c2 = self.cols[2];
        c0.x * (c1.y * c2.z - c1.z * c2.y)
            - c1.x * (c0.y * c2.z - c0.z * c2.y)
            + c2.x * (c0.y * c1.z - c0.z * c1.y)
    }
}

impl<T: Scalar> fmt::Debug for Mat33<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mat33([{:?}, {:?}, {:?}])", self.cols[0], self.cols[1], self.cols[2])
    }
}

impl<T: Scalar> Add for Mat33<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self { cols: [self.cols[0] + rhs.cols[0], self.cols[1] + rhs.cols[1], self.cols[2] + rhs.cols[2]] }
    }
}

impl<T: Scalar> Sub for Mat33<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self { cols: [self.cols[0] - rhs.cols[0], self.cols[1] - rhs.cols[1], self.cols[2] - rhs.cols[2]] }
    }
}

impl<T: Scalar> Neg for Mat33<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self { cols: [-self.cols[0], -self.cols[1], -self.cols[2]] }
    }
}

// Mat33 * Vec3
impl<T: Scalar> Mul<Vec3<T>> for Mat33<T> {
    type Output = Vec3<T>;
    #[inline]
    fn mul(self, v: Vec3<T>) -> Vec3<T> {
        self.cols[0] * v.x + self.cols[1] * v.y + self.cols[2] * v.z
    }
}

// Mat33 * Mat33
impl<T: Scalar> Mul for Mat33<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::from_cols(
            self * rhs.cols[0],
            self * rhs.cols[1],
            self * rhs.cols[2],
        )
    }
}

// Scalar * Mat33
impl Mul<Mat33<f32>> for f32 {
    type Output = Mat33<f32>;
    #[inline]
    fn mul(self, rhs: Mat33<f32>) -> Mat33<f32> {
        Mat33::from_cols(rhs.cols[0] * self, rhs.cols[1] * self, rhs.cols[2] * self)
    }
}

// ---------------------------------------------------------------------------
// Mat44
// ---------------------------------------------------------------------------

/// A 4x4 matrix (column-major).
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Default)]
pub struct Mat44<T: Scalar> {
    pub cols: [Vec4<T>; 4],
}

impl<T: Scalar> Mat44<T> {
    #[inline]
    pub fn from_cols(c0: Vec4<T>, c1: Vec4<T>, c2: Vec4<T>, c3: Vec4<T>) -> Self {
        Self { cols: [c0, c1, c2, c3] }
    }

    #[inline]
    pub fn identity() -> Self {
        Self::from_cols(
            Vec4::new(T::ONE, T::ZERO, T::ZERO, T::ZERO),
            Vec4::new(T::ZERO, T::ONE, T::ZERO, T::ZERO),
            Vec4::new(T::ZERO, T::ZERO, T::ONE, T::ZERO),
            Vec4::new(T::ZERO, T::ZERO, T::ZERO, T::ONE),
        )
    }

    #[inline]
    pub fn zero() -> Self {
        Self::from_cols(Vec4::zero(), Vec4::zero(), Vec4::zero(), Vec4::zero())
    }

    #[inline]
    pub fn transpose(self) -> Self {
        Self::from_cols(
            Vec4::new(self.cols[0].x, self.cols[1].x, self.cols[2].x, self.cols[3].x),
            Vec4::new(self.cols[0].y, self.cols[1].y, self.cols[2].y, self.cols[3].y),
            Vec4::new(self.cols[0].z, self.cols[1].z, self.cols[2].z, self.cols[3].z),
            Vec4::new(self.cols[0].w, self.cols[1].w, self.cols[2].w, self.cols[3].w),
        )
    }
}

impl<T: Scalar> fmt::Debug for Mat44<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mat44([{:?}, {:?}, {:?}, {:?}])",
            self.cols[0], self.cols[1], self.cols[2], self.cols[3])
    }
}

// Mat44 * Vec4
impl<T: Scalar> Mul<Vec4<T>> for Mat44<T> {
    type Output = Vec4<T>;
    #[inline]
    fn mul(self, v: Vec4<T>) -> Vec4<T> {
        self.cols[0] * v.x + self.cols[1] * v.y + self.cols[2] * v.z + self.cols[3] * v.w
    }
}

// Mat44 * Mat44
impl<T: Scalar> Mul for Mat44<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::from_cols(
            self * rhs.cols[0],
            self * rhs.cols[1],
            self * rhs.cols[2],
            self * rhs.cols[3],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat33_identity() {
        let m = Mat33::<f32>::identity();
        let v = Vec3::new(1.0f32, 2.0, 3.0);
        assert_eq!(m * v, v);
    }

    #[test]
    fn test_mat33_mul() {
        let m = Mat33::from_cols(
            Vec3::new(2.0f32, 0.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
            Vec3::new(0.0, 0.0, 4.0),
        );
        let v = Vec3::new(1.0f32, 1.0, 1.0);
        assert_eq!(m * v, Vec3::new(2.0, 3.0, 4.0));
    }

    #[test]
    fn test_mat33_determinant() {
        let m = Mat33::<f32>::identity();
        assert!((m.determinant() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mat33_transpose() {
        let m = Mat33::from_cols(
            Vec3::new(1.0f32, 4.0, 7.0),
            Vec3::new(2.0, 5.0, 8.0),
            Vec3::new(3.0, 6.0, 9.0),
        );
        let mt = m.transpose();
        assert_eq!(mt.cols[0], Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(mt.cols[1], Vec3::new(4.0, 5.0, 6.0));
    }

    #[test]
    fn test_mat44_identity() {
        let m = Mat44::<f32>::identity();
        let v = Vec4::new(1.0f32, 2.0, 3.0, 4.0);
        assert_eq!(m * v, v);
    }
}
