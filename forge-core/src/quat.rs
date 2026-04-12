//! Quaternion type for GPU computation.
//!
//! `Quat<T>` — Hamilton quaternion (x, y, z, w) convention,
//! `#[repr(C)]` for GPU compatibility.

use crate::scalar::{Float, Scalar};
use crate::vec::Vec3;
use crate::mat::Mat33;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// A quaternion (x, y, z, w) — w is the scalar part.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Default)]
pub struct Quat<T: Scalar> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

impl<T: Scalar> Quat<T> {
    #[inline]
    pub const fn new(x: T, y: T, z: T, w: T) -> Self {
        Self { x, y, z, w }
    }

    /// The identity quaternion (no rotation).
    #[inline]
    pub fn identity() -> Self {
        Self { x: T::ZERO, y: T::ZERO, z: T::ZERO, w: T::ONE }
    }

    /// Dot product of two quaternions.
    #[inline]
    pub fn dot(self, other: Self) -> T {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    /// Conjugate (negate the vector part).
    #[inline]
    pub fn conjugate(self) -> Self {
        Self { x: -self.x, y: -self.y, z: -self.z, w: self.w }
    }
}

impl<T: Float> Quat<T> {
    /// Create a quaternion from an axis and angle (radians).
    #[inline]
    pub fn from_axis_angle(axis: Vec3<T>, angle: T) -> Self {
        let half = angle * (T::ONE / (T::ONE + T::ONE));
        let s = half.sin();
        let c = half.cos();
        Self {
            x: axis.x * s,
            y: axis.y * s,
            z: axis.z * s,
            w: c,
        }
    }

    /// Length (norm) of the quaternion.
    #[inline]
    pub fn length(self) -> T {
        self.dot(self).sqrt()
    }

    /// Normalize to unit quaternion.
    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        Self {
            x: self.x / len,
            y: self.y / len,
            z: self.z / len,
            w: self.w / len,
        }
    }

    /// Inverse (conjugate / norm²). For unit quaternions, this equals conjugate.
    #[inline]
    pub fn inverse(self) -> Self {
        let norm_sq = self.dot(self);
        Self {
            x: -self.x / norm_sq,
            y: -self.y / norm_sq,
            z: -self.z / norm_sq,
            w: self.w / norm_sq,
        }
    }

    /// Rotate a 3D vector by this quaternion.
    #[inline]
    pub fn rotate(self, v: Vec3<T>) -> Vec3<T> {
        // q * v * q^-1, optimized
        let qv = Vec3::new(self.x, self.y, self.z);
        let t = qv.cross(v) * (T::ONE + T::ONE);
        v + t * self.w + qv.cross(t)
    }

    /// Convert to a 3x3 rotation matrix.
    #[inline]
    pub fn to_mat33(self) -> Mat33<T> {
        let two = T::ONE + T::ONE;
        let xx = self.x * self.x;
        let yy = self.y * self.y;
        let zz = self.z * self.z;
        let xy = self.x * self.y;
        let xz = self.x * self.z;
        let yz = self.y * self.z;
        let wx = self.w * self.x;
        let wy = self.w * self.y;
        let wz = self.w * self.z;

        Mat33::from_cols(
            Vec3::new(T::ONE - two * (yy + zz), two * (xy + wz), two * (xz - wy)),
            Vec3::new(two * (xy - wz), T::ONE - two * (xx + zz), two * (yz + wx)),
            Vec3::new(two * (xz + wy), two * (yz - wx), T::ONE - two * (xx + yy)),
        )
    }
}

impl<T: Scalar> fmt::Debug for Quat<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Quat({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

impl<T: Scalar> fmt::Display for Quat<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} + {}i + {}j + {}k)", self.w, self.x, self.y, self.z)
    }
}

impl<T: Scalar> Add for Quat<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z, w: self.w + rhs.w }
    }
}

impl<T: Scalar> Sub for Quat<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z, w: self.w - rhs.w }
    }
}

impl<T: Scalar> Neg for Quat<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self { x: -self.x, y: -self.y, z: -self.z, w: -self.w }
    }
}

/// Hamilton product: q1 * q2
impl<T: Scalar> Mul for Quat<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
        }
    }
}

/// Scalar * Quat
impl Mul<Quat<f32>> for f32 {
    type Output = Quat<f32>;
    #[inline]
    fn mul(self, rhs: Quat<f32>) -> Quat<f32> {
        Quat { x: rhs.x * self, y: rhs.y * self, z: rhs.z * self, w: rhs.w * self }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let q = Quat::<f32>::identity();
        let v = Vec3::new(1.0f32, 2.0, 3.0);
        let rotated = q.rotate(v);
        assert!((rotated.x - v.x).abs() < 1e-6);
        assert!((rotated.y - v.y).abs() < 1e-6);
        assert!((rotated.z - v.z).abs() < 1e-6);
    }

    #[test]
    fn test_90deg_rotation() {
        let q = Quat::from_axis_angle(
            Vec3::new(0.0f32, 0.0, 1.0),  // Z axis
            std::f32::consts::FRAC_PI_2,    // 90 degrees
        );
        let v = Vec3::new(1.0f32, 0.0, 0.0);  // X axis
        let rotated = q.rotate(v);
        // Should now point along Y
        assert!((rotated.x - 0.0).abs() < 1e-5);
        assert!((rotated.y - 1.0).abs() < 1e-5);
        assert!((rotated.z - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_conjugate() {
        let q = Quat::new(1.0f32, 2.0, 3.0, 4.0);
        let c = q.conjugate();
        assert_eq!(c, Quat::new(-1.0, -2.0, -3.0, 4.0));
    }

    #[test]
    fn test_hamilton_product() {
        let q = Quat::<f32>::identity();
        let r = q * q;
        assert!((r.w - 1.0).abs() < 1e-6);
        assert!(r.x.abs() < 1e-6);
    }
}
