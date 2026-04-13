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
        let two = T::ONE + T::ONE;
        let half = angle / two;
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

    /// Rotate a 3D vector by this quaternion (alias for `rotate`).
    #[inline]
    pub fn rotate_vec(&self, v: Vec3<T>) -> Vec3<T> {
        self.rotate(v)
    }

    /// Rotate a 3D vector by this quaternion.
    #[inline]
    pub fn rotate(self, v: Vec3<T>) -> Vec3<T> {
        // q * v * q^-1, optimized
        let qv = Vec3::new(self.x, self.y, self.z);
        let two = T::ONE + T::ONE;
        let t = qv.cross(v) * two;
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

    /// Spherical linear interpolation between two unit quaternions.
    pub fn slerp(a: Quat<T>, b: Quat<T>, t: T) -> Quat<T> {
        let mut d = a.dot(b);
        // If dot is negative, negate one quaternion to take the shorter arc
        let mut b = b;
        if d < T::ZERO {
            b = -b;
            d = T::ZERO - d;
        }
        // Build threshold ~0.9995 from T::ONE
        // 0.9995 = 1 - 0.0005 = 1 - 1/2000
        // We'll use a generous threshold via repeated halving
        let two = T::ONE + T::ONE;
        let ten = two * two * two + two; // 10
        let hundred = ten * ten;
        let two_thousand = hundred * ten * two;
        let threshold = T::ONE - T::ONE / two_thousand;

        if d > threshold {
            // Linear interpolation for very close quaternions
            let one_t = T::ONE - t;
            Quat::new(
                one_t * a.x + t * b.x,
                one_t * a.y + t * b.y,
                one_t * a.z + t * b.z,
                one_t * a.w + t * b.w,
            ).normalize()
        } else {
            let theta = d.acos_safe();
            let sin_theta = theta.sin();
            let wa = ((T::ONE - t) * theta).sin() / sin_theta;
            let wb = (t * theta).sin() / sin_theta;
            Quat::new(
                wa * a.x + wb * b.x,
                wa * a.y + wb * b.y,
                wa * a.z + wb * b.z,
                wa * a.w + wb * b.w,
            ).normalize()
        }
    }

    /// Create a quaternion from Euler angles (roll, pitch, yaw) in radians.
    /// Uses ZYX convention (yaw around Z, pitch around Y, roll around X).
    pub fn from_euler(roll: T, pitch: T, yaw: T) -> Quat<T> {
        let two = T::ONE + T::ONE;
        let hr = roll / two;
        let hp = pitch / two;
        let hy = yaw / two;
        let (sr, cr) = (hr.sin(), hr.cos());
        let (sp, cp) = (hp.sin(), hp.cos());
        let (sy, cy) = (hy.sin(), hy.cos());
        Quat::new(
            sr * cp * cy - cr * sp * sy,  // x
            cr * sp * cy + sr * cp * sy,  // y
            cr * cp * sy - sr * sp * cy,  // z
            cr * cp * cy + sr * sp * sy,  // w
        )
    }

    /// Convert to Euler angles (roll, pitch, yaw) in radians.
    /// Uses ZYX convention. Returns (roll, pitch, yaw).
    pub fn to_euler(&self) -> (T, T, T) {
        let two = T::ONE + T::ONE;
        // Roll (x-axis)
        let sinr_cosp = two * (self.w * self.x + self.y * self.z);
        let cosr_cosp = T::ONE - two * (self.x * self.x + self.y * self.y);
        let roll = sinr_cosp.atan2(cosr_cosp);

        // Pitch (y-axis)
        let sinp = two * (self.w * self.y - self.z * self.x);
        let pitch = sinp.asin_safe();

        // Yaw (z-axis)
        let siny_cosp = two * (self.w * self.z + self.x * self.y);
        let cosy_cosp = T::ONE - two * (self.y * self.y + self.z * self.z);
        let yaw = siny_cosp.atan2(cosy_cosp);

        (roll, pitch, yaw)
    }

    /// Extract the rotation axis and angle from a unit quaternion.
    /// Returns (axis, angle) where angle is in radians [0, 2π).
    /// If the rotation is near zero, returns (Vec3(1,0,0), 0).
    pub fn to_axis_angle(self) -> (Vec3<T>, T) {
        let q = self.normalize();
        let two = T::ONE + T::ONE;
        let w_clamped = q.w.min(T::ONE).max(T::ZERO - T::ONE);
        let angle = two * w_clamped.acos_safe();
        let sin_half = (T::ONE - w_clamped * w_clamped).sqrt();
        // Epsilon ≈ 1e-7 built from generic arithmetic
        let ten = (T::ONE + T::ONE) * (T::ONE + T::ONE) + T::ONE + T::ONE +
            T::ONE + T::ONE + T::ONE + T::ONE; // 10
        let million = ten * ten * ten * ten * ten * ten; // 1e6
        let eps = T::ONE / (million * ten); // 1e-7
        if sin_half > eps {
            let inv = T::ONE / sin_half;
            (Vec3::new(q.x * inv, q.y * inv, q.z * inv), angle)
        } else {
            (Vec3::new(T::ONE, T::ZERO, T::ZERO), T::ZERO)
        }
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

    #[test]
    fn test_slerp_endpoints() {
        let a = Quat::<f32>::identity();
        let b = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), std::f32::consts::FRAC_PI_2);
        // slerp at t=0 should give a
        let s0 = Quat::slerp(a, b, 0.0);
        assert!((s0.w - a.w).abs() < 1e-5);
        assert!((s0.x - a.x).abs() < 1e-5);
        // slerp at t=1 should give b
        let s1 = Quat::slerp(a, b, 1.0);
        assert!((s1.w - b.w).abs() < 1e-5);
        assert!((s1.z - b.z).abs() < 1e-5);
    }

    #[test]
    fn test_axis_angle_roundtrip() {
        let axis = Vec3::new(0.0f32, 1.0, 0.0);
        let angle = 1.23f32;
        let q = Quat::from_axis_angle(axis, angle);
        let (ax2, ang2) = q.to_axis_angle();
        assert!((ax2.x - axis.x).abs() < 1e-5);
        assert!((ax2.y - axis.y).abs() < 1e-5);
        assert!((ax2.z - axis.z).abs() < 1e-5);
        assert!((ang2 - angle).abs() < 1e-5);
    }

    #[test]
    fn test_euler_roundtrip() {
        let roll = 0.3f32;
        let pitch = 0.5f32;
        let yaw = 0.7f32;
        let q = Quat::from_euler(roll, pitch, yaw);
        let (r2, p2, y2) = q.to_euler();
        assert!((r2 - roll).abs() < 1e-5, "roll: {} vs {}", r2, roll);
        assert!((p2 - pitch).abs() < 1e-5, "pitch: {} vs {}", p2, pitch);
        assert!((y2 - yaw).abs() < 1e-5, "yaw: {} vs {}", y2, yaw);
    }

    #[test]
    fn test_rotate_vec() {
        let q = Quat::from_axis_angle(Vec3::new(0.0f32, 0.0, 1.0), std::f32::consts::PI);
        let v = Vec3::new(1.0f32, 0.0, 0.0);
        let r = q.rotate_vec(&v);
        assert!((r.x - (-1.0)).abs() < 1e-5);
        assert!((r.y - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_inverse() {
        let q = Quat::from_axis_angle(Vec3::new(1.0f32, 0.0, 0.0), 1.0);
        let qi = q.inverse();
        let product = q * qi;
        assert!((product.w - 1.0).abs() < 1e-5);
        assert!(product.x.abs() < 1e-5);
        assert!(product.y.abs() < 1e-5);
        assert!(product.z.abs() < 1e-5);
    }
}
