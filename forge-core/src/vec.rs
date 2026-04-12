//! Vector types for GPU computation.
//!
//! `Vec2<T>`, `Vec3<T>`, `Vec4<T>` — generic over scalar type,
//! `#[repr(C)]` for direct GPU memory compatibility.

use crate::scalar::{Float, Scalar};
use std::fmt;
use std::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};

// ---------------------------------------------------------------------------
// Vec2
// ---------------------------------------------------------------------------

/// A 2-component vector.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Default)]
pub struct Vec2<T: Scalar> {
    pub x: T,
    pub y: T,
}

impl<T: Scalar> Vec2<T> {
    #[inline]
    pub const fn new(x: T, y: T) -> Self {
        Self { x, y }
    }

    #[inline]
    pub fn splat(v: T) -> Self {
        Self { x: v, y: v }
    }

    #[inline]
    pub fn zero() -> Self {
        Self { x: T::ZERO, y: T::ZERO }
    }

    #[inline]
    pub fn dot(self, other: Self) -> T {
        self.x * other.x + self.y * other.y
    }
}

impl<T: Float> Vec2<T> {
    #[inline]
    pub fn length(self) -> T {
        self.dot(self).sqrt()
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        Self { x: self.x / len, y: self.y / len }
    }
}

impl<T: Scalar> fmt::Debug for Vec2<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vec2({}, {})", self.x, self.y)
    }
}

impl<T: Scalar> fmt::Display for Vec2<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

impl<T: Scalar> Add for Vec2<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self { x: self.x + rhs.x, y: self.y + rhs.y }
    }
}

impl<T: Scalar> Sub for Vec2<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self { x: self.x - rhs.x, y: self.y - rhs.y }
    }
}

impl<T: Scalar> Mul<T> for Vec2<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: T) -> Self {
        Self { x: self.x * rhs, y: self.y * rhs }
    }
}

impl<T: Scalar> Div<T> for Vec2<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: T) -> Self {
        Self { x: self.x / rhs, y: self.y / rhs }
    }
}

impl<T: Scalar> Neg for Vec2<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self { x: -self.x, y: -self.y }
    }
}

impl<T: Scalar> Index<usize> for Vec2<T> {
    type Output = T;
    #[inline]
    fn index(&self, i: usize) -> &T {
        match i {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("Vec2 index out of bounds: {}", i),
        }
    }
}

impl<T: Scalar> IndexMut<usize> for Vec2<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("Vec2 index out of bounds: {}", i),
        }
    }
}

// ---------------------------------------------------------------------------
// Vec3
// ---------------------------------------------------------------------------

/// A 3-component vector.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Default)]
pub struct Vec3<T: Scalar> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: Scalar> Vec3<T> {
    #[inline]
    pub const fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn splat(v: T) -> Self {
        Self { x: v, y: v, z: v }
    }

    #[inline]
    pub fn zero() -> Self {
        Self { x: T::ZERO, y: T::ZERO, z: T::ZERO }
    }

    #[inline]
    pub fn dot(self, other: Self) -> T {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
}

impl<T: Float> Vec3<T> {
    #[inline]
    pub fn length(self) -> T {
        self.dot(self).sqrt()
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        Self { x: self.x / len, y: self.y / len, z: self.z / len }
    }
}

impl<T: Scalar> fmt::Debug for Vec3<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vec3({}, {}, {})", self.x, self.y, self.z)
    }
}

impl<T: Scalar> fmt::Display for Vec3<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

impl<T: Scalar> Add for Vec3<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z }
    }
}

impl<T: Scalar> Sub for Vec3<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z }
    }
}

impl<T: Scalar> Mul<T> for Vec3<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: T) -> Self {
        Self { x: self.x * rhs, y: self.y * rhs, z: self.z * rhs }
    }
}

impl<T: Scalar> Div<T> for Vec3<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: T) -> Self {
        Self { x: self.x / rhs, y: self.y / rhs, z: self.z / rhs }
    }
}

impl<T: Scalar> Neg for Vec3<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self { x: -self.x, y: -self.y, z: -self.z }
    }
}

impl<T: Scalar> Index<usize> for Vec3<T> {
    type Output = T;
    #[inline]
    fn index(&self, i: usize) -> &T {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Vec3 index out of bounds: {}", i),
        }
    }
}

impl<T: Scalar> IndexMut<usize> for Vec3<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Vec3 index out of bounds: {}", i),
        }
    }
}

// ---------------------------------------------------------------------------
// Vec4
// ---------------------------------------------------------------------------

/// A 4-component vector.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Default)]
pub struct Vec4<T: Scalar> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

impl<T: Scalar> Vec4<T> {
    #[inline]
    pub const fn new(x: T, y: T, z: T, w: T) -> Self {
        Self { x, y, z, w }
    }

    #[inline]
    pub fn splat(v: T) -> Self {
        Self { x: v, y: v, z: v, w: v }
    }

    #[inline]
    pub fn zero() -> Self {
        Self { x: T::ZERO, y: T::ZERO, z: T::ZERO, w: T::ZERO }
    }

    #[inline]
    pub fn dot(self, other: Self) -> T {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }
}

impl<T: Float> Vec4<T> {
    #[inline]
    pub fn length(self) -> T {
        self.dot(self).sqrt()
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        Self { x: self.x / len, y: self.y / len, z: self.z / len, w: self.w / len }
    }
}

impl<T: Scalar> fmt::Debug for Vec4<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vec4({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

impl<T: Scalar> fmt::Display for Vec4<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

impl<T: Scalar> Add for Vec4<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z, w: self.w + rhs.w }
    }
}

impl<T: Scalar> Sub for Vec4<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z, w: self.w - rhs.w }
    }
}

impl<T: Scalar> Mul<T> for Vec4<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: T) -> Self {
        Self { x: self.x * rhs, y: self.y * rhs, z: self.z * rhs, w: self.w * rhs }
    }
}

impl<T: Scalar> Div<T> for Vec4<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: T) -> Self {
        Self { x: self.x / rhs, y: self.y / rhs, z: self.z / rhs, w: self.w / rhs }
    }
}

impl<T: Scalar> Neg for Vec4<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self { x: -self.x, y: -self.y, z: -self.z, w: -self.w }
    }
}

impl<T: Scalar> Index<usize> for Vec4<T> {
    type Output = T;
    #[inline]
    fn index(&self, i: usize) -> &T {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("Vec4 index out of bounds: {}", i),
        }
    }
}

impl<T: Scalar> IndexMut<usize> for Vec4<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("Vec4 index out of bounds: {}", i),
        }
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Compute the dot product of two vectors.
#[inline]
pub fn dot<T: Scalar>(a: Vec3<T>, b: Vec3<T>) -> T {
    a.dot(b)
}

/// Compute the cross product of two 3D vectors.
#[inline]
pub fn cross<T: Scalar>(a: Vec3<T>, b: Vec3<T>) -> Vec3<T> {
    a.cross(b)
}

/// Compute the length of a 3D vector.
#[inline]
pub fn length<T: Float>(v: Vec3<T>) -> T {
    v.length()
}

/// Normalize a 3D vector.
#[inline]
pub fn normalize<T: Float>(v: Vec3<T>) -> Vec3<T> {
    v.normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec3_basic() {
        let a = Vec3::new(1.0f32, 2.0, 3.0);
        let b = Vec3::new(4.0f32, 5.0, 6.0);

        let c = a + b;
        assert_eq!(c, Vec3::new(5.0, 7.0, 9.0));

        let d = a - b;
        assert_eq!(d, Vec3::new(-3.0, -3.0, -3.0));

        let e = a * 2.0;
        assert_eq!(e, Vec3::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn test_vec3_dot() {
        let a = Vec3::new(1.0f32, 0.0, 0.0);
        let b = Vec3::new(0.0f32, 1.0, 0.0);
        assert_eq!(a.dot(b), 0.0);

        let c = Vec3::new(1.0f32, 2.0, 3.0);
        assert_eq!(c.dot(c), 14.0);
    }

    #[test]
    fn test_vec3_cross() {
        let x = Vec3::new(1.0f32, 0.0, 0.0);
        let y = Vec3::new(0.0f32, 1.0, 0.0);
        let z = x.cross(y);
        assert_eq!(z, Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_vec3_normalize() {
        let v = Vec3::new(3.0f32, 0.0, 4.0);
        let n = v.normalize();
        assert!((n.length() - 1.0).abs() < 1e-6);
        assert!((n.x - 0.6).abs() < 1e-6);
        assert!((n.z - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_vec2_basic() {
        let a = Vec2::new(1.0f32, 2.0);
        let b = Vec2::new(3.0f32, 4.0);
        assert_eq!(a + b, Vec2::new(4.0, 6.0));
        assert_eq!(a.dot(b), 11.0);
    }

    #[test]
    fn test_vec4_basic() {
        let a = Vec4::new(1.0f32, 2.0, 3.0, 4.0);
        let b = Vec4::new(5.0f32, 6.0, 7.0, 8.0);
        assert_eq!(a + b, Vec4::new(6.0, 8.0, 10.0, 12.0));
        assert_eq!(a.dot(b), 70.0);
    }

    #[test]
    fn test_index() {
        let v = Vec3::new(10.0f32, 20.0, 30.0);
        assert_eq!(v[0], 10.0);
        assert_eq!(v[1], 20.0);
        assert_eq!(v[2], 30.0);
    }
}
