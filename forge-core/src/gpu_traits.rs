//! cudarc trait implementations for Forge vector types.
//!
//! Enabled by the `cuda` feature. These impls allow Vec2/3/4 types
//! to be used in GPU arrays via cudarc.

use crate::vec::*;
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};

// Vec2
unsafe impl DeviceRepr for Vec2<f32> {}
unsafe impl ValidAsZeroBits for Vec2<f32> {}
unsafe impl DeviceRepr for Vec2<f64> {}
unsafe impl ValidAsZeroBits for Vec2<f64> {}

// Vec3
unsafe impl DeviceRepr for Vec3<f32> {}
unsafe impl ValidAsZeroBits for Vec3<f32> {}
unsafe impl DeviceRepr for Vec3<f64> {}
unsafe impl ValidAsZeroBits for Vec3<f64> {}

// Vec4
unsafe impl DeviceRepr for Vec4<f32> {}
unsafe impl ValidAsZeroBits for Vec4<f32> {}
unsafe impl DeviceRepr for Vec4<f64> {}
unsafe impl ValidAsZeroBits for Vec4<f64> {}
