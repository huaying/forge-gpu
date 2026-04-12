//! # forge-runtime
//!
//! GPU runtime for Forge compute framework.
//!
//! Provides:
//! - Device discovery and management (CUDA, CPU fallback)
//! - GPU memory allocation and transfer (host ↔ device)
//! - `Array<T>` — typed GPU arrays with ownership semantics
//! - Kernel launch dispatch
//! - Stream and synchronization management
//!
//! ## Design
//!
//! The runtime wraps the CUDA driver API (or other backend APIs) behind
//! safe Rust abstractions. Memory is managed via Rust's ownership system —
//! when an `Array<T>` is dropped, GPU memory is freed automatically.
//!
//! ## Status
//!
//! 🚧 Under development — device abstraction and Array type are first targets.

/// Represents a compute device (GPU or CPU).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    /// CPU fallback (uses Rayon for parallelism).
    Cpu,
    /// CUDA GPU device with the given ordinal.
    Cuda(usize),
}

impl Device {
    /// Returns the best available device (prefers CUDA over CPU).
    pub fn best_available() -> Self {
        // TODO: Actually query CUDA device availability
        // For now, default to CPU
        Device::Cpu
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(i) => write!(f, "cuda:{}", i),
        }
    }
}

/// Top-level Forge context. Initializes the runtime.
///
/// Must be created before any GPU operations. Dropped when done.
pub struct Forge {
    device: Device,
}

impl Forge {
    /// Initialize the Forge runtime with the best available device.
    pub fn init() -> Self {
        let device = Device::best_available();
        println!("Forge initialized on {}", device);
        Self { device }
    }

    /// Initialize with a specific device.
    pub fn with_device(device: Device) -> Self {
        println!("Forge initialized on {}", device);
        Self { device }
    }

    /// Get the current device.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Synchronize all pending operations.
    pub fn synchronize(&self) {
        // TODO: cuCtxSynchronize or equivalent
    }
}

/// A typed GPU array with ownership semantics.
///
/// When this struct is dropped, the underlying GPU (or CPU) memory is freed.
/// This prevents memory leaks and double-frees at compile time.
///
/// # Type Parameters
///
/// - `T`: The element type (must implement `forge_core::Scalar` or be a Forge vector type)
pub struct Array<T: Copy> {
    data: Vec<T>,       // CPU backing store (TODO: replace with GPU allocation)
    len: usize,
    device: Device,
}

impl<T: Copy> Array<T> {
    /// Create a zero-initialized array on the given device.
    pub fn zeros(len: usize, device: Device) -> Self
    where
        T: Default,
    {
        Self {
            data: vec![T::default(); len],
            len,
            device,
        }
    }

    /// Create an array filled with a value.
    pub fn fill(len: usize, value: T, device: Device) -> Self {
        Self {
            data: vec![value; len],
            len,
            device,
        }
    }

    /// Create an array from a Vec.
    pub fn from_vec(data: Vec<T>, device: Device) -> Self {
        let len = data.len();
        Self { data, len, device }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the array is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The device this array lives on.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Copy to a Vec (device → host transfer).
    pub fn to_vec(&self) -> Vec<T> {
        self.data.clone()
    }

    /// Copy to a different device.
    pub fn to(&self, device: Device) -> Self {
        // TODO: actual device transfer via CUDA memcpy
        Self {
            data: self.data.clone(),
            len: self.len,
            device,
        }
    }
}

impl<T: Copy> std::ops::Index<usize> for Array<T> {
    type Output = T;
    fn index(&self, i: usize) -> &T {
        &self.data[i]
    }
}

impl<T: Copy> std::ops::IndexMut<usize> for Array<T> {
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.data[i]
    }
}

impl<T: Copy + std::fmt::Debug> std::fmt::Debug for Array<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Array<{}>(len={}, device={})", std::any::type_name::<T>(), self.len, self.device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_zeros() {
        let arr = Array::<f32>::zeros(100, Device::Cpu);
        assert_eq!(arr.len(), 100);
        assert_eq!(arr[0], 0.0);
        assert_eq!(arr[99], 0.0);
    }

    #[test]
    fn test_array_fill() {
        let arr = Array::<f32>::fill(10, 42.0, Device::Cpu);
        assert_eq!(arr[0], 42.0);
        assert_eq!(arr[9], 42.0);
    }

    #[test]
    fn test_array_from_vec() {
        let arr = Array::from_vec(vec![1.0f32, 2.0, 3.0], Device::Cpu);
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[1], 2.0);
    }

    #[test]
    fn test_array_to_vec() {
        let arr = Array::from_vec(vec![1, 2, 3], Device::Cpu);
        assert_eq!(arr.to_vec(), vec![1, 2, 3]);
    }

    #[test]
    fn test_array_device_transfer() {
        let arr = Array::from_vec(vec![1.0f32, 2.0], Device::Cpu);
        let gpu_arr = arr.to(Device::Cuda(0));
        assert_eq!(gpu_arr.device(), Device::Cuda(0));
        assert_eq!(gpu_arr.to_vec(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_device_display() {
        assert_eq!(format!("{}", Device::Cpu), "cpu");
        assert_eq!(format!("{}", Device::Cuda(0)), "cuda:0");
    }
}
