//! Typed array with GPU memory support.

use crate::device::Device;

#[cfg(feature = "cuda")]
use cudarc::driver::DeviceRepr;

/// Storage backend for Array data.
enum ArrayStorage<T: Copy> {
    /// CPU-backed storage.
    Cpu(Vec<T>),
    /// CUDA device-backed storage.
    #[cfg(feature = "cuda")]
    Cuda {
        /// Device memory managed by cudarc.
        buf: cudarc::driver::safe::CudaSlice<T>,
        /// Which device ordinal.
        ordinal: usize,
    },
}

/// A typed array that lives on a specific device.
///
/// When this struct is dropped, the underlying memory (CPU or GPU) is freed.
/// GPU memory is managed via cudarc's RAII wrappers — no manual free needed.
pub struct Array<T: Copy> {
    storage: ArrayStorage<T>,
    len: usize,
    device: Device,
}

// ---- CPU-only methods (always available) ----
impl<T: Copy + Default> Array<T> {
    /// Create a zero-initialized array on CPU.
    pub fn cpu_zeros(len: usize) -> Self {
        Self {
            storage: ArrayStorage::Cpu(vec![T::default(); len]),
            len,
            device: Device::Cpu,
        }
    }
}

impl<T: Copy> Array<T> {
    /// Create an array from a Vec on CPU.
    pub fn cpu_from_vec(data: Vec<T>) -> Self {
        let len = data.len();
        Self {
            storage: ArrayStorage::Cpu(data),
            len,
            device: Device::Cpu,
        }
    }

    /// Create an array filled with a value on CPU.
    pub fn cpu_fill(len: usize, value: T) -> Self {
        Self {
            storage: ArrayStorage::Cpu(vec![value; len]),
            len,
            device: Device::Cpu,
        }
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
}

// ---- Methods that work with both CPU and CUDA ----

#[cfg(feature = "cuda")]
impl<T: Copy + Default + DeviceRepr + cudarc::driver::ValidAsZeroBits> Array<T> {
    /// Create a zero-initialized array on the given device.
    pub fn zeros(len: usize, device: Device) -> Self {
        match device {
            Device::Cpu => Self {
                storage: ArrayStorage::Cpu(vec![T::default(); len]),
                len,
                device,
            },
            Device::Cuda(ordinal) => {
                let stream = crate::cuda::default_stream(ordinal);
                let buf = stream
                    .alloc_zeros::<T>(len)
                    .expect("CUDA alloc_zeros failed");
                Self {
                    storage: ArrayStorage::Cuda { buf, ordinal },
                    len,
                    device,
                }
            }
        }
    }
}

#[cfg(not(feature = "cuda"))]
impl<T: Copy + Default> Array<T> {
    /// Create a zero-initialized array (CPU-only build).
    pub fn zeros(len: usize, device: Device) -> Self {
        match device {
            Device::Cpu => Self {
                storage: ArrayStorage::Cpu(vec![T::default(); len]),
                len,
                device,
            },
            Device::Cuda(_) => panic!("CUDA support not enabled. Compile with --features cuda"),
        }
    }
}

#[cfg(feature = "cuda")]
impl<T: Copy + DeviceRepr> Array<T> {
    /// Create an array filled with a value on the given device.
    pub fn fill(len: usize, value: T, device: Device) -> Self {
        match device {
            Device::Cpu => Self {
                storage: ArrayStorage::Cpu(vec![value; len]),
                len,
                device,
            },
            Device::Cuda(ordinal) => {
                let data = vec![value; len];
                let stream = crate::cuda::default_stream(ordinal);
                let buf = stream
                    .clone_htod(data.as_slice())
                    .expect("CUDA clone_htod failed");
                Self {
                    storage: ArrayStorage::Cuda { buf, ordinal },
                    len,
                    device,
                }
            }
        }
    }

    /// Create an array from a Vec on the given device.
    pub fn from_vec(data: Vec<T>, device: Device) -> Self {
        let len = data.len();
        match device {
            Device::Cpu => Self {
                storage: ArrayStorage::Cpu(data),
                len,
                device,
            },
            Device::Cuda(ordinal) => {
                let stream = crate::cuda::default_stream(ordinal);
                let buf = stream
                    .clone_htod(data.as_slice())
                    .expect("CUDA clone_htod failed");
                Self {
                    storage: ArrayStorage::Cuda { buf, ordinal },
                    len,
                    device,
                }
            }
        }
    }

    /// Copy data back to host as a Vec.
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Default + Clone,
    {
        match &self.storage {
            ArrayStorage::Cpu(data) => data.clone(),
            ArrayStorage::Cuda { buf, ordinal } => {
                let stream = crate::cuda::default_stream(*ordinal);
                stream
                    .clone_dtoh(buf)
                    .expect("CUDA clone_dtoh failed")
            }
        }
    }

    /// Copy to a different device.
    pub fn to(&self, target_device: Device) -> Self
    where
        T: Default + Clone,
    {
        let data = self.to_vec();
        Self::from_vec(data, target_device)
    }

    /// Get a reference to the underlying CudaSlice (for kernel launches).
    pub fn cuda_slice(&self) -> Option<&cudarc::driver::safe::CudaSlice<T>> {
        match &self.storage {
            ArrayStorage::Cuda { buf, .. } => Some(buf),
            _ => None,
        }
    }

    /// Get a mutable reference to the underlying CudaSlice.
    pub fn cuda_slice_mut(&mut self) -> Option<&mut cudarc::driver::safe::CudaSlice<T>> {
        match &mut self.storage {
            ArrayStorage::Cuda { buf, .. } => Some(buf),
            _ => None,
        }
    }
}

#[cfg(not(feature = "cuda"))]
impl<T: Copy> Array<T> {
    /// Create an array filled with a value (CPU-only build).
    pub fn fill(len: usize, value: T, device: Device) -> Self {
        match device {
            Device::Cpu => Self {
                storage: ArrayStorage::Cpu(vec![value; len]),
                len,
                device,
            },
            Device::Cuda(_) => panic!("CUDA support not enabled"),
        }
    }

    /// Create from Vec (CPU-only build).
    pub fn from_vec(data: Vec<T>, device: Device) -> Self {
        let len = data.len();
        match device {
            Device::Cpu => Self {
                storage: ArrayStorage::Cpu(data),
                len,
                device,
            },
            Device::Cuda(_) => panic!("CUDA support not enabled"),
        }
    }

    /// Copy to Vec (CPU-only build).
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        match &self.storage {
            ArrayStorage::Cpu(data) => data.clone(),
        }
    }

    /// Copy to device (CPU-only build).
    pub fn to(&self, target_device: Device) -> Self
    where
        T: Clone,
    {
        let data = self.to_vec();
        Self::from_vec(data, target_device)
    }
}

// CPU-only indexing
impl<T: Copy> std::ops::Index<usize> for Array<T> {
    type Output = T;
    fn index(&self, i: usize) -> &T {
        match &self.storage {
            ArrayStorage::Cpu(data) => &data[i],
            #[cfg(feature = "cuda")]
            ArrayStorage::Cuda { .. } => {
                panic!("Cannot index GPU array directly. Use to_vec() first.")
            }
        }
    }
}

impl<T: Copy> std::ops::IndexMut<usize> for Array<T> {
    fn index_mut(&mut self, i: usize) -> &mut T {
        match &mut self.storage {
            ArrayStorage::Cpu(data) => &mut data[i],
            #[cfg(feature = "cuda")]
            ArrayStorage::Cuda { .. } => {
                panic!("Cannot index GPU array directly. Use to_vec() first.")
            }
        }
    }
}

impl<T: Copy + std::fmt::Debug> std::fmt::Debug for Array<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Array<{}>(len={}, device={})",
            std::any::type_name::<T>(),
            self.len,
            self.device
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_cpu_zeros() {
        let arr = Array::<f32>::zeros(100, Device::Cpu);
        assert_eq!(arr.len(), 100);
        assert_eq!(arr[0], 0.0);
        assert_eq!(arr[99], 0.0);
    }

    #[test]
    fn test_array_cpu_fill() {
        let arr = Array::<f32>::fill(10, 42.0, Device::Cpu);
        assert_eq!(arr[0], 42.0);
        assert_eq!(arr[9], 42.0);
    }

    #[test]
    fn test_array_cpu_from_vec() {
        let arr = Array::from_vec(vec![1.0f32, 2.0, 3.0], Device::Cpu);
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[1], 2.0);
    }

    #[test]
    fn test_array_cpu_to_vec() {
        let arr = Array::from_vec(vec![1.0f32, 2.0, 3.0], Device::Cpu);
        assert_eq!(arr.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_device_display() {
        assert_eq!(format!("{}", Device::Cpu), "cpu");
        assert_eq!(format!("{}", Device::Cuda(0)), "cuda:0");
    }

    #[cfg(feature = "cuda")]
    mod gpu_tests {
        use super::*;

        fn skip_if_no_gpu() -> bool {
            crate::cuda::init();
            crate::cuda::device_count() == 0
        }

        #[test]
        fn test_array_gpu_roundtrip() {
            if skip_if_no_gpu() {
                eprintln!("Skipping: no GPU");
                return;
            }
            let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
            let arr = Array::from_vec(data.clone(), Device::Cuda(0));
            assert_eq!(arr.len(), 5);
            assert_eq!(arr.device(), Device::Cuda(0));
            let result = arr.to_vec();
            assert_eq!(result, data);
        }

        #[test]
        fn test_array_gpu_zeros() {
            if skip_if_no_gpu() {
                return;
            }
            let arr = Array::<f32>::zeros(1000, Device::Cuda(0));
            let result = arr.to_vec();
            assert!(result.iter().all(|&x| x == 0.0));
        }

        #[test]
        fn test_array_gpu_fill() {
            if skip_if_no_gpu() {
                return;
            }
            let arr = Array::<f32>::fill(100, 3.14, Device::Cuda(0));
            let result = arr.to_vec();
            assert!(result.iter().all(|&x| (x - 3.14).abs() < 1e-6));
        }

        #[test]
        fn test_array_gpu_large() {
            if skip_if_no_gpu() {
                return;
            }
            let n = 1_000_000;
            let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let arr = Array::from_vec(data.clone(), Device::Cuda(0));
            let result = arr.to_vec();
            assert_eq!(result.len(), n);
            assert_eq!(result[0], 0.0);
            assert_eq!(result[n - 1], (n - 1) as f32);
        }

        #[test]
        fn test_array_gpu_to_cpu() {
            if skip_if_no_gpu() {
                return;
            }
            let data = vec![10.0f32, 20.0, 30.0];
            let gpu_arr = Array::from_vec(data.clone(), Device::Cuda(0));
            let cpu_arr = gpu_arr.to(Device::Cpu);
            assert_eq!(cpu_arr.device(), Device::Cpu);
            assert_eq!(cpu_arr.to_vec(), data);
        }
    }
}
