//! Typed array with GPU memory support, up to 4 dimensions.

use crate::device::Device;

#[cfg(feature = "cuda")]
use cudarc::driver::DeviceRepr;

/// Shape descriptor for up to 4 dimensions (row-major / C-order).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub dims: [usize; 4],
    pub ndim: u8,
}

impl Shape {
    pub fn new_1d(n: usize) -> Self {
        Shape { dims: [n, 1, 1, 1], ndim: 1 }
    }
    pub fn new_2d(d0: usize, d1: usize) -> Self {
        Shape { dims: [d0, d1, 1, 1], ndim: 2 }
    }
    pub fn new_3d(d0: usize, d1: usize, d2: usize) -> Self {
        Shape { dims: [d0, d1, d2, 1], ndim: 3 }
    }
    pub fn new_4d(d0: usize, d1: usize, d2: usize, d3: usize) -> Self {
        Shape { dims: [d0, d1, d2, d3], ndim: 4 }
    }

    /// Total number of elements (product of all dimensions).
    pub fn total(&self) -> usize {
        let n = self.ndim as usize;
        self.dims[..n].iter().product()
    }

    /// Row-major strides for indexing.
    pub fn strides(&self) -> [usize; 4] {
        let n = self.ndim as usize;
        let mut s = [0usize; 4];
        if n == 0 { return s; }
        s[n - 1] = 1;
        for i in (0..n.saturating_sub(1)).rev() {
            s[i] = s[i + 1] * self.dims[i + 1];
        }
        s
    }

    /// Flat index for 2D (row, col).
    #[inline]
    pub fn index_2d(&self, i: usize, j: usize) -> usize {
        debug_assert!(self.ndim >= 2);
        i * self.dims[1] + j
    }

    /// Flat index for 3D (i, j, k).
    #[inline]
    pub fn index_3d(&self, i: usize, j: usize, k: usize) -> usize {
        debug_assert!(self.ndim >= 3);
        (i * self.dims[1] + j) * self.dims[2] + k
    }

    /// Flat index for 4D (i, j, k, l).
    #[inline]
    pub fn index_4d(&self, i: usize, j: usize, k: usize, l: usize) -> usize {
        debug_assert!(self.ndim >= 4);
        ((i * self.dims[1] + j) * self.dims[2] + k) * self.dims[3] + l
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n = self.ndim as usize;
        write!(f, "(")?;
        for i in 0..n {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{}", self.dims[i])?;
        }
        write!(f, ")")
    }
}

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

/// A typed array that lives on a specific device, with up to 4 dimensions.
///
/// Data is stored in row-major (C-order) layout. Multi-dimensional arrays
/// are views over a flat contiguous buffer — no extra indirection.
///
/// When this struct is dropped, the underlying memory (CPU or GPU) is freed.
/// GPU memory is managed via cudarc's RAII wrappers — no manual free needed.
pub struct Array<T: Copy> {
    storage: ArrayStorage<T>,
    len: usize,
    device: Device,
    shape: Shape,
}

/// Type aliases for multi-dimensional arrays (same underlying type).
pub type Array2D<T> = Array<T>;
pub type Array3D<T> = Array<T>;
pub type Array4D<T> = Array<T>;

// ---- CPU-only methods (always available) ----
impl<T: Copy + Default> Array<T> {
    /// Create a zero-initialized array on CPU.
    pub fn cpu_zeros(len: usize) -> Self {
        Self {
            storage: ArrayStorage::Cpu(vec![T::default(); len]),
            len,
            device: Device::Cpu,
            shape: Shape::new_1d(len),
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
            shape: Shape::new_1d(len),
        }
    }

    /// Create an array filled with a value on CPU.
    pub fn cpu_fill(len: usize, value: T) -> Self {
        Self {
            storage: ArrayStorage::Cpu(vec![value; len]),
            len,
            device: Device::Cpu,
            shape: Shape::new_1d(len),
        }
    }

    /// Number of elements (total across all dimensions).
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

    /// The shape of this array.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Number of dimensions (1-4).
    pub fn ndim(&self) -> usize {
        self.shape.ndim as usize
    }

    /// Reshape without copying data. Panics if total elements differ.
    pub fn reshape(&mut self, new_shape: Shape) {
        assert_eq!(
            self.len,
            new_shape.total(),
            "Cannot reshape array of {} elements to shape {} ({} elements)",
            self.len,
            new_shape,
            new_shape.total()
        );
        self.shape = new_shape;
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
                shape: Shape::new_1d(len),
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
                    shape: Shape::new_1d(len),
                }
            }
        }
    }

    /// Create a zero-initialized array with a specific shape.
    pub fn zeros_nd(shape: Shape, device: Device) -> Self {
        let len = shape.total();
        match device {
            Device::Cpu => Self {
                storage: ArrayStorage::Cpu(vec![T::default(); len]),
                len,
                device,
                shape,
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
                    shape,
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
                shape: Shape::new_1d(len),
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
                shape: Shape::new_1d(len),
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
                    shape: Shape::new_1d(len),
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
                shape: Shape::new_1d(len),
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
                    shape: Shape::new_1d(len),
                }
            }
        }
    }

    /// Create an array from a Vec with a specific shape.

    /// Create an array from a slice (borrows data, no Vec clone needed).
    /// For GPU, this is faster than from_vec for large arrays since
    /// it avoids the CPU-side Vec clone.
    pub fn from_slice(data: &[T], device: Device) -> Self {
        let len = data.len();
        match device {
            Device::Cpu => Self {
                storage: ArrayStorage::Cpu(data.to_vec()),
                len,
                device,
                shape: Shape::new_1d(len),
            },
            Device::Cuda(ordinal) => {
                let stream = crate::cuda::default_stream(ordinal);
                let buf = stream
                    .clone_htod(data)
                    .expect("CUDA clone_htod failed");
                Self {
                    storage: ArrayStorage::Cuda { buf, ordinal },
                    len,
                    device,
                    shape: Shape::new_1d(len),
                }
            }
        }
    }

    /// Create an array from a Vec with a specific shape.
    pub fn from_vec_nd(data: Vec<T>, shape: Shape, device: Device) -> Self {
        let len = data.len();
        assert_eq!(len, shape.total(), "Data length {} != shape total {}", len, shape.total());
        match device {
            Device::Cpu => Self {
                storage: ArrayStorage::Cpu(data),
                len,
                device,
                shape,
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
                    shape,
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

    /// Copy to a different device (preserves shape).
    pub fn to(&self, target_device: Device) -> Self
    where
        T: Default + Clone,
    {
        let data = self.to_vec();
        let mut arr = Self::from_vec(data, target_device);
        arr.shape = self.shape;
        arr
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
                shape: Shape::new_1d(len),
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
                shape: Shape::new_1d(len),
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

    /// Copy to device (CPU-only build, preserves shape).
    pub fn to(&self, target_device: Device) -> Self
    where
        T: Clone,
    {
        let data = self.to_vec();
        let mut arr = Self::from_vec(data, target_device);
        arr.shape = self.shape;
        arr
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
            "Array<{}>(shape={}, device={})",
            std::any::type_name::<T>(),
            self.shape,
            self.device
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Shape tests ──

    #[test]
    fn test_shape_1d() {
        let s = Shape::new_1d(100);
        assert_eq!(s.total(), 100);
        assert_eq!(s.ndim, 1);
        assert_eq!(s.strides(), [1, 0, 0, 0]);
        assert_eq!(format!("{}", s), "(100)");
    }

    #[test]
    fn test_shape_2d() {
        let s = Shape::new_2d(3, 4);
        assert_eq!(s.total(), 12);
        assert_eq!(s.ndim, 2);
        assert_eq!(s.strides(), [4, 1, 0, 0]);
        assert_eq!(s.index_2d(0, 0), 0);
        assert_eq!(s.index_2d(0, 3), 3);
        assert_eq!(s.index_2d(1, 0), 4);
        assert_eq!(s.index_2d(2, 3), 11);
    }

    #[test]
    fn test_shape_3d() {
        let s = Shape::new_3d(2, 3, 4);
        assert_eq!(s.total(), 24);
        assert_eq!(s.strides(), [12, 4, 1, 0]);
        assert_eq!(s.index_3d(0, 0, 0), 0);
        assert_eq!(s.index_3d(1, 2, 3), 23);
    }

    #[test]
    fn test_shape_4d() {
        let s = Shape::new_4d(2, 3, 4, 5);
        assert_eq!(s.total(), 120);
        assert_eq!(s.strides(), [60, 20, 5, 1]);
        assert_eq!(s.index_4d(0, 0, 0, 0), 0);
        assert_eq!(s.index_4d(1, 2, 3, 4), 119);
    }

    // ── Array with shape tests ──
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

        #[test]
        fn test_array_2d_gpu() {
            if skip_if_no_gpu() { return; }
            let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
            let mut arr = Array::from_vec(data.clone(), Device::Cuda(0));
            arr.reshape(Shape::new_2d(3, 4));
            assert_eq!(arr.ndim(), 2);
            assert_eq!(arr.shape().dims[0], 3);
            assert_eq!(arr.shape().dims[1], 4);
            let result = arr.to_vec();
            assert_eq!(result, data);
        }

        #[test]
        fn test_array_zeros_nd_gpu() {
            if skip_if_no_gpu() { return; }
            let arr = Array::<f32>::zeros_nd(Shape::new_3d(2, 3, 4), Device::Cuda(0));
            assert_eq!(arr.len(), 24);
            assert_eq!(arr.ndim(), 3);
            let result = arr.to_vec();
            assert!(result.iter().all(|&x| x == 0.0));
        }

        #[test]
        fn test_array_from_vec_nd_gpu() {
            if skip_if_no_gpu() { return; }
            let data: Vec<f32> = (0..60).map(|i| i as f32).collect();
            let arr = Array::from_vec_nd(data.clone(), Shape::new_3d(3, 4, 5), Device::Cuda(0));
            assert_eq!(arr.len(), 60);
            assert_eq!(arr.ndim(), 3);
            let result = arr.to_vec();
            assert_eq!(result, data);
        }

        #[test]
        fn test_array_to_preserves_shape() {
            if skip_if_no_gpu() { return; }
            let arr = Array::<f32>::zeros_nd(Shape::new_2d(8, 16), Device::Cuda(0));
            let cpu = arr.to(Device::Cpu);
            assert_eq!(cpu.ndim(), 2);
            assert_eq!(cpu.shape().dims[0], 8);
            assert_eq!(cpu.shape().dims[1], 16);
        }
    }
}
