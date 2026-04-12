//! # Sparse matrices — CSR format with GPU SpMV.
//!
//! Compressed Sparse Row (CSR) format for sparse matrix storage.
//! Supports sparse matrix-vector multiplication (SpMV) on GPU.

use crate::{Array, Device, ForgeError};

/// Compressed Sparse Row (CSR) sparse matrix.
///
/// Stores a matrix of shape (rows × cols) with nnz non-zero entries.
///
/// Layout:
/// - `row_ptr[i]` = start index in `col_idx`/`values` for row i
/// - `row_ptr[rows]` = nnz (total non-zeros)
/// - `col_idx[k]` = column index of k-th non-zero
/// - `values[k]` = value of k-th non-zero
pub struct CsrMatrix {
    pub rows: usize,
    pub cols: usize,
    pub nnz: usize,
    /// Row pointers. Length = rows + 1.
    pub row_ptr: Vec<u32>,
    /// Column indices. Length = nnz.
    pub col_idx: Vec<u32>,
    /// Non-zero values. Length = nnz.
    pub values: Vec<f32>,
}

impl CsrMatrix {
    /// Create an empty CSR matrix.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            nnz: 0,
            row_ptr: vec![0; rows + 1],
            col_idx: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Build CSR from a list of (row, col, value) triplets.
    ///
    /// Triplets do not need to be sorted. Duplicate entries are summed.
    pub fn from_triplets(rows: usize, cols: usize, triplets: &[(u32, u32, f32)]) -> Self {
        // Sort by row, then column
        let mut sorted = triplets.to_vec();
        sorted.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        // Merge duplicates
        let mut merged: Vec<(u32, u32, f32)> = Vec::new();
        for &(r, c, v) in &sorted {
            if let Some(last) = merged.last_mut() {
                if last.0 == r && last.1 == c {
                    last.2 += v;
                    continue;
                }
            }
            merged.push((r, c, v));
        }

        let nnz = merged.len();
        let mut row_ptr = vec![0u32; rows + 1];
        let mut col_idx = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        for &(r, c, v) in &merged {
            row_ptr[r as usize + 1] += 1;
            col_idx.push(c);
            values.push(v);
        }

        // Prefix sum
        for i in 0..rows {
            row_ptr[i + 1] += row_ptr[i];
        }

        Self {
            rows,
            cols,
            nnz,
            row_ptr,
            col_idx,
            values,
        }
    }

    /// Create a diagonal matrix.
    pub fn diagonal(n: usize, diag: &[f32]) -> Self {
        let mut triplets = Vec::with_capacity(n);
        for i in 0..n {
            triplets.push((i as u32, i as u32, diag[i]));
        }
        Self::from_triplets(n, n, &triplets)
    }

    /// Create an identity matrix.
    pub fn identity(n: usize) -> Self {
        Self::diagonal(n, &vec![1.0; n])
    }

    /// Get value at (row, col). Returns 0.0 if not present.
    pub fn get(&self, row: usize, col: usize) -> f32 {
        let start = self.row_ptr[row] as usize;
        let end = self.row_ptr[row + 1] as usize;
        for k in start..end {
            if self.col_idx[k] as usize == col {
                return self.values[k];
            }
        }
        0.0
    }

    /// Sparse matrix-vector multiply: y = A * x (CPU).
    pub fn spmv(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.cols, "x length must match cols");
        let mut y = vec![0.0f32; self.rows];
        for row in 0..self.rows {
            let start = self.row_ptr[row] as usize;
            let end = self.row_ptr[row + 1] as usize;
            let mut sum = 0.0f32;
            for k in start..end {
                sum += self.values[k] * x[self.col_idx[k] as usize];
            }
            y[row] = sum;
        }
        y
    }

    /// Sparse matrix-vector multiply on GPU: y = A * x.
    ///
    /// Uses a simple one-thread-per-row approach.
    #[cfg(feature = "cuda")]
    pub fn spmv_gpu(
        &self,
        x: &Array<f32>,
        y: &mut Array<f32>,
        device: Device,
    ) -> Result<(), ForgeError> {
        use crate::cuda;

        // Upload CSR arrays to GPU
        let row_ptr = Array::from_vec(self.row_ptr.clone(), device);
        let col_idx = Array::from_vec(self.col_idx.clone(), device);
        let values = Array::from_vec(self.values.clone(), device);

        // Compile SpMV kernel
        let source = r#"
extern "C" __global__ void spmv(
    const unsigned int* row_ptr,
    const unsigned int* col_idx,
    const float* values,
    const float* x,
    float* y,
    int rows
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < rows) {
        float sum = 0.0f;
        unsigned int start = row_ptr[tid];
        unsigned int end = row_ptr[tid + 1];
        for (unsigned int k = start; k < end; k++) {
            sum += values[k] * x[col_idx[k]];
        }
        y[tid] = sum;
    }
}
"#;

        let kernel = crate::CompiledKernel::compile(source, "spmv")?;
        let ordinal = match device {
            Device::Cuda(i) => i,
            _ => 0,
        };
        let func = kernel.get_function(ordinal)?;
        let stream = cuda::default_stream(ordinal);
        let config = cuda::LaunchConfig::for_num_elems(self.rows as u32);

        unsafe {
            use cuda::PushKernelArg;
            let rows_i32 = self.rows as i32;
            let mut builder = stream.launch_builder(&func);
            builder.arg(row_ptr.cuda_slice().unwrap());
            builder.arg(col_idx.cuda_slice().unwrap());
            builder.arg(values.cuda_slice().unwrap());
            builder.arg(x.cuda_slice().unwrap());
            builder.arg(y.cuda_slice_mut().unwrap());
            builder.arg(&rows_i32);
            builder.launch(config)
                .map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
        }

        stream.synchronize()
            .map_err(|e| ForgeError::SyncFailed(format!("{:?}", e)))?;

        Ok(())
    }

    /// Transpose the matrix (returns a new CSR).
    pub fn transpose(&self) -> Self {
        let mut triplets = Vec::with_capacity(self.nnz);
        for row in 0..self.rows {
            let start = self.row_ptr[row] as usize;
            let end = self.row_ptr[row + 1] as usize;
            for k in start..end {
                triplets.push((self.col_idx[k], row as u32, self.values[k]));
            }
        }
        Self::from_triplets(self.cols, self.rows, &triplets)
    }

    /// Number of non-zero entries per row (for diagnostics).
    pub fn nnz_per_row(&self) -> Vec<u32> {
        (0..self.rows)
            .map(|i| self.row_ptr[i + 1] - self.row_ptr[i])
            .collect()
    }
}
