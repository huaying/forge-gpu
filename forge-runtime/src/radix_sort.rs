//! GPU Radix Sort — LSB radix sort with 4-bit passes.
//!
//! Sorts (key, value) pairs using a least-significant-bit radix sort
//! with 16 radix buckets per pass (4 bits at a time).
//!
//! This module provides:
//! - `RadixSorter` — pre-allocated sorter for GPU radix sort
//! - `radix_sort_pairs` — sort (u32, u32) key-value pairs on GPU
//!
//! All temp buffers are pre-allocated for graph-capture compatibility.

#[cfg(feature = "cuda")]
use crate::ForgeError;

#[cfg(feature = "cuda")]
use std::sync::OnceLock;

#[cfg(feature = "cuda")]
use crate::cuda;

/// Pre-allocated GPU radix sorter.
///
/// All temporary buffers are allocated once at construction time,
/// making this safe for CUDA graph capture (no dynamic allocations).
#[cfg(feature = "cuda")]
pub struct RadixSorter {
    /// Maximum number of elements this sorter can handle.
    max_n: usize,
    /// Temporary keys buffer (double-buffered).
    keys_alt: crate::Array<u32>,
    /// Temporary values buffer (double-buffered).
    vals_alt: crate::Array<u32>,
    /// Per-block histograms: [num_blocks * 16]
    histograms: crate::Array<u32>,
    /// Prefix-summed histograms (global offsets).
    global_offsets: crate::Array<u32>,
    /// Number of thread blocks used.
    num_blocks: usize,
    /// Device ordinal.
    ordinal: usize,
}

#[cfg(feature = "cuda")]
static HISTOGRAM_KERNEL: OnceLock<crate::CompiledKernel> = OnceLock::new();
#[cfg(feature = "cuda")]
static PREFIX_SUM_KERNEL: OnceLock<crate::CompiledKernel> = OnceLock::new();
#[cfg(feature = "cuda")]
static SCATTER_KERNEL: OnceLock<crate::CompiledKernel> = OnceLock::new();
#[cfg(feature = "cuda")]
static MEMSET_KERNEL: OnceLock<crate::CompiledKernel> = OnceLock::new();

#[cfg(feature = "cuda")]
impl RadixSorter {
    /// Create a new radix sorter for up to `max_n` elements.
    pub fn new(max_n: usize, ordinal: usize) -> Self {
        let device = crate::Device::Cuda(ordinal);
        let block_size = 256;
        let num_blocks = (max_n + block_size - 1) / block_size;

        Self {
            max_n,
            keys_alt: crate::Array::<u32>::zeros(max_n, device),
            vals_alt: crate::Array::<u32>::zeros(max_n, device),
            // 16 buckets per block
            histograms: crate::Array::<u32>::zeros(num_blocks * 16, device),
            global_offsets: crate::Array::<u32>::zeros(16, device),
            num_blocks,
            ordinal,
        }
    }

    /// Sort (keys, values) pairs in-place by key.
    ///
    /// After sorting, `keys` is sorted in ascending order and `values`
    /// contains the corresponding values (e.g., particle indices).
    ///
    /// Only sorts the lowest `num_bits` bits (must be a multiple of 4).
    /// For hash grid cell IDs, 20-24 bits is typical.
    pub fn sort(
        &mut self,
        keys: &mut crate::Array<u32>,
        values: &mut crate::Array<u32>,
        n: usize,
        num_bits: u32,
    ) -> Result<(), ForgeError> {
        assert!(n <= self.max_n, "n={} exceeds max_n={}", n, self.max_n);
        assert!(num_bits % 4 == 0 && num_bits <= 32, "num_bits must be multiple of 4, <= 32");

        let stream = cuda::default_stream(self.ordinal);

        // Compile kernels
        let histogram_k = HISTOGRAM_KERNEL.get_or_init(|| {
            crate::CompiledKernel::compile(RADIX_HISTOGRAM_CUDA, "radix_histogram")
                .expect("compile radix_histogram")
        });
        let prefix_k = PREFIX_SUM_KERNEL.get_or_init(|| {
            crate::CompiledKernel::compile(RADIX_PREFIX_SUM_CUDA, "radix_prefix_sum")
                .expect("compile radix_prefix_sum")
        });
        let scatter_k = SCATTER_KERNEL.get_or_init(|| {
            crate::CompiledKernel::compile(RADIX_SCATTER_CUDA, "radix_scatter")
                .expect("compile radix_scatter")
        });
        let memset_k = MEMSET_KERNEL.get_or_init(|| {
            crate::CompiledKernel::compile(
                r#"extern "C" __global__ void memset_u32(unsigned int* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = 0;
}"#,
                "memset_u32",
            ).expect("compile memset_u32")
        });

        let n_i32 = n as i32;
        let block_size = 256u32;
        let grid_size = ((n as u32) + block_size - 1) / block_size;
        let config_n = cuda::LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let hist_total = (self.num_blocks * 16) as i32;
        let hist_grid = ((hist_total as u32) + block_size - 1) / block_size;
        let config_hist_clear = cuda::LaunchConfig {
            grid_dim: (hist_grid, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let num_passes = num_bits / 4;

        for pass in 0..num_passes {
            let bit_offset = pass * 4;

            // Step 1: Clear histograms
            unsafe {
                use cuda::PushKernelArg;
                let func = memset_k.get_function(self.ordinal)?;
                let mut b = stream.launch_builder(&func);
                b.arg(self.histograms.cuda_slice_mut().unwrap());
                b.arg(&hist_total);
                b.launch(config_hist_clear)
                    .map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
            }

            // Clear global offsets
            unsafe {
                use cuda::PushKernelArg;
                let func = memset_k.get_function(self.ordinal)?;
                let sixteen = 16i32;
                let mut b = stream.launch_builder(&func);
                b.arg(self.global_offsets.cuda_slice_mut().unwrap());
                b.arg(&sixteen);
                b.launch(cuda::LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (16, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
            }

            // Step 2: Build per-block histograms
            unsafe {
                use cuda::PushKernelArg;
                let func = histogram_k.get_function(self.ordinal)?;
                let bit_i32 = bit_offset as i32;
                let mut b = stream.launch_builder(&func);
                b.arg(keys.cuda_slice().unwrap());
                b.arg(self.histograms.cuda_slice_mut().unwrap());
                b.arg(&bit_i32);
                b.arg(&n_i32);
                b.launch(config_n)
                    .map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
            }

            // Step 3: Prefix sum across histograms to get global offsets
            unsafe {
                use cuda::PushKernelArg;
                let func = prefix_k.get_function(self.ordinal)?;
                let nb_i32 = self.num_blocks as i32;
                let mut b = stream.launch_builder(&func);
                b.arg(self.histograms.cuda_slice().unwrap());
                b.arg(self.global_offsets.cuda_slice_mut().unwrap());
                b.arg(&nb_i32);
                // Single block with 16 threads (one per bucket)
                b.launch(cuda::LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (16, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
            }

            // Step 4: Scatter into alternate buffers
            unsafe {
                use cuda::PushKernelArg;
                let func = scatter_k.get_function(self.ordinal)?;
                let bit_i32 = bit_offset as i32;
                let mut b = stream.launch_builder(&func);
                b.arg(keys.cuda_slice().unwrap());
                b.arg(values.cuda_slice().unwrap());
                b.arg(self.keys_alt.cuda_slice_mut().unwrap());
                b.arg(self.vals_alt.cuda_slice_mut().unwrap());
                b.arg(self.histograms.cuda_slice().unwrap());
                b.arg(self.global_offsets.cuda_slice().unwrap());
                b.arg(&bit_i32);
                b.arg(&n_i32);
                b.launch(config_n)
                    .map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
            }

            // Swap buffers: copy alt back to primary
            // (We swap the GPU buffers by copy to avoid pointer aliasing issues)
            std::mem::swap(keys, &mut self.keys_alt);
            std::mem::swap(values, &mut self.vals_alt);
        }

        Ok(())
    }
}

/// After radix sort, find cell boundaries by linear scan.
///
/// Given sorted `cell_ids[0..n]`, writes to `cell_start` and `cell_end`
/// arrays where `cell_start[c]` is the index of the first particle in cell c,
/// and `cell_end[c]` is one past the last particle in cell c.
#[cfg(feature = "cuda")]
pub fn find_cell_boundaries(
    sorted_cell_ids: &crate::Array<u32>,
    cell_start: &mut crate::Array<u32>,
    cell_end: &mut crate::Array<u32>,
    n: usize,
    num_cells: usize,
    ordinal: usize,
) -> Result<(), ForgeError> {
    static BOUNDARY_KERNEL: OnceLock<crate::CompiledKernel> = OnceLock::new();
    static MEMSET_K: OnceLock<crate::CompiledKernel> = OnceLock::new();

    let stream = cuda::default_stream(ordinal);

    let memset_k = MEMSET_K.get_or_init(|| {
        crate::CompiledKernel::compile(
            r#"extern "C" __global__ void memset_u32_v2(unsigned int* data, unsigned int val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = val;
}"#,
            "memset_u32_v2",
        ).expect("compile memset_u32_v2")
    });

    let boundary_k = BOUNDARY_KERNEL.get_or_init(|| {
        crate::CompiledKernel::compile(FIND_BOUNDARIES_CUDA, "find_cell_boundaries")
            .expect("compile find_cell_boundaries")
    });

    let block_size = 256u32;

    // Initialize cell_start to 0xFFFFFFFF (sentinel for "no particles")
    {
        let nc = num_cells as i32;
        let grid = ((num_cells as u32) + block_size - 1) / block_size;
        unsafe {
            use cuda::PushKernelArg;
            let func = memset_k.get_function(ordinal)?;
            let sentinel = 0xFFFFFFFFu32;
            let mut b = stream.launch_builder(&func);
            b.arg(cell_start.cuda_slice_mut().unwrap());
            b.arg(&sentinel);
            b.arg(&nc);
            b.launch(cuda::LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
        }
        unsafe {
            use cuda::PushKernelArg;
            let func = memset_k.get_function(ordinal)?;
            let zero = 0u32;
            let mut b = stream.launch_builder(&func);
            b.arg(cell_end.cuda_slice_mut().unwrap());
            b.arg(&zero);
            b.arg(&nc);
            b.launch(cuda::LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
        }
    }

    // Find boundaries
    {
        let n_i32 = n as i32;
        let grid = ((n as u32) + block_size - 1) / block_size;
        unsafe {
            use cuda::PushKernelArg;
            let func = boundary_k.get_function(ordinal)?;
            let mut b = stream.launch_builder(&func);
            b.arg(sorted_cell_ids.cuda_slice().unwrap());
            b.arg(cell_start.cuda_slice_mut().unwrap());
            b.arg(cell_end.cuda_slice_mut().unwrap());
            b.arg(&n_i32);
            b.launch(cuda::LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
        }
    }

    Ok(())
}

// ── CUDA kernel sources ──

/// Histogram kernel: each block counts occurrences of each 4-bit digit.
#[cfg(feature = "cuda")]
const RADIX_HISTOGRAM_CUDA: &str = r#"
extern "C" __global__ void radix_histogram(
    const unsigned int* keys,
    unsigned int* histograms,  // [num_blocks * 16]
    int bit_offset,
    int n
) {
    __shared__ unsigned int local_hist[16];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Initialize shared histogram
    if (tid < 16) local_hist[tid] = 0;
    __syncthreads();

    // Count digits in this block
    if (gid < n) {
        unsigned int digit = (keys[gid] >> bit_offset) & 0xF;
        atomicAdd(&local_hist[digit], 1);
    }
    __syncthreads();

    // Write to global histogram
    if (tid < 16) {
        histograms[blockIdx.x * 16 + tid] = local_hist[tid];
    }
}
"#;

/// Prefix sum kernel: compute global offsets from per-block histograms.
/// Each of the 16 threads handles one bucket, summing across all blocks.
#[cfg(feature = "cuda")]
const RADIX_PREFIX_SUM_CUDA: &str = r#"
extern "C" __global__ void radix_prefix_sum(
    const unsigned int* histograms,  // [num_blocks * 16]
    unsigned int* global_offsets,     // [16] — exclusive prefix sum
    int num_blocks
) {
    int bucket = threadIdx.x;  // 0..15
    if (bucket >= 16) return;

    // Sum this bucket across all blocks
    unsigned int total = 0;
    for (int b = 0; b < num_blocks; b++) {
        total += histograms[b * 16 + bucket];
    }

    // Now we need an exclusive prefix sum across the 16 buckets
    __shared__ unsigned int sums[16];
    sums[bucket] = total;
    __syncthreads();

    // Simple serial prefix sum (only 16 elements)
    if (bucket == 0) {
        unsigned int running = 0;
        for (int i = 0; i < 16; i++) {
            unsigned int val = sums[i];
            sums[i] = running;
            running += val;
        }
    }
    __syncthreads();

    global_offsets[bucket] = sums[bucket];
}
"#;

/// Scatter kernel: move elements to sorted positions.
/// Uses per-block prefix sums + global offsets to compute destination indices.
#[cfg(feature = "cuda")]
const RADIX_SCATTER_CUDA: &str = r#"
extern "C" __global__ void radix_scatter(
    const unsigned int* keys_in,
    const unsigned int* vals_in,
    unsigned int* keys_out,
    unsigned int* vals_out,
    const unsigned int* histograms,    // [num_blocks * 16]
    const unsigned int* global_offsets, // [16]
    int bit_offset,
    int n
) {
    __shared__ unsigned int local_prefix[16];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Compute block-local exclusive prefix sum for this block's histogram
    if (tid < 16) {
        // Sum histograms of all previous blocks for this digit
        unsigned int prev_sum = 0;
        for (int b = 0; b < (int)blockIdx.x; b++) {
            prev_sum += histograms[b * 16 + tid];
        }
        local_prefix[tid] = global_offsets[tid] + prev_sum;
    }
    __syncthreads();

    if (gid < n) {
        unsigned int key = keys_in[gid];
        unsigned int digit = (key >> bit_offset) & 0xF;

        // Compute position within this digit's range for this block
        // Count how many elements before me in this block have the same digit
        unsigned int pos = 0;
        int block_start = blockIdx.x * blockDim.x;
        for (int j = block_start; j < gid; j++) {
            if (j < n && ((keys_in[j] >> bit_offset) & 0xF) == digit) {
                pos++;
            }
        }

        unsigned int dst = local_prefix[digit] + pos;
        keys_out[dst] = key;
        vals_out[dst] = vals_in[gid];
    }
}
"#;

/// Find cell boundaries after sorting by cell ID.
#[cfg(feature = "cuda")]
const FIND_BOUNDARIES_CUDA: &str = r#"
extern "C" __global__ void find_cell_boundaries(
    const unsigned int* sorted_cell_ids,
    unsigned int* cell_start,
    unsigned int* cell_end,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    unsigned int cell = sorted_cell_ids[i];

    // First element or different from previous → start of cell
    if (i == 0 || sorted_cell_ids[i - 1] != cell) {
        cell_start[cell] = (unsigned int)i;
    }

    // Last element or different from next → end of cell
    if (i == n - 1 || sorted_cell_ids[i + 1] != cell) {
        cell_end[cell] = (unsigned int)(i + 1);
    }
}
"#;
