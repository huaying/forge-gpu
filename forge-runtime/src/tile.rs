//! Tile primitives for cooperative block-level operations.
//!
//! Tiles are block-sized chunks of data that live in shared or register memory.
//! Operations on tiles are cooperative — all threads in a block participate.
//!
//! This module provides the runtime support for tile operations.
//! The codegen side (`forge-macros`) translates Rust tile operations into
//! CUDA shared memory + cooperative operations.
//!
//! ## Tile Operations
//! - `tile_load` — load from global to shared/register
//! - `tile_store` — store from shared/register to global
//! - `tile_zeros` — zero-fill a tile
//! - `tile_sum` — cooperative reduction (sum)
//! - `tile_max` / `tile_min` — cooperative reduction
//! - `tile_matmul` — cooperative matrix multiply
//! - `tile_scan` — cooperative prefix scan
//!
//! ## Memory Model
//! - **Register tiles**: each thread holds `tile_size / block_dim` elements
//! - **Shared tiles**: stored in `__shared__` memory, accessible by all threads

/// Tile configuration for a kernel launch.
#[derive(Debug, Clone, Copy)]
pub struct TileConfig {
    /// Tile dimensions (e.g., [M, N] for 2D tile).
    pub dims: [usize; 4],
    /// Number of active dimensions.
    pub ndim: u8,
    /// Number of threads per block.
    pub block_dim: u32,
}

impl TileConfig {
    pub fn new_1d(size: usize, block_dim: u32) -> Self {
        Self { dims: [size, 1, 1, 1], ndim: 1, block_dim }
    }

    pub fn new_2d(m: usize, n: usize, block_dim: u32) -> Self {
        Self { dims: [m, n, 1, 1], ndim: 2, block_dim }
    }

    /// Total elements in the tile.
    pub fn total(&self) -> usize {
        let n = self.ndim as usize;
        self.dims[..n].iter().product()
    }

    /// Elements per thread (register tile).
    pub fn elems_per_thread(&self) -> usize {
        (self.total() + self.block_dim as usize - 1) / self.block_dim as usize
    }

    /// Shared memory bytes needed for this tile (f32).
    pub fn shared_bytes_f32(&self) -> usize {
        self.total() * 4
    }
}

/// CUDA source for tile utility functions.
/// These are included in kernels that use tile operations.
pub const TILE_DEVICE_UTILS: &str = r#"
// ── Tile cooperative primitives ──

// Cooperative load: each thread loads its portion of the tile
__device__ void tile_load_f32(float* shared, const float* global, int tile_size, int n) {
    for (int i = threadIdx.x; i < tile_size && (blockIdx.x * tile_size + i) < n; i += blockDim.x) {
        shared[i] = global[blockIdx.x * tile_size + i];
    }
    __syncthreads();
}

// Cooperative store
__device__ void tile_store_f32(float* global, const float* shared, int tile_size, int n) {
    for (int i = threadIdx.x; i < tile_size && (blockIdx.x * tile_size + i) < n; i += blockDim.x) {
        global[blockIdx.x * tile_size + i] = shared[i];
    }
    __syncthreads();
}

// Cooperative sum reduction using shared memory
__device__ float tile_sum_f32(float* shared, int tile_size) {
    // Phase 1: each thread sums its elements
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
        local_sum += shared[i];
    }

    // Phase 2: warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }

    // Phase 3: write per-warp results to shared memory
    __shared__ float warp_sums[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = local_sum;
    __syncthreads();

    // Phase 4: first warp reduces the warp sums
    if (warp_id == 0) {
        local_sum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? warp_sums[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }
    }
    return local_sum; // Only thread 0 has the correct result
}

// Cooperative max reduction
__device__ float tile_max_f32(float* shared, int tile_size) {
    float local_max = -1e38f;
    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
        local_max = fmaxf(local_max, shared[i]);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
    }
    __shared__ float warp_maxs[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_maxs[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        local_max = (lane < (blockDim.x + warpSize - 1) / warpSize) ? warp_maxs[lane] : -1e38f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
        }
    }
    return local_max;
}

// Cooperative min reduction
__device__ float tile_min_f32(float* shared, int tile_size) {
    float local_min = 1e38f;
    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
        local_min = fminf(local_min, shared[i]);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_min = fminf(local_min, __shfl_down_sync(0xFFFFFFFF, local_min, offset));
    }
    __shared__ float warp_mins[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_mins[warp_id] = local_min;
    __syncthreads();
    if (warp_id == 0) {
        local_min = (lane < (blockDim.x + warpSize - 1) / warpSize) ? warp_mins[lane] : 1e38f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            local_min = fminf(local_min, __shfl_down_sync(0xFFFFFFFF, local_min, offset));
        }
    }
    return local_min;
}

// Cooperative inclusive prefix scan (Blelloch within block)
__device__ void tile_scan_inclusive_f32(float* shared, int tile_size) {
    // Simple serial scan per thread, then fixup across warps
    // For tile_size <= blockDim, each thread handles one element
    int tid = threadIdx.x;
    if (tid >= tile_size) return;

    // Up-sweep (reduce)
    for (int stride = 1; stride < tile_size; stride <<= 1) {
        __syncthreads();
        float val = shared[tid];
        if (tid >= stride) val += shared[tid - stride];
        __syncthreads();
        shared[tid] = val;
    }
}

// Cooperative matrix multiply: C[M,N] += A[M,K] * B[K,N]
// All matrices in shared memory
__device__ void tile_matmul_f32(
    float* C, const float* A, const float* B,
    int M, int N, int K
) {
    // Each thread computes one or more elements of C
    int total = M * N;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int row = idx / N;
        int col = idx % N;
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] += sum;
    }
    __syncthreads();
}
"#;
