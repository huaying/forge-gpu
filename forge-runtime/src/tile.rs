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

/// CUDA source for Tensor Core tile_matmul_tc using inline PTX.
///
/// Uses `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` instruction
/// for SM 8.0+ (Ampere and above). This includes L40 (SM 8.9).
///
/// The function operates on tiles in shared memory:
///   C[M,N] += A[M,K] * B[K,N]
/// Where A and B are f32 in shared memory, converted to f16 on the fly.
///
/// Each warp computes a 16x8 output tile using Tensor Cores.
pub const TILE_MATMUL_TC_CUDA: &str = r#"
// ── Tensor Core matrix multiply via inline PTX ──
// Requires SM 8.0+ (Ampere)
// Uses mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32

// Helper: convert two f32 values to packed __half2
__device__ unsigned int __float2_to_half2(float a, float b) {
    unsigned short ha, hb;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(ha) : "f"(a));
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hb) : "f"(b));
    return ((unsigned int)hb << 16) | (unsigned int)ha;
}

__device__ void tile_matmul_tc(
    float* C, const float* A, const float* B,
    int M, int N, int K
) {
    // Each warp handles a 16x8 output tile
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = blockDim.x / 32;

    // Number of 16x8 tiles to cover M x N
    int tiles_m = (M + 15) / 16;
    int tiles_n = (N + 7) / 8;
    int total_tiles = tiles_m * tiles_n;

    for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += num_warps) {
        int tile_row = (tile_idx / tiles_n) * 16;  // Starting row in C
        int tile_col = (tile_idx % tiles_n) * 8;   // Starting col in C

        // Accumulator registers (4 floats for 16x8 output per thread)
        float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

        // Load existing C values
        // MMA m16n8k16: each thread contributes to specific output elements
        // Thread mapping for m16n8: thread holds 2x2 output elements
        int c_row_base = lane_id / 4;          // 0..7 (maps to rows 0..7 or 8..15)
        int c_col_base = (lane_id % 4) * 2;   // 0,2,4,6

        int cr0 = tile_row + c_row_base;
        int cr1 = tile_row + c_row_base + 8;
        int cc0 = tile_col + c_col_base;
        int cc1 = tile_col + c_col_base + 1;

        if (cr0 < M && cc0 < N) c0 = C[cr0 * N + cc0];
        if (cr0 < M && cc1 < N) c1 = C[cr0 * N + cc1];
        if (cr1 < M && cc0 < N) c2 = C[cr1 * N + cc0];
        if (cr1 < M && cc1 < N) c3 = C[cr1 * N + cc1];

        // Process K in chunks of 16
        for (int k_base = 0; k_base < K; k_base += 16) {
            // Load A fragment (m16k16, row major)
            // Each thread loads specific elements into 8 half values (4 packed u32)
            unsigned int a0, a1, a2, a3;
            {
                // A fragment layout for m16k16 row-major:
                // Thread t loads A[row][k] where mapping depends on lane
                int a_row0 = tile_row + (lane_id / 4);          // rows 0..7
                int a_row1 = tile_row + (lane_id / 4) + 8;      // rows 8..15
                int a_k0 = k_base + (lane_id % 4) * 2;
                int a_k1 = a_k0 + 1;
                int a_k2 = a_k0 + 8;
                int a_k3 = a_k2 + 1;

                float av00 = (a_row0 < M && a_k0 < K) ? A[a_row0 * K + a_k0] : 0.0f;
                float av01 = (a_row0 < M && a_k1 < K) ? A[a_row0 * K + a_k1] : 0.0f;
                float av10 = (a_row0 < M && a_k2 < K) ? A[a_row0 * K + a_k2] : 0.0f;
                float av11 = (a_row0 < M && a_k3 < K) ? A[a_row0 * K + a_k3] : 0.0f;
                float av20 = (a_row1 < M && a_k0 < K) ? A[a_row1 * K + a_k0] : 0.0f;
                float av21 = (a_row1 < M && a_k1 < K) ? A[a_row1 * K + a_k1] : 0.0f;
                float av30 = (a_row1 < M && a_k2 < K) ? A[a_row1 * K + a_k2] : 0.0f;
                float av31 = (a_row1 < M && a_k3 < K) ? A[a_row1 * K + a_k3] : 0.0f;

                a0 = __float2_to_half2(av00, av01);
                a1 = __float2_to_half2(av10, av11);
                a2 = __float2_to_half2(av20, av21);
                a3 = __float2_to_half2(av30, av31);
            }

            // Load B fragment (k16n8, col major)
            unsigned int b0, b1;
            {
                // B fragment for k16n8 col-major:
                int b_k0 = k_base + (lane_id / 4);
                int b_k1 = b_k0 + 8;
                int b_col = tile_col + (lane_id % 4) * 2;
                int b_col1 = b_col + 1;

                float bv00 = (b_k0 < K && b_col < N) ? B[b_k0 * N + b_col] : 0.0f;
                float bv01 = (b_k0 < K && b_col1 < N) ? B[b_k0 * N + b_col1] : 0.0f;
                float bv10 = (b_k1 < K && b_col < N) ? B[b_k1 * N + b_col] : 0.0f;
                float bv11 = (b_k1 < K && b_col1 < N) ? B[b_k1 * N + b_col1] : 0.0f;

                b0 = __float2_to_half2(bv00, bv01);
                b1 = __float2_to_half2(bv10, bv11);
            }

            // Execute Tensor Core MMA
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%10, %11, %12, %13};"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "f"(c0), "f"(c1), "f"(c2), "f"(c3)
            );
        }

        // Store C results back
        if (cr0 < M && cc0 < N) C[cr0 * N + cc0] = c0;
        if (cr0 < M && cc1 < N) C[cr0 * N + cc1] = c1;
        if (cr1 < M && cc0 < N) C[cr1 * N + cc0] = c2;
        if (cr1 < M && cc1 < N) C[cr1 * N + cc1] = c3;
    }
    __syncthreads();
}
"#;
