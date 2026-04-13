use crate::modules::{FieldSet, SimModule};
use forge_runtime::ForgeError;
use std::sync::OnceLock;

/// Fused SPH module: density (pass 1) + pressure+viscosity (pass 2).
///
/// Instead of 3 separate modules each doing a full neighbor traversal,
/// this does 2 passes:
///   Pass 1: compute density (same as SphDensityModule, including grid rebuild)
///   Pass 2: compute pressure + viscosity forces in a single neighbor traversal
///
/// This is an automatic optimization — users still write 3 separate forces in TOML,
/// but build_pipeline() detects the pattern and substitutes this fused module.
pub struct SphFusedModule {
    pub smoothing_radius: f32,
    pub cell_size: f32,
    pub grid_dims: [u32; 3],
    pub particle_mass: f32,
    pub gas_constant: f32,
    pub rest_density: f32,
    pub viscosity_coefficient: f32,
}

// Re-use the hash grid build kernels from sph_density
// (they use OnceLock so they're compiled once globally)
static CELL_INDEX_KERNEL: OnceLock<forge_runtime::CompiledKernel> = OnceLock::new();
static COUNT_KERNEL: OnceLock<forge_runtime::CompiledKernel> = OnceLock::new();
static PREFIX_SUM_KERNEL: OnceLock<forge_runtime::CompiledKernel> = OnceLock::new();
static PREFIX_SUM_FIXUP_KERNEL: OnceLock<forge_runtime::CompiledKernel> = OnceLock::new();
static SCATTER_KERNEL: OnceLock<forge_runtime::CompiledKernel> = OnceLock::new();
static DENSITY_KERNEL: OnceLock<forge_runtime::CompiledKernel> = OnceLock::new();
static FUSED_FORCE_KERNEL: OnceLock<forge_runtime::CompiledKernel> = OnceLock::new();

impl SphFusedModule {
    fn rebuild_hashgrid_gpu(&self, fields: &mut FieldSet) -> Result<(), ForgeError> {
        let n = fields.particle_count;
        let (nx, ny, nz) = (self.grid_dims[0], self.grid_dims[1], self.grid_dims[2]);
        let num_cells = (nx * ny * nz) as usize;
        let cs = self.cell_size;
        let device = fields.device;

        // Ensure GPU arrays exist
        if !fields.u32_fields.contains_key("cell_idx") {
            fields.u32_fields.insert("cell_idx".to_string(),
                forge_runtime::Array::<u32>::zeros(n, device));
        }
        if !fields.u32_fields.contains_key("cell_count") {
            fields.u32_fields.insert("cell_count".to_string(),
                forge_runtime::Array::<u32>::zeros(num_cells, device));
        }
        if !fields.u32_fields.contains_key("cell_start") {
            fields.u32_fields.insert("cell_start".to_string(),
                forge_runtime::Array::<u32>::zeros(num_cells + 1, device));
        }
        if !fields.u32_fields.contains_key("sorted_indices") {
            fields.u32_fields.insert("sorted_indices".to_string(),
                forge_runtime::Array::<u32>::zeros(n, device));
        }
        if !fields.u32_fields.contains_key("cell_offset") {
            fields.u32_fields.insert("cell_offset".to_string(),
                forge_runtime::Array::<u32>::zeros(num_cells, device));
        }
        // Pre-allocate prefix sum temp buffers
        let block_elems = 1024usize;
        let num_blocks_psum = (num_cells + block_elems - 1) / block_elems;
        if !fields.u32_fields.contains_key("_psum_block_sums") {
            fields.u32_fields.insert("_psum_block_sums".to_string(),
                forge_runtime::Array::<u32>::zeros(num_blocks_psum.max(1), device));
            fields.u32_fields.insert("_psum_block_sums_scanned".to_string(),
                forge_runtime::Array::<u32>::zeros(num_blocks_psum.max(1), device));
            fields.u32_fields.insert("_psum_dummy".to_string(),
                forge_runtime::Array::<u32>::zeros(1, device));
        }

        let stream = forge_runtime::cuda::default_stream(0);
        let n_i32 = n as i32;
        let nx_i32 = nx as i32;
        let ny_i32 = ny as i32;
        let nz_i32 = nz as i32;

        let cell_idx_kernel = CELL_INDEX_KERNEL.get_or_init(|| {
            forge_runtime::CompiledKernel::compile(
                r#"extern "C" __global__ void compute_cell_index(
    const float* px, const float* py, const float* pz,
    unsigned int* cell_idx,
    float cell_size, int grid_nx, int grid_ny, int grid_nz, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int cx = (int)floorf(px[i] / cell_size);
    int cy = (int)floorf(py[i] / cell_size);
    int cz = (int)floorf(pz[i] / cell_size);
    if (cx < 0) cx = 0; if (cx >= grid_nx) cx = grid_nx - 1;
    if (cy < 0) cy = 0; if (cy >= grid_ny) cy = grid_ny - 1;
    if (cz < 0) cz = 0; if (cz >= grid_nz) cz = grid_nz - 1;
    cell_idx[i] = (unsigned int)cx
        + (unsigned int)cy * (unsigned int)grid_nx
        + (unsigned int)cz * (unsigned int)grid_nx * (unsigned int)grid_ny;
}"#,
                "compute_cell_index",
            ).expect("compile compute_cell_index")
        });

        let count_kernel = COUNT_KERNEL.get_or_init(|| {
            forge_runtime::CompiledKernel::compile(
                r#"extern "C" __global__ void count_cells(
    const unsigned int* cell_idx, unsigned int* cell_count, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    atomicAdd(&cell_count[cell_idx[i]], 1u);
}"#,
                "count_cells",
            ).expect("compile count_cells")
        });

        let scatter_kernel = SCATTER_KERNEL.get_or_init(|| {
            forge_runtime::CompiledKernel::compile(
                r#"extern "C" __global__ void scatter_particles(
    const unsigned int* cell_idx,
    const unsigned int* cell_start,
    unsigned int* cell_offset,
    unsigned int* sorted_indices,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int cell = cell_idx[i];
    unsigned int offset = atomicAdd(&cell_offset[cell], 1u);
    sorted_indices[cell_start[cell] + offset] = (unsigned int)i;
}"#,
                "scatter_particles",
            ).expect("compile scatter_particles")
        });

        let prefix_sum_kernel = PREFIX_SUM_KERNEL.get_or_init(|| {
            forge_runtime::CompiledKernel::compile(
                r#"
#define BLOCK_SIZE 512
extern "C" __global__ void prefix_sum(
    const unsigned int* input,
    unsigned int* output,
    unsigned int* block_sums,
    int n
) {
    __shared__ unsigned int temp[BLOCK_SIZE * 2];
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * BLOCK_SIZE * 2;
    int ai = tid;
    int bi = tid + BLOCK_SIZE;
    temp[ai] = (block_offset + ai < n) ? input[block_offset + ai] : 0;
    temp[bi] = (block_offset + bi < n) ? input[block_offset + bi] : 0;
    int offset = 1;
    for (int d = BLOCK_SIZE; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai2 = offset * (2 * tid + 1) - 1;
            int bi2 = offset * (2 * tid + 2) - 1;
            temp[bi2] += temp[ai2];
        }
        offset *= 2;
    }
    __syncthreads();
    if (tid == 0) {
        if (block_sums != NULL) block_sums[blockIdx.x] = temp[BLOCK_SIZE * 2 - 1];
        temp[BLOCK_SIZE * 2 - 1] = 0;
    }
    for (int d = 1; d < BLOCK_SIZE * 2; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai2 = offset * (2 * tid + 1) - 1;
            int bi2 = offset * (2 * tid + 2) - 1;
            unsigned int t = temp[ai2];
            temp[ai2] = temp[bi2];
            temp[bi2] += t;
        }
    }
    __syncthreads();
    if (block_offset + ai < n) output[block_offset + ai] = temp[ai];
    if (block_offset + bi < n) output[block_offset + bi] = temp[bi];
}
"#,
                "prefix_sum",
            ).expect("compile prefix_sum")
        });

        let fixup_kernel = PREFIX_SUM_FIXUP_KERNEL.get_or_init(|| {
            forge_runtime::CompiledKernel::compile(
                r#"
#define BLOCK_SIZE 512
extern "C" __global__ void prefix_sum_fixup(
    unsigned int* data,
    const unsigned int* block_sums,
    int n
) {
    int idx = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x;
    if (blockIdx.x == 0) return;
    unsigned int add_val = block_sums[blockIdx.x];
    if (idx < n) data[idx] += add_val;
    idx += BLOCK_SIZE;
    if (idx < n) data[idx] += add_val;
}
"#,
                "prefix_sum_fixup",
            ).expect("compile prefix_sum_fixup")
        });

        let config_n = forge_runtime::cuda::LaunchConfig::for_num_elems(n as u32);

        let px = fields.f32_fields.get("pos_x").unwrap() as *const forge_runtime::Array<f32>;
        let py = fields.f32_fields.get("pos_y").unwrap() as *const forge_runtime::Array<f32>;
        let pz = fields.f32_fields.get("pos_z").unwrap() as *const forge_runtime::Array<f32>;
        let cell_idx = fields.u32_fields.get_mut("cell_idx").unwrap() as *mut forge_runtime::Array<u32>;
        let cell_count = fields.u32_fields.get_mut("cell_count").unwrap() as *mut forge_runtime::Array<u32>;
        let cell_start = fields.u32_fields.get_mut("cell_start").unwrap() as *mut forge_runtime::Array<u32>;
        let sorted_idx = fields.u32_fields.get_mut("sorted_indices").unwrap() as *mut forge_runtime::Array<u32>;
        let cell_offset = fields.u32_fields.get_mut("cell_offset").unwrap() as *mut forge_runtime::Array<u32>;

        // Zero out cell_count and cell_offset using GPU memset (no allocation)
        unsafe {
            stream.memset_zeros((*cell_count).cuda_slice_mut().unwrap())
                .map_err(|e| ForgeError::LaunchFailed(format!("memset cell_count: {:?}", e)))?;
            stream.memset_zeros((*cell_offset).cuda_slice_mut().unwrap())
                .map_err(|e| ForgeError::LaunchFailed(format!("memset cell_offset: {:?}", e)))?;
        }

        // Step 1: compute cell indices
        unsafe {
            use forge_runtime::cuda::PushKernelArg;
            let func = cell_idx_kernel.get_function(0)?;
            let mut b = stream.launch_builder(&func);
            b.arg((*px).cuda_slice().unwrap());
            b.arg((*py).cuda_slice().unwrap());
            b.arg((*pz).cuda_slice().unwrap());
            b.arg((*cell_idx).cuda_slice_mut().unwrap());
            b.arg(&cs);
            b.arg(&nx_i32);
            b.arg(&ny_i32);
            b.arg(&nz_i32);
            b.arg(&n_i32);
            b.launch(config_n).map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
        }

        // Step 2: count particles per cell
        unsafe {
            use forge_runtime::cuda::PushKernelArg;
            let func = count_kernel.get_function(0)?;
            let mut b = stream.launch_builder(&func);
            b.arg((*cell_idx).cuda_slice().unwrap());
            b.arg((*cell_count).cuda_slice_mut().unwrap());
            b.arg(&n_i32);
            b.launch(config_n).map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
        }

        // Step 3: GPU prefix sum (using pre-allocated temp buffers)
        {
            let block_elems = 1024;
            let num_blocks = (num_cells + block_elems - 1) / block_elems;

            let block_sums = fields.u32_fields.get_mut("_psum_block_sums").unwrap() as *mut forge_runtime::Array<u32>;
            let block_sums_scanned = fields.u32_fields.get_mut("_psum_block_sums_scanned").unwrap() as *mut forge_runtime::Array<u32>;
            let dummy = fields.u32_fields.get_mut("_psum_dummy").unwrap() as *mut forge_runtime::Array<u32>;

            // Zero temp buffers
            unsafe {
                stream.memset_zeros((*block_sums).cuda_slice_mut().unwrap())
                    .map_err(|e| ForgeError::LaunchFailed(format!("memset block_sums: {:?}", e)))?;
                stream.memset_zeros((*block_sums_scanned).cuda_slice_mut().unwrap())
                    .map_err(|e| ForgeError::LaunchFailed(format!("memset block_sums_scanned: {:?}", e)))?;
            }

            unsafe {
                use forge_runtime::cuda::PushKernelArg;
                let func = prefix_sum_kernel.get_function(0)?;
                let launch_cfg = forge_runtime::cuda::LaunchConfig {
                    grid_dim: (num_blocks as u32, 1, 1),
                    block_dim: (512, 1, 1),
                    shared_mem_bytes: 0,
                };
                let nc_i32 = num_cells as i32;
                let mut b = stream.launch_builder(&func);
                b.arg((*cell_count).cuda_slice().unwrap());
                b.arg((*cell_start).cuda_slice_mut().unwrap());
                b.arg((*block_sums).cuda_slice_mut().unwrap());
                b.arg(&nc_i32);
                b.launch(launch_cfg).map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
            }

            if num_blocks > 1 {
                unsafe {
                    use forge_runtime::cuda::PushKernelArg;
                    let func = prefix_sum_kernel.get_function(0)?;
                    let launch_cfg = forge_runtime::cuda::LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (512, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let nb_i32 = num_blocks as i32;
                    let mut b = stream.launch_builder(&func);
                    b.arg((*block_sums).cuda_slice().unwrap());
                    b.arg((*block_sums_scanned).cuda_slice_mut().unwrap());
                    b.arg((*dummy).cuda_slice_mut().unwrap());
                    b.arg(&nb_i32);
                    b.launch(launch_cfg).map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
                }

                unsafe {
                    use forge_runtime::cuda::PushKernelArg;
                    let func = fixup_kernel.get_function(0)?;
                    let launch_cfg = forge_runtime::cuda::LaunchConfig {
                        grid_dim: (num_blocks as u32, 1, 1),
                        block_dim: (512, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let nc_i32 = num_cells as i32;
                    let mut b = stream.launch_builder(&func);
                    b.arg((*cell_start).cuda_slice_mut().unwrap());
                    b.arg((*block_sums_scanned).cuda_slice().unwrap());
                    b.arg(&nc_i32);
                    b.launch(launch_cfg).map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
                }
            }

            // Set cell_start[num_cells] = n
            {
                static SET_LAST_KERNEL: OnceLock<forge_runtime::CompiledKernel> = OnceLock::new();
                let set_last = SET_LAST_KERNEL.get_or_init(|| {
                    forge_runtime::CompiledKernel::compile(
                        r#"extern "C" __global__ void set_last(unsigned int* arr, int idx, unsigned int val) {
    if (threadIdx.x == 0 && blockIdx.x == 0) arr[idx] = val;
}"#,
                        "set_last",
                    ).expect("compile set_last")
                });
                unsafe {
                    use forge_runtime::cuda::PushKernelArg;
                    let func = set_last.get_function(0)?;
                    let launch_cfg = forge_runtime::cuda::LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let nc_i32 = num_cells as i32;
                    let n_u32 = n as u32;
                    let mut b = stream.launch_builder(&func);
                    b.arg((*cell_start).cuda_slice_mut().unwrap());
                    b.arg(&nc_i32);
                    b.arg(&n_u32);
                    b.launch(launch_cfg).map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
                }
            }
        }

        // Step 4: scatter particles into sorted order
        unsafe {
            use forge_runtime::cuda::PushKernelArg;
            let func = scatter_kernel.get_function(0)?;
            let mut b = stream.launch_builder(&func);
            b.arg((*cell_idx).cuda_slice().unwrap());
            b.arg((*cell_start).cuda_slice().unwrap());
            b.arg((*cell_offset).cuda_slice_mut().unwrap());
            b.arg((*sorted_idx).cuda_slice_mut().unwrap());
            b.arg(&n_i32);
            b.launch(config_n).map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
        }
        stream.synchronize().map_err(|e| ForgeError::SyncFailed(format!("{:?}", e)))?;

        Ok(())
    }
}

impl SimModule for SphFusedModule {
    fn name(&self) -> &str { "sph_fused" }

    fn execute(&self, fields: &mut FieldSet, dt: f32) -> Result<(), ForgeError> {
        let n = fields.particle_count;

        // Rebuild hash grid
        self.rebuild_hashgrid_gpu(fields)?;

        // Ensure density field exists
        if !fields.f32_fields.contains_key("density") {
            fields.add_f32_zeros("density", n);
        }

        // ── Pass 1: Compute density (with shared memory tiling) ──
        let density_kernel = DENSITY_KERNEL.get_or_init(|| {
            forge_runtime::CompiledKernel::compile(
                r#"
#define TILE_SIZE 128

extern "C" __global__ void sph_density(
    const float* px, const float* py, const float* pz,
    float* density,
    const unsigned int* cell_start, const unsigned int* sorted_idx,
    float h, float mass, float cell_size,
    int grid_nx, int grid_ny, int grid_nz, int n
) {
    // Shared memory tile for neighbor positions
    __shared__ float spx[TILE_SIZE];
    __shared__ float spy[TILE_SIZE];
    __shared__ float spz[TILE_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    float xi, yi, zi;
    bool valid = (i < n);
    if (valid) { xi = px[i]; yi = py[i]; zi = pz[i]; }

    float h2 = h * h;
    float h9 = h2 * h2 * h2 * h2 * h;
    float poly6_coeff = 315.0f / (64.0f * 3.14159265f * h9);
    float rho = 0.0f;

    // Each thread computes its own cell
    int cx = valid ? (int)floorf(xi / cell_size) : 0;
    int cy = valid ? (int)floorf(yi / cell_size) : 0;
    int cz = valid ? (int)floorf(zi / cell_size) : 0;
    if (cx < 0) cx = 0; if (cx >= grid_nx) cx = grid_nx - 1;
    if (cy < 0) cy = 0; if (cy >= grid_ny) cy = grid_ny - 1;
    if (cz < 0) cz = 0; if (cz >= grid_nz) cz = grid_nz - 1;

    // Iterate 3x3x3 neighbor cells
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int ncx = cx + dx, ncy = cy + dy, ncz = cz + dz;
                if (ncx < 0 || ncx >= grid_nx || ncy < 0 || ncy >= grid_ny || ncz < 0 || ncz >= grid_nz) continue;

                unsigned int cell = (unsigned int)ncx + (unsigned int)ncy * (unsigned int)grid_nx + (unsigned int)ncz * (unsigned int)grid_nx * (unsigned int)grid_ny;
                unsigned int cstart = cell_start[cell];
                unsigned int cend = cell_start[cell + 1];
                unsigned int count = cend - cstart;

                // Process neighbors in tiles
                for (unsigned int tile_base = 0; tile_base < count; tile_base += TILE_SIZE) {
                    // Cooperatively load tile into shared memory
                    unsigned int load_idx = tile_base + tid;
                    if (load_idx < count) {
                        int j = (int)sorted_idx[cstart + load_idx];
                        spx[tid] = px[j];
                        spy[tid] = py[j];
                        spz[tid] = pz[j];
                    }
                    __syncthreads();

                    // Compute from shared memory
                    unsigned int tile_count = count - tile_base;
                    if (tile_count > TILE_SIZE) tile_count = TILE_SIZE;

                    if (valid) {
                        for (unsigned int t = 0; t < tile_count; t++) {
                            float djx = xi - spx[t];
                            float djy = yi - spy[t];
                            float djz = zi - spz[t];
                            float r2 = djx*djx + djy*djy + djz*djz;
                            if (r2 < h2) {
                                float diff = h2 - r2;
                                rho += mass * poly6_coeff * diff * diff * diff;
                            }
                        }
                    }
                    __syncthreads();
                }
            }
        }
    }

    if (valid) density[i] = rho;
}"#,
                "sph_density",
            ).expect("compile sph_density")
        });

        // ── Pass 2: Fused pressure + viscosity (with shared memory tiling) ──
        let fused_kernel = FUSED_FORCE_KERNEL.get_or_init(|| {
            forge_runtime::CompiledKernel::compile(
                r#"
#define TILE_SIZE 128

extern "C" __global__ void sph_fused_force(
    const float* px, const float* py, const float* pz,
    float* vx, float* vy, float* vz,
    const float* density,
    const unsigned int* cell_start, const unsigned int* sorted_idx,
    float h, float mass, float gas_constant, float rest_density, float visc_coeff,
    float cell_size, int grid_nx, int grid_ny, int grid_nz,
    float dt, int n
) {
    // Shared memory tiles: pos + vel + density of neighbors
    __shared__ float spx[TILE_SIZE];
    __shared__ float spy[TILE_SIZE];
    __shared__ float spz[TILE_SIZE];
    __shared__ float svx[TILE_SIZE];
    __shared__ float svy[TILE_SIZE];
    __shared__ float svz[TILE_SIZE];
    __shared__ float srho[TILE_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float xi, yi, zi, vxi, vyi, vzi, rho_i, p_i;
    bool valid = (i < n);
    if (valid) {
        xi = px[i]; yi = py[i]; zi = pz[i];
        vxi = vx[i]; vyi = vy[i]; vzi = vz[i];
        rho_i = density[i];
        p_i = gas_constant * (rho_i - rest_density);
    }
    if (valid && rho_i < 1e-6f) valid = false;

    float h2 = h * h;
    float h6 = h2 * h2 * h2;
    float spiky_grad_coeff = -45.0f / (3.14159265f * h6);
    float visc_lap_coeff = 45.0f / (3.14159265f * h6);

    float pax = 0.0f, pay = 0.0f, paz = 0.0f;
    float vax = 0.0f, vay = 0.0f, vaz = 0.0f;

    int cx = valid ? (int)floorf(xi / cell_size) : 0;
    int cy = valid ? (int)floorf(yi / cell_size) : 0;
    int cz = valid ? (int)floorf(zi / cell_size) : 0;
    if (cx < 0) cx = 0; if (cx >= grid_nx) cx = grid_nx - 1;
    if (cy < 0) cy = 0; if (cy >= grid_ny) cy = grid_ny - 1;
    if (cz < 0) cz = 0; if (cz >= grid_nz) cz = grid_nz - 1;

    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int ncx = cx + dx, ncy = cy + dy, ncz = cz + dz;
                if (ncx < 0 || ncx >= grid_nx || ncy < 0 || ncy >= grid_ny || ncz < 0 || ncz >= grid_nz) continue;

                unsigned int cell = (unsigned int)ncx + (unsigned int)ncy * (unsigned int)grid_nx + (unsigned int)ncz * (unsigned int)grid_nx * (unsigned int)grid_ny;
                unsigned int cstart = cell_start[cell];
                unsigned int cend = cell_start[cell + 1];
                unsigned int count = cend - cstart;

                for (unsigned int tile_base = 0; tile_base < count; tile_base += TILE_SIZE) {
                    // Cooperatively load tile
                    unsigned int load_idx = tile_base + tid;
                    if (load_idx < count) {
                        int j = (int)sorted_idx[cstart + load_idx];
                        spx[tid] = px[j];
                        spy[tid] = py[j];
                        spz[tid] = pz[j];
                        svx[tid] = vx[j];
                        svy[tid] = vy[j];
                        svz[tid] = vz[j];
                        srho[tid] = density[j];
                    }
                    __syncthreads();

                    unsigned int tile_count = count - tile_base;
                    if (tile_count > TILE_SIZE) tile_count = TILE_SIZE;

                    if (valid) {
                        for (unsigned int t = 0; t < tile_count; t++) {
                            // Skip self (compare position — sorted index not available in tile)
                            float djx = xi - spx[t];
                            float djy = yi - spy[t];
                            float djz = zi - spz[t];
                            float r2 = djx*djx + djy*djy + djz*djz;

                            if (r2 < h2 && r2 > 1e-12f) {
                                float r = sqrtf(r2);
                                float rho_j = srho[t];
                                if (rho_j < 1e-6f) continue;
                                float diff = h - r;

                                // Pressure
                                float p_j = gas_constant * (rho_j - rest_density);
                                float grad_scale = spiky_grad_coeff * diff * diff / r;
                                float f_scale = -mass * (p_i + p_j) / (2.0f * rho_j) * grad_scale;
                                pax += f_scale * djx;
                                pay += f_scale * djy;
                                paz += f_scale * djz;

                                // Viscosity
                                float lap = visc_lap_coeff * diff;
                                float v_scale = visc_coeff * mass / rho_j * lap;
                                vax += v_scale * (svx[t] - vxi);
                                vay += v_scale * (svy[t] - vyi);
                                vaz += v_scale * (svz[t] - vzi);
                            }
                        }
                    }
                    __syncthreads();
                }
            }
        }
    }

    if (valid) {
        float inv_rho = 1.0f / rho_i;
        vx[i] += ((pax + vax) * inv_rho) * dt;
        vy[i] += ((pay + vay) * inv_rho) * dt;
        vz[i] += ((paz + vaz) * inv_rho) * dt;
    }
}"#,
                "sph_fused_force",
            ).expect("compile sph_fused_force")
        });

        let stream = forge_runtime::cuda::default_stream(0);
        // Block size must match TILE_SIZE in kernels (128)
        let sph_block_size = 128u32;
        let sph_grid_size = (n as u32 + sph_block_size - 1) / sph_block_size;
        let config = forge_runtime::cuda::LaunchConfig {
            grid_dim: (sph_grid_size, 1, 1),
            block_dim: (sph_block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i32 = n as i32;
        let grid_nx = self.grid_dims[0] as i32;
        let grid_ny = self.grid_dims[1] as i32;
        let grid_nz = self.grid_dims[2] as i32;

        // Pass 1: density
        {
            let func = density_kernel.get_function(0)?;
            let px = fields.f32_fields.get("pos_x").unwrap() as *const forge_runtime::Array<f32>;
            let py = fields.f32_fields.get("pos_y").unwrap() as *const forge_runtime::Array<f32>;
            let pz = fields.f32_fields.get("pos_z").unwrap() as *const forge_runtime::Array<f32>;
            let density = fields.f32_fields.get_mut("density").unwrap() as *mut forge_runtime::Array<f32>;
            let cell_start = fields.u32_fields.get("cell_start").unwrap() as *const forge_runtime::Array<u32>;
            let sorted_idx = fields.u32_fields.get("sorted_indices").unwrap() as *const forge_runtime::Array<u32>;

            unsafe {
                use forge_runtime::cuda::PushKernelArg;
                let mut b = stream.launch_builder(&func);
                b.arg((*px).cuda_slice().unwrap());
                b.arg((*py).cuda_slice().unwrap());
                b.arg((*pz).cuda_slice().unwrap());
                b.arg((*density).cuda_slice_mut().unwrap());
                b.arg((*cell_start).cuda_slice().unwrap());
                b.arg((*sorted_idx).cuda_slice().unwrap());
                b.arg(&self.smoothing_radius);
                b.arg(&self.particle_mass);
                b.arg(&self.cell_size);
                b.arg(&grid_nx);
                b.arg(&grid_ny);
                b.arg(&grid_nz);
                b.arg(&n_i32);
                b.launch(config).map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
            }
        }

        // Sync before pass 2 (pressure needs density to be fully computed)
        stream.synchronize().map_err(|e| ForgeError::SyncFailed(format!("{:?}", e)))?;

        // Pass 2: fused pressure + viscosity
        {
            let func = fused_kernel.get_function(0)?;
            let px = fields.f32_fields.get("pos_x").unwrap() as *const forge_runtime::Array<f32>;
            let py = fields.f32_fields.get("pos_y").unwrap() as *const forge_runtime::Array<f32>;
            let pz = fields.f32_fields.get("pos_z").unwrap() as *const forge_runtime::Array<f32>;
            let vx = fields.f32_fields.get_mut("vel_x").unwrap() as *mut forge_runtime::Array<f32>;
            let vy = fields.f32_fields.get_mut("vel_y").unwrap() as *mut forge_runtime::Array<f32>;
            let vz = fields.f32_fields.get_mut("vel_z").unwrap() as *mut forge_runtime::Array<f32>;
            let density = fields.f32_fields.get("density").unwrap() as *const forge_runtime::Array<f32>;
            let cell_start = fields.u32_fields.get("cell_start").unwrap() as *const forge_runtime::Array<u32>;
            let sorted_idx = fields.u32_fields.get("sorted_indices").unwrap() as *const forge_runtime::Array<u32>;

            unsafe {
                use forge_runtime::cuda::PushKernelArg;
                let mut b = stream.launch_builder(&func);
                b.arg((*px).cuda_slice().unwrap());
                b.arg((*py).cuda_slice().unwrap());
                b.arg((*pz).cuda_slice().unwrap());
                b.arg((*vx).cuda_slice_mut().unwrap());
                b.arg((*vy).cuda_slice_mut().unwrap());
                b.arg((*vz).cuda_slice_mut().unwrap());
                b.arg((*density).cuda_slice().unwrap());
                b.arg((*cell_start).cuda_slice().unwrap());
                b.arg((*sorted_idx).cuda_slice().unwrap());
                b.arg(&self.smoothing_radius);
                b.arg(&self.particle_mass);
                b.arg(&self.gas_constant);
                b.arg(&self.rest_density);
                b.arg(&self.viscosity_coefficient);
                b.arg(&self.cell_size);
                b.arg(&grid_nx);
                b.arg(&grid_ny);
                b.arg(&grid_nz);
                b.arg(&dt);
                b.arg(&n_i32);
                b.launch(config).map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
            }
        }

        // No final sync — Pipeline::step() handles end-of-step barrier
        Ok(())
    }
}
