use crate::modules::{FieldSet, SimModule};
use forge_runtime::ForgeError;
use std::sync::OnceLock;

/// SPH density computation module.
///
/// Rebuilds the spatial hash grid entirely on GPU each step,
/// then computes density for each particle using cubic spline (Poly6) kernel.
///
/// GPU hash grid build pipeline:
///   1. Kernel: compute cell index per particle
///   2. Kernel: atomicAdd to count particles per cell
///   3. CPU prefix sum on cell counts (small array, fast)
///   4. Kernel: scatter particles into sorted order
///   5. Kernel: compute density from neighbor queries
pub struct SphDensityModule {
    pub smoothing_radius: f32,
    pub cell_size: f32,
    pub grid_dims: [u32; 3],
    pub particle_mass: f32,
}

static CELL_INDEX_KERNEL: OnceLock<forge_runtime::CompiledKernel> = OnceLock::new();
static COUNT_KERNEL: OnceLock<forge_runtime::CompiledKernel> = OnceLock::new();
static SCATTER_KERNEL: OnceLock<forge_runtime::CompiledKernel> = OnceLock::new();
static DENSITY_KERNEL: OnceLock<forge_runtime::CompiledKernel> = OnceLock::new();

impl SphDensityModule {
    /// Rebuild hash grid on GPU from current positions.
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
        // We need a per-cell offset counter for the scatter pass
        if !fields.u32_fields.contains_key("cell_offset") {
            fields.u32_fields.insert("cell_offset".to_string(),
                forge_runtime::Array::<u32>::zeros(num_cells, device));
        }

        let stream = forge_runtime::cuda::default_stream(0);
        let n_i32 = n as i32;
        let nx_i32 = nx as i32;
        let ny_i32 = ny as i32;
        let nz_i32 = nz as i32;
        let num_cells_i32 = num_cells as i32;

        // ── Step 1: Compute cell index per particle ──
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

        // ── Step 2: Count particles per cell (atomicAdd) ──
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

        // ── Step 4: Scatter particles into sorted order ──
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

        let config_n = forge_runtime::cuda::LaunchConfig::for_num_elems(n as u32);
        let config_cells = forge_runtime::cuda::LaunchConfig::for_num_elems(num_cells as u32);

        // Get raw pointers for simultaneous field access
        let px = fields.f32_fields.get("pos_x").unwrap() as *const forge_runtime::Array<f32>;
        let py = fields.f32_fields.get("pos_y").unwrap() as *const forge_runtime::Array<f32>;
        let pz = fields.f32_fields.get("pos_z").unwrap() as *const forge_runtime::Array<f32>;
        let cell_idx = fields.u32_fields.get_mut("cell_idx").unwrap() as *mut forge_runtime::Array<u32>;
        let cell_count = fields.u32_fields.get_mut("cell_count").unwrap() as *mut forge_runtime::Array<u32>;
        let cell_start = fields.u32_fields.get_mut("cell_start").unwrap() as *mut forge_runtime::Array<u32>;
        let sorted_idx = fields.u32_fields.get_mut("sorted_indices").unwrap() as *mut forge_runtime::Array<u32>;
        let cell_offset = fields.u32_fields.get_mut("cell_offset").unwrap() as *mut forge_runtime::Array<u32>;

        // Zero out cell_count and cell_offset by re-allocating
        // (num_cells is small, this is negligible cost)
        unsafe {
            *cell_count = forge_runtime::Array::<u32>::zeros(num_cells, device);
            *cell_offset = forge_runtime::Array::<u32>::zeros(num_cells, device);
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
        stream.synchronize().map_err(|e| ForgeError::SyncFailed(format!("{:?}", e)))?;

        // Step 3: Prefix sum on cell_count → cell_start
        // This is a small array (num_cells), so CPU prefix sum is fine
        // (GPU prefix sum only wins for >1M cells)
        unsafe {
            let counts = (*cell_count).to_vec();
            let mut starts = vec![0u32; num_cells + 1];
            for i in 0..num_cells {
                starts[i + 1] = starts[i] + counts[i];
            }
            *cell_start = forge_runtime::Array::from_vec(starts, device);
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

impl SimModule for SphDensityModule {
    fn name(&self) -> &str { "sph_density" }

    fn execute(&self, fields: &mut FieldSet, _dt: f32) -> Result<(), ForgeError> {
        let n = fields.particle_count;

        // Rebuild hash grid on GPU
        self.rebuild_hashgrid_gpu(fields)?;

        // Ensure density field exists
        if !fields.f32_fields.contains_key("density") {
            fields.add_f32_zeros("density", n);
        }

        let kernel = DENSITY_KERNEL.get_or_init(|| {
            forge_runtime::CompiledKernel::compile(
                r#"extern "C" __global__ void sph_density(
    const float* px, const float* py, const float* pz,
    float* density,
    const unsigned int* cell_start, const unsigned int* sorted_idx,
    float h, float mass, float cell_size,
    int grid_nx, int grid_ny, int grid_nz,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float xi = px[i], yi = py[i], zi = pz[i];
    float h2 = h * h;
    // Poly6 kernel normalization: 315 / (64 * pi * h^9)
    float h9 = h2 * h2 * h2 * h2 * h;
    float poly6_coeff = 315.0f / (64.0f * 3.14159265f * h9);

    float rho = 0.0f;

    // Find which cell this particle is in
    int cx = (int)floorf(xi / cell_size);
    int cy = (int)floorf(yi / cell_size);
    int cz = (int)floorf(zi / cell_size);
    if (cx < 0) cx = 0; if (cx >= grid_nx) cx = grid_nx - 1;
    if (cy < 0) cy = 0; if (cy >= grid_ny) cy = grid_ny - 1;
    if (cz < 0) cz = 0; if (cz >= grid_nz) cz = grid_nz - 1;

    // Query 3x3x3 neighborhood
    for (int dz = -1; dz <= 1; dz++) {
        int nz = cz + dz;
        if (nz < 0 || nz >= grid_nz) continue;
        for (int dy = -1; dy <= 1; dy++) {
            int ny = cy + dy;
            if (ny < 0 || ny >= grid_ny) continue;
            for (int dx = -1; dx <= 1; dx++) {
                int nx = cx + dx;
                if (nx < 0 || nx >= grid_nx) continue;

                unsigned int cell = (unsigned int)nx
                    + (unsigned int)ny * (unsigned int)grid_nx
                    + (unsigned int)nz * (unsigned int)grid_nx * (unsigned int)grid_ny;
                unsigned int start = cell_start[cell];
                unsigned int end = cell_start[cell + 1];

                for (unsigned int s = start; s < end; s++) {
                    int j = (int)sorted_idx[s];
                    float dx2 = xi - px[j];
                    float dy2 = yi - py[j];
                    float dz2 = zi - pz[j];
                    float r2 = dx2*dx2 + dy2*dy2 + dz2*dz2;
                    if (r2 < h2) {
                        float diff = h2 - r2;
                        rho += mass * poly6_coeff * diff * diff * diff;
                    }
                }
            }
        }
    }

    density[i] = rho;
}"#,
                "sph_density",
            ).expect("compile sph_density")
        });

        let func = kernel.get_function(0)?;
        let stream = forge_runtime::cuda::default_stream(0);
        let config = forge_runtime::cuda::LaunchConfig::for_num_elems(n as u32);
        let n_i32 = n as i32;
        let grid_nx = self.grid_dims[0] as i32;
        let grid_ny = self.grid_dims[1] as i32;
        let grid_nz = self.grid_dims[2] as i32;

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
        stream.synchronize().map_err(|e| ForgeError::SyncFailed(format!("{:?}", e)))?;
        Ok(())
    }
}
