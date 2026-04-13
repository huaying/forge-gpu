use crate::modules::{FieldSet, SimModule};
use forge_runtime::ForgeError;
use std::sync::OnceLock;

/// SPH density computation module.
///
/// Rebuilds the spatial hash grid from current positions (CPU),
/// then computes density for each particle on GPU using cubic spline kernel.
pub struct SphDensityModule {
    pub smoothing_radius: f32,
    pub cell_size: f32,
    pub grid_dims: [u32; 3],
    pub particle_mass: f32,
}

static KERNEL: OnceLock<forge_runtime::CompiledKernel> = OnceLock::new();

impl SphDensityModule {
    /// Build hash grid on CPU from current positions, upload to FieldSet.
    fn rebuild_hashgrid(&self, fields: &mut FieldSet) -> Result<(), ForgeError> {
        let n = fields.particle_count;
        let px = fields.f32_fields.get("pos_x")
            .ok_or_else(|| ForgeError::LaunchFailed("missing pos_x".into()))?.to_vec();
        let py = fields.f32_fields.get("pos_y")
            .ok_or_else(|| ForgeError::LaunchFailed("missing pos_y".into()))?.to_vec();
        let pz = fields.f32_fields.get("pos_z")
            .ok_or_else(|| ForgeError::LaunchFailed("missing pos_z".into()))?.to_vec();

        let (nx, ny, nz) = (self.grid_dims[0], self.grid_dims[1], self.grid_dims[2]);
        let num_cells = (nx * ny * nz) as usize;
        let cs = self.cell_size;

        // Step 1: Compute cell index for each particle
        let mut cell_idx = vec![0u32; n];
        for i in 0..n {
            let cx = ((px[i] / cs).floor() as i32).max(0).min(nx as i32 - 1) as u32;
            let cy = ((py[i] / cs).floor() as i32).max(0).min(ny as i32 - 1) as u32;
            let cz = ((pz[i] / cs).floor() as i32).max(0).min(nz as i32 - 1) as u32;
            cell_idx[i] = cx + cy * nx + cz * nx * ny;
        }

        // Step 2: Count particles per cell
        let mut counts = vec![0u32; num_cells];
        for &c in &cell_idx {
            if (c as usize) < num_cells {
                counts[c as usize] += 1;
            }
        }

        // Step 3: Prefix sum → cell_start
        let mut starts = vec![0u32; num_cells + 1];
        for i in 0..num_cells {
            starts[i + 1] = starts[i] + counts[i];
        }

        // Step 4: Scatter particles into sorted order
        let mut offsets = starts[..num_cells].to_vec();
        let mut sorted = vec![0u32; n];
        for i in 0..n {
            let c = cell_idx[i] as usize;
            if c < num_cells {
                sorted[offsets[c] as usize] = i as u32;
                offsets[c] += 1;
            }
        }

        // Upload to GPU
        let device = fields.device;
        fields.u32_fields.insert(
            "cell_start".to_string(),
            forge_runtime::Array::from_vec(starts, device),
        );
        fields.u32_fields.insert(
            "sorted_indices".to_string(),
            forge_runtime::Array::from_vec(sorted, device),
        );

        Ok(())
    }
}

impl SimModule for SphDensityModule {
    fn name(&self) -> &str { "sph_density" }

    fn execute(&self, fields: &mut FieldSet, _dt: f32) -> Result<(), ForgeError> {
        let n = fields.particle_count;

        // Rebuild hash grid from current positions
        self.rebuild_hashgrid(fields)?;

        // Ensure density field exists
        if !fields.f32_fields.contains_key("density") {
            fields.add_f32_zeros("density", n);
        }

        let kernel = KERNEL.get_or_init(|| {
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
