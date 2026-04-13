use crate::modules::{FieldSet, SimModule};
use forge_runtime::ForgeError;
use std::sync::OnceLock;

/// SPH viscosity module.
///
/// Applies viscosity force using the viscosity kernel laplacian.
/// Smooths out velocity differences between neighboring particles.
pub struct SphViscosityModule {
    pub coefficient: f32,
    pub smoothing_radius: f32,
    pub cell_size: f32,
    pub grid_dims: [u32; 3],
    pub particle_mass: f32,
}

static KERNEL: OnceLock<forge_runtime::CompiledKernel> = OnceLock::new();

impl SimModule for SphViscosityModule {
    fn name(&self) -> &str { "sph_viscosity" }

    fn execute(&self, fields: &mut FieldSet, dt: f32) -> Result<(), ForgeError> {
        let n = fields.particle_count;

        let kernel = KERNEL.get_or_init(|| {
            forge_runtime::CompiledKernel::compile(
                r#"extern "C" __global__ void sph_viscosity(
    const float* px, const float* py, const float* pz,
    float* vx, float* vy, float* vz,
    const float* density,
    const unsigned int* cell_start, const unsigned int* sorted_idx,
    float h, float mass, float visc_coeff,
    float cell_size, int grid_nx, int grid_ny, int grid_nz,
    float dt, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float xi = px[i], yi = py[i], zi = pz[i];
    float vxi = vx[i], vyi = vy[i], vzi = vz[i];
    float rho_i = density[i];
    if (rho_i < 1e-6f) return;

    float h2 = h * h;
    // Viscosity kernel laplacian: 45 / (pi * h^6) * (h - r)
    float h6 = h2 * h2 * h2;
    float visc_lap_coeff = 45.0f / (3.14159265f * h6);

    float ax = 0.0f, ay = 0.0f, az = 0.0f;

    int cx = (int)floorf(xi / cell_size);
    int cy = (int)floorf(yi / cell_size);
    int cz = (int)floorf(zi / cell_size);
    if (cx < 0) cx = 0; if (cx >= grid_nx) cx = grid_nx - 1;
    if (cy < 0) cy = 0; if (cy >= grid_ny) cy = grid_ny - 1;
    if (cz < 0) cz = 0; if (cz >= grid_nz) cz = grid_nz - 1;

    for (int dz = -1; dz <= 1; dz++) {
        int nzz = cz + dz;
        if (nzz < 0 || nzz >= grid_nz) continue;
        for (int dy = -1; dy <= 1; dy++) {
            int nyy = cy + dy;
            if (nyy < 0 || nyy >= grid_ny) continue;
            for (int dx = -1; dx <= 1; dx++) {
                int nxx = cx + dx;
                if (nxx < 0 || nxx >= grid_nx) continue;

                unsigned int cell = (unsigned int)nxx
                    + (unsigned int)nyy * (unsigned int)grid_nx
                    + (unsigned int)nzz * (unsigned int)grid_nx * (unsigned int)grid_ny;
                unsigned int start = cell_start[cell];
                unsigned int end = cell_start[cell + 1];

                for (unsigned int s = start; s < end; s++) {
                    int j = (int)sorted_idx[s];
                    if (j == i) continue;

                    float djx = xi - px[j];
                    float djy = yi - py[j];
                    float djz = zi - pz[j];
                    float r2 = djx*djx + djy*djy + djz*djz;

                    if (r2 < h2 && r2 > 1e-12f) {
                        float r = sqrtf(r2);
                        float rho_j = density[j];
                        if (rho_j < 1e-6f) continue;

                        // Viscosity laplacian kernel value
                        float lap = visc_lap_coeff * (h - r);

                        // Viscosity force: mu * mass * (v_j - v_i) / rho_j * lap_W
                        float scale = visc_coeff * mass / rho_j * lap;
                        ax += scale * (vx[j] - vxi);
                        ay += scale * (vy[j] - vyi);
                        az += scale * (vz[j] - vzi);
                    }
                }
            }
        }
    }

    // Apply: vel += (viscosity_accel / rho_i) * dt
    vx[i] += (ax / rho_i) * dt;
    vy[i] += (ay / rho_i) * dt;
    vz[i] += (az / rho_i) * dt;
}"#,
                "sph_viscosity",
            ).expect("compile sph_viscosity")
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
            b.arg(&self.coefficient);
            b.arg(&self.cell_size);
            b.arg(&grid_nx);
            b.arg(&grid_ny);
            b.arg(&grid_nz);
            b.arg(&dt);
            b.arg(&n_i32);
            b.launch(config).map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
        }

        Ok(())
    }
}
