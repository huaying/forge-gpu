use crate::modules::{FieldSet, SimModule};
use forge_runtime::ForgeError;
use std::sync::OnceLock;

pub struct SpringModule {
    pub stiffness: f32,
    pub damping: f32,
}

static KERNEL: OnceLock<forge_runtime::CompiledKernel> = OnceLock::new();

impl SimModule for SpringModule {
    fn name(&self) -> &str { "spring" }

    fn execute(&self, fields: &mut FieldSet, dt: f32) -> Result<(), ForgeError> {
        let n_springs = fields.index_pairs.get("springs")
            .map(|p| p.len())
            .unwrap_or(0);
        if n_springs == 0 { return Ok(()); }

        // ── First call: upload spring topology + rest lengths to GPU (cached) ──
        if !fields.i32_fields.contains_key("spring_i") {
            let pairs = fields.index_pairs.get("springs").unwrap();
            let si: Vec<i32> = pairs.iter().map(|p| p[0] as i32).collect();
            let sj: Vec<i32> = pairs.iter().map(|p| p[1] as i32).collect();
            fields.i32_fields.insert("spring_i".to_string(),
                forge_runtime::Array::from_vec(si, fields.device));
            fields.i32_fields.insert("spring_j".to_string(),
                forge_runtime::Array::from_vec(sj, fields.device));
        }

        if !fields.f32_fields.contains_key("rest_length") {
            let pairs = fields.index_pairs.get("springs").unwrap();
            let px = fields.f32_fields.get("pos_x").unwrap().to_vec();
            let py = fields.f32_fields.get("pos_y").unwrap().to_vec();
            let pz = fields.f32_fields.get("pos_z").unwrap().to_vec();
            let rests: Vec<f32> = pairs.iter().map(|p| {
                let dx = px[p[0] as usize] - px[p[1] as usize];
                let dy = py[p[0] as usize] - py[p[1] as usize];
                let dz = pz[p[0] as usize] - pz[p[1] as usize];
                (dx*dx + dy*dy + dz*dz).sqrt()
            }).collect();
            fields.f32_fields.insert("rest_length".to_string(),
                forge_runtime::Array::from_vec(rests, fields.device));
        }

        // ── Kernel (compiled once) ──
        let kernel = KERNEL.get_or_init(|| {
            forge_runtime::CompiledKernel::compile(
                r#"extern "C" __global__ void spring_force(
                    float* px, float* py, float* pz,
                    float* vx, float* vy, float* vz,
                    const int* si, const int* sj,
                    const float* rest,
                    float k, float damp, float dt, int n_springs) {
                    int tid = blockIdx.x * blockDim.x + threadIdx.x;
                    if (tid < n_springs) {
                        int i = si[tid], j = sj[tid];
                        float dx = px[i]-px[j], dy = py[i]-py[j], dz = pz[i]-pz[j];
                        float dist = sqrtf(dx*dx + dy*dy + dz*dz) + 1e-8f;
                        float stretch = dist - rest[tid];
                        float f = -k * stretch / dist;
                        float fx = f*dx, fy = f*dy, fz = f*dz;
                        // Damping
                        float dvx = vx[i]-vx[j], dvy = vy[i]-vy[j], dvz = vz[i]-vz[j];
                        float vrel = (dvx*dx + dvy*dy + dvz*dz) / dist;
                        float fd = -damp * vrel / dist;
                        fx += fd*dx; fy += fd*dy; fz += fd*dz;
                        // Apply forces (atomicAdd for thread safety)
                        atomicAdd(&vx[i], fx*dt); atomicAdd(&vy[i], fy*dt); atomicAdd(&vz[i], fz*dt);
                        atomicAdd(&vx[j], -fx*dt); atomicAdd(&vy[j], -fy*dt); atomicAdd(&vz[j], -fz*dt);
                    }
                }"#,
                "spring_force",
            ).expect("compile spring")
        });

        // ── Launch (all data already on GPU — graph-safe) ──
        let func = kernel.get_function(0)?;
        let stream = forge_runtime::cuda::default_stream(0);
        let config = forge_runtime::cuda::LaunchConfig::for_num_elems(n_springs as u32);
        let ns_i32 = n_springs as i32;

        let px = fields.f32_fields.get_mut("pos_x").unwrap() as *mut forge_runtime::Array<f32>;
        let py = fields.f32_fields.get_mut("pos_y").unwrap() as *mut forge_runtime::Array<f32>;
        let pz = fields.f32_fields.get_mut("pos_z").unwrap() as *mut forge_runtime::Array<f32>;
        let vx = fields.f32_fields.get_mut("vel_x").unwrap() as *mut forge_runtime::Array<f32>;
        let vy = fields.f32_fields.get_mut("vel_y").unwrap() as *mut forge_runtime::Array<f32>;
        let vz = fields.f32_fields.get_mut("vel_z").unwrap() as *mut forge_runtime::Array<f32>;
        let spring_i = fields.i32_fields.get("spring_i").unwrap() as *const forge_runtime::Array<i32>;
        let spring_j = fields.i32_fields.get("spring_j").unwrap() as *const forge_runtime::Array<i32>;
        let rest = fields.f32_fields.get("rest_length").unwrap() as *const forge_runtime::Array<f32>;

        unsafe {
            use forge_runtime::cuda::PushKernelArg;
            let mut b = stream.launch_builder(&func);
            b.arg((*px).cuda_slice_mut().unwrap());
            b.arg((*py).cuda_slice_mut().unwrap());
            b.arg((*pz).cuda_slice_mut().unwrap());
            b.arg((*vx).cuda_slice_mut().unwrap());
            b.arg((*vy).cuda_slice_mut().unwrap());
            b.arg((*vz).cuda_slice_mut().unwrap());
            b.arg((*spring_i).cuda_slice().unwrap());
            b.arg((*spring_j).cuda_slice().unwrap());
            b.arg((*rest).cuda_slice().unwrap());
            b.arg(&self.stiffness);
            b.arg(&self.damping);
            b.arg(&dt);
            b.arg(&ns_i32);
            b.launch(config).map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
        }

        Ok(())
    }
}
