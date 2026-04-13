use crate::modules::{FieldSet, SimModule};
use forge_runtime::ForgeError;
use std::sync::OnceLock;

pub struct SphereColliderModule {
    pub cx: f32,
    pub cy: f32,
    pub cz: f32,
    pub radius: f32,
    pub restitution: f32,
}

static KERNEL: OnceLock<forge_runtime::CompiledKernel> = OnceLock::new();

impl SimModule for SphereColliderModule {
    fn name(&self) -> &str { "sphere_collider" }

    fn execute(&self, fields: &mut FieldSet, _dt: f32) -> Result<(), ForgeError> {
        let n = fields.particle_count;
        let kernel = KERNEL.get_or_init(|| {
            forge_runtime::CompiledKernel::compile(
                r#"extern "C" __global__ void sphere_collide(
                    float* px, float* py, float* pz,
                    float* vx, float* vy, float* vz,
                    float cx, float cy, float cz, float r, float rest, int n) {
                    int i = blockIdx.x * blockDim.x + threadIdx.x;
                    if (i < n) {
                        float dx = px[i]-cx, dy = py[i]-cy, dz = pz[i]-cz;
                        float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                        if (dist < r && dist > 1e-8f) {
                            float inv = 1.0f / dist;
                            float nx = dx*inv, ny = dy*inv, nz = dz*inv;
                            px[i] = cx + nx*r; py[i] = cy + ny*r; pz[i] = cz + nz*r;
                            float vn = vx[i]*nx + vy[i]*ny + vz[i]*nz;
                            if (vn < 0) {
                                vx[i] -= (1+rest)*vn*nx;
                                vy[i] -= (1+rest)*vn*ny;
                                vz[i] -= (1+rest)*vn*nz;
                            }
                        }
                    }
                }"#,
                "sphere_collide",
            ).expect("compile sphere")
        });

        let func = kernel.get_function(0)?;
        let stream = forge_runtime::cuda::default_stream(0);
        let config = forge_runtime::cuda::LaunchConfig::for_num_elems(n as u32);
        let n_i32 = n as i32;

        let px = fields.f32_fields.get_mut("pos_x").unwrap() as *mut forge_runtime::Array<f32>;
        let py = fields.f32_fields.get_mut("pos_y").unwrap() as *mut forge_runtime::Array<f32>;
        let pz = fields.f32_fields.get_mut("pos_z").unwrap() as *mut forge_runtime::Array<f32>;
        let vx = fields.f32_fields.get_mut("vel_x").unwrap() as *mut forge_runtime::Array<f32>;
        let vy = fields.f32_fields.get_mut("vel_y").unwrap() as *mut forge_runtime::Array<f32>;
        let vz = fields.f32_fields.get_mut("vel_z").unwrap() as *mut forge_runtime::Array<f32>;

        unsafe {
            use forge_runtime::cuda::PushKernelArg;
            let mut b = stream.launch_builder(&func);
            b.arg((*px).cuda_slice_mut().unwrap());
            b.arg((*py).cuda_slice_mut().unwrap());
            b.arg((*pz).cuda_slice_mut().unwrap());
            b.arg((*vx).cuda_slice_mut().unwrap());
            b.arg((*vy).cuda_slice_mut().unwrap());
            b.arg((*vz).cuda_slice_mut().unwrap());
            b.arg(&self.cx); b.arg(&self.cy); b.arg(&self.cz);
            b.arg(&self.radius); b.arg(&self.restitution);
            b.arg(&n_i32);
            b.launch(config).map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
        }

        Ok(())
    }
}
