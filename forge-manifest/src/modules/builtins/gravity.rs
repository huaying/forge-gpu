use crate::modules::{FieldSet, SimModule};
use forge_runtime::ForgeError;
use std::sync::OnceLock;

pub struct GravityModule {
    pub gx: f32,
    pub gy: f32,
    pub gz: f32,
}

impl GravityModule {
    pub fn new(gx: f32, gy: f32, gz: f32) -> Self {
        Self { gx, gy, gz }
    }

    pub fn default_earth() -> Self {
        Self::new(0.0, -9.81, 0.0)
    }
}

static KERNEL: OnceLock<forge_runtime::CompiledKernel> = OnceLock::new();

impl SimModule for GravityModule {
    fn name(&self) -> &str { "gravity" }

    fn execute(&self, fields: &mut FieldSet, dt: f32) -> Result<(), ForgeError> {
        let n = fields.particle_count;

        let kernel = KERNEL.get_or_init(|| {
            forge_runtime::CompiledKernel::compile(
                r#"extern "C" __global__ void gravity(float* vx, float* vy, float* vz, float gx, float gy, float gz, float dt, int n) {
                    int i = blockIdx.x * blockDim.x + threadIdx.x;
                    if (i < n) { vx[i] += gx * dt; vy[i] += gy * dt; vz[i] += gz * dt; }
                }"#,
                "gravity",
            ).expect("compile gravity")
        });

        let func = kernel.get_function(0)?;
        let stream = forge_runtime::cuda::default_stream(0);
        let config = forge_runtime::cuda::LaunchConfig::for_num_elems(n as u32);
        let n_i32 = n as i32;

        let vx = fields.f32_fields.get_mut("vel_x").ok_or_else(|| ForgeError::LaunchFailed("missing vel_x".into()))? as *mut forge_runtime::Array<f32>;
        let vy = fields.f32_fields.get_mut("vel_y").ok_or_else(|| ForgeError::LaunchFailed("missing vel_y".into()))? as *mut forge_runtime::Array<f32>;
        let vz = fields.f32_fields.get_mut("vel_z").ok_or_else(|| ForgeError::LaunchFailed("missing vel_z".into()))? as *mut forge_runtime::Array<f32>;

        unsafe {
            use forge_runtime::cuda::PushKernelArg;
            let mut b = stream.launch_builder(&func);
            b.arg((*vx).cuda_slice_mut().unwrap());
            b.arg((*vy).cuda_slice_mut().unwrap());
            b.arg((*vz).cuda_slice_mut().unwrap());
            b.arg(&self.gx);
            b.arg(&self.gy);
            b.arg(&self.gz);
            b.arg(&dt);
            b.arg(&n_i32);
            b.launch(config).map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
        }

        Ok(())
    }
}
