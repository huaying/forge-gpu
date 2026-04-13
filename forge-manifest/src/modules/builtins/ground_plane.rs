use crate::modules::{FieldSet, SimModule};
use forge_runtime::ForgeError;
use std::sync::OnceLock;

pub struct GroundPlaneModule {
    pub y: f32,
    pub restitution: f32,
}

static KERNEL: OnceLock<forge_runtime::CompiledKernel> = OnceLock::new();

impl SimModule for GroundPlaneModule {
    fn name(&self) -> &str { "ground_plane" }

    fn execute(&self, fields: &mut FieldSet, _dt: f32) -> Result<(), ForgeError> {
        let n = fields.particle_count;
        let kernel = KERNEL.get_or_init(|| {
            forge_runtime::CompiledKernel::compile(
                r#"extern "C" __global__ void ground(float* py, float* vy, float ground_y, float rest, int n) {
                    int i = blockIdx.x * blockDim.x + threadIdx.x;
                    if (i < n && py[i] < ground_y) { py[i] = ground_y; vy[i] = -vy[i] * rest; }
                }"#,
                "ground",
            ).expect("compile ground")
        });

        let func = kernel.get_function(0)?;
        let stream = forge_runtime::cuda::default_stream(0);
        let config = forge_runtime::cuda::LaunchConfig::for_num_elems(n as u32);
        let n_i32 = n as i32;

        let py = fields.f32_fields.get_mut("pos_y").ok_or_else(|| ForgeError::LaunchFailed("missing pos_y".into()))?
            as *mut forge_runtime::Array<f32>;
        let vy = fields.f32_fields.get_mut("vel_y").ok_or_else(|| ForgeError::LaunchFailed("missing vel_y".into()))?
            as *mut forge_runtime::Array<f32>;

        unsafe {
            use forge_runtime::cuda::PushKernelArg;
            let mut b = stream.launch_builder(&func);
            b.arg((*py).cuda_slice_mut().unwrap());
            b.arg((*vy).cuda_slice_mut().unwrap());
            b.arg(&self.y);
            b.arg(&self.restitution);
            b.arg(&n_i32);
            b.launch(config).map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
        }

        Ok(())
    }
}
