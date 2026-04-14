use crate::modules::{FieldSet, SimModule};
use forge_runtime::ForgeError;
use std::sync::OnceLock;

pub struct IntegrateModule;

static KERNEL: OnceLock<forge_runtime::CompiledKernel> = OnceLock::new();

impl SimModule for IntegrateModule {
    fn name(&self) -> &str { "integrate" }

    fn execute(&self, fields: &mut FieldSet, dt: f32) -> Result<(), ForgeError> {
        let n = fields.particle_count;
        let kernel = KERNEL.get_or_init(|| {
            forge_runtime::CompiledKernel::compile(
                r#"extern "C" __global__ void integrate(
                    float* px, float* py, float* pz,
                    float* vx, float* vy, float* vz,
                    float dt, int n) {
                    int i = blockIdx.x * blockDim.x + threadIdx.x;
                    if (i < n) {
                        // NaN/Inf safety: clamp velocity to prevent simulation explosion
                        float _vx = vx[i], _vy = vy[i], _vz = vz[i];
                        if (!isfinite(_vx)) { _vx = 0.0f; vx[i] = 0.0f; }
                        if (!isfinite(_vy)) { _vy = 0.0f; vy[i] = 0.0f; }
                        if (!isfinite(_vz)) { _vz = 0.0f; vz[i] = 0.0f; }
                        px[i] += _vx * dt;
                        py[i] += _vy * dt;
                        pz[i] += _vz * dt;
                    }
                }"#,
                "integrate",
            ).expect("compile integrate")
        });

        let func = kernel.get_function(0)?;
        let stream = forge_runtime::cuda::default_stream(0);
        let config = forge_runtime::cuda::LaunchConfig::for_num_elems(n as u32);
        let n_i32 = n as i32;

        // Need mutable refs to 6 fields simultaneously — get raw pointers
        let px = fields.f32_fields.get_mut("pos_x").ok_or_else(|| ForgeError::LaunchFailed("missing pos_x".into()))?
            as *mut forge_runtime::Array<f32>;
        let py = fields.f32_fields.get_mut("pos_y").ok_or_else(|| ForgeError::LaunchFailed("missing pos_y".into()))?
            as *mut forge_runtime::Array<f32>;
        let pz = fields.f32_fields.get_mut("pos_z").ok_or_else(|| ForgeError::LaunchFailed("missing pos_z".into()))?
            as *mut forge_runtime::Array<f32>;
        let vx = fields.f32_fields.get_mut("vel_x").ok_or_else(|| ForgeError::LaunchFailed("missing vel_x".into()))?
            as *mut forge_runtime::Array<f32>;
        let vy = fields.f32_fields.get_mut("vel_y").ok_or_else(|| ForgeError::LaunchFailed("missing vel_y".into()))?
            as *mut forge_runtime::Array<f32>;
        let vz = fields.f32_fields.get_mut("vel_z").ok_or_else(|| ForgeError::LaunchFailed("missing vel_z".into()))?
            as *mut forge_runtime::Array<f32>;

        unsafe {
            use forge_runtime::cuda::PushKernelArg;
            let mut b = stream.launch_builder(&func);
            b.arg((*px).cuda_slice_mut().unwrap());
            b.arg((*py).cuda_slice_mut().unwrap());
            b.arg((*pz).cuda_slice_mut().unwrap());
            b.arg((*vx).cuda_slice_mut().unwrap());
            b.arg((*vy).cuda_slice_mut().unwrap());
            b.arg((*vz).cuda_slice_mut().unwrap());
            b.arg(&dt);
            b.arg(&n_i32);
            b.launch(config).map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
        }

        Ok(())
    }
}
