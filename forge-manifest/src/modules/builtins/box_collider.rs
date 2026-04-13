use crate::modules::{FieldSet, SimModule};
use forge_runtime::ForgeError;
use std::sync::OnceLock;

/// Box collider constraint — clamps particles to within a box, reflects velocity.
pub struct BoxColliderModule {
    pub min_x: f32,
    pub min_y: f32,
    pub min_z: f32,
    pub max_x: f32,
    pub max_y: f32,
    pub max_z: f32,
    pub restitution: f32,
}

static KERNEL: OnceLock<forge_runtime::CompiledKernel> = OnceLock::new();

impl SimModule for BoxColliderModule {
    fn name(&self) -> &str { "box_collider" }

    fn execute(&self, fields: &mut FieldSet, _dt: f32) -> Result<(), ForgeError> {
        let n = fields.particle_count;

        let kernel = KERNEL.get_or_init(|| {
            forge_runtime::CompiledKernel::compile(
                r#"extern "C" __global__ void box_collider(
    float* px, float* py, float* pz,
    float* vx, float* vy, float* vz,
    float min_x, float min_y, float min_z,
    float max_x, float max_y, float max_z,
    float restitution, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // X axis
    if (px[i] < min_x) { px[i] = min_x; vx[i] = fabsf(vx[i]) * restitution; }
    if (px[i] > max_x) { px[i] = max_x; vx[i] = -fabsf(vx[i]) * restitution; }

    // Y axis
    if (py[i] < min_y) { py[i] = min_y; vy[i] = fabsf(vy[i]) * restitution; }
    if (py[i] > max_y) { py[i] = max_y; vy[i] = -fabsf(vy[i]) * restitution; }

    // Z axis
    if (pz[i] < min_z) { pz[i] = min_z; vz[i] = fabsf(vz[i]) * restitution; }
    if (pz[i] > max_z) { pz[i] = max_z; vz[i] = -fabsf(vz[i]) * restitution; }
}"#,
                "box_collider",
            ).expect("compile box_collider")
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
            b.arg(&self.min_x);
            b.arg(&self.min_y);
            b.arg(&self.min_z);
            b.arg(&self.max_x);
            b.arg(&self.max_y);
            b.arg(&self.max_z);
            b.arg(&self.restitution);
            b.arg(&n_i32);
            b.launch(config).map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
        }

        Ok(())
    }
}
