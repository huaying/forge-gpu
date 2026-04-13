use crate::modules::{FieldSet, SimModule};
use crate::expr;
use forge_runtime::ForgeError;
use std::collections::HashMap;
use std::sync::Mutex;

/// Custom expression module — compiles a user expression to a CUDA kernel.
///
/// Example TOML:
/// ```toml
/// [[forces]]
/// type = "custom"
/// expr = "vel.y += sin(pos.x * 3.14) * 0.1"
/// ```
pub struct ExprModule {
    pub expr: String,
    pub kernel_name: String,
    pub cuda_source: String,
    pub reads_density: bool,
}

// Use a global cache for compiled expression kernels
static KERNEL_CACHE: std::sync::LazyLock<Mutex<HashMap<String, forge_runtime::CompiledKernel>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

impl ExprModule {
    pub fn new(expr_str: &str, id: usize) -> Self {
        let kernel_name = format!("expr_kernel_{}", id);
        let fields = expr::analyze_expr(expr_str);
        let cuda_source = expr::compile_expr_to_cuda(expr_str, &kernel_name);

        Self {
            expr: expr_str.to_string(),
            kernel_name,
            cuda_source,
            reads_density: fields.reads_density,
        }
    }
}

impl SimModule for ExprModule {
    fn name(&self) -> &str { "custom_expr" }

    fn execute(&self, fields: &mut FieldSet, dt: f32) -> Result<(), ForgeError> {
        let n = fields.particle_count;

        // Compile kernel (cached)
        let mut cache = KERNEL_CACHE.lock().unwrap();
        if !cache.contains_key(&self.kernel_name) {
            let compiled = forge_runtime::CompiledKernel::compile(
                &self.cuda_source,
                &self.kernel_name,
            ).map_err(|e| ForgeError::LaunchFailed(
                format!("Failed to compile expression '{}': {:?}", self.expr, e)
            ))?;
            cache.insert(self.kernel_name.clone(), compiled);
        }
        let kernel = cache.get(&self.kernel_name).unwrap();

        let func = kernel.get_function(0)?;
        let stream = forge_runtime::cuda::default_stream(0);
        let config = forge_runtime::cuda::LaunchConfig::for_num_elems(n as u32);
        let n_i32 = n as i32;

        let px = fields.f32_fields.get_mut("pos_x")
            .ok_or_else(|| ForgeError::LaunchFailed("missing pos_x".into()))? as *mut forge_runtime::Array<f32>;
        let py = fields.f32_fields.get_mut("pos_y")
            .ok_or_else(|| ForgeError::LaunchFailed("missing pos_y".into()))? as *mut forge_runtime::Array<f32>;
        let pz = fields.f32_fields.get_mut("pos_z")
            .ok_or_else(|| ForgeError::LaunchFailed("missing pos_z".into()))? as *mut forge_runtime::Array<f32>;
        let vx = fields.f32_fields.get_mut("vel_x")
            .ok_or_else(|| ForgeError::LaunchFailed("missing vel_x".into()))? as *mut forge_runtime::Array<f32>;
        let vy = fields.f32_fields.get_mut("vel_y")
            .ok_or_else(|| ForgeError::LaunchFailed("missing vel_y".into()))? as *mut forge_runtime::Array<f32>;
        let vz = fields.f32_fields.get_mut("vel_z")
            .ok_or_else(|| ForgeError::LaunchFailed("missing vel_z".into()))? as *mut forge_runtime::Array<f32>;

        if self.reads_density {
            let density = fields.f32_fields.get("density")
                .ok_or_else(|| ForgeError::LaunchFailed("expression uses 'density' but field not found".into()))?
                as *const forge_runtime::Array<f32>;

            unsafe {
                use forge_runtime::cuda::PushKernelArg;
                let mut b = stream.launch_builder(&func);
                b.arg((*px).cuda_slice_mut().unwrap());
                b.arg((*py).cuda_slice_mut().unwrap());
                b.arg((*pz).cuda_slice_mut().unwrap());
                b.arg((*vx).cuda_slice_mut().unwrap());
                b.arg((*vy).cuda_slice_mut().unwrap());
                b.arg((*vz).cuda_slice_mut().unwrap());
                b.arg((*density).cuda_slice().unwrap());
                b.arg(&dt);
                b.arg(&n_i32);
                b.launch(config).map_err(|e| ForgeError::LaunchFailed(format!("{:?}", e)))?;
            }
        } else {
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
        }


        Ok(())
    }
}
