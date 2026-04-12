//! Kernel compilation (nvrtc) and launch.

#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};

/// Compiled CUDA kernel ready for launch.
#[cfg(feature = "cuda")]
pub struct CompiledKernel {
    /// The compiled PTX source.
    ptx: cudarc::nvrtc::Ptx,
    /// Kernel function name in the PTX.
    name: String,
    /// Loaded modules per device ordinal.
    modules: Mutex<std::collections::HashMap<usize, Arc<cudarc::driver::safe::CudaModule>>>,
}

#[cfg(feature = "cuda")]
impl CompiledKernel {
    /// Compile CUDA C++ source to PTX via nvrtc.
    pub fn compile(source: &str, kernel_name: &str) -> Result<Self, crate::ForgeError> {
        let ptx = cudarc::nvrtc::safe::compile_ptx_with_opts(
            source,
            cudarc::nvrtc::CompileOptions {
                arch: None,
                include_paths: vec![],
                ..Default::default()
            },
        )
        .map_err(|e| crate::ForgeError::CompilationFailed(format!("{}", e)))?;

        Ok(Self {
            ptx,
            name: kernel_name.to_string(),
            modules: Mutex::new(std::collections::HashMap::new()),
        })
    }

    /// Get the CudaFunction handle for a specific device, loading if needed.
    pub fn get_function(
        &self,
        ordinal: usize,
    ) -> Result<cudarc::driver::safe::CudaFunction, crate::ForgeError> {
        let mut modules = self.modules.lock().unwrap();

        if !modules.contains_key(&ordinal) {
            let ctx = crate::cuda::get_context(ordinal);
            let module = ctx
                .load_module(self.ptx.clone())
                .map_err(|e| crate::ForgeError::ModuleLoadFailed(format!("{}", e)))?;
            modules.insert(ordinal, module);
        }

        let module = modules.get(&ordinal).unwrap();
        module
            .load_function(&self.name)
            .map_err(|e| crate::ForgeError::FunctionLoadFailed(format!("'{}': {}", self.name, e)))
    }

    /// Launch this kernel on a device with the given dimension.
    pub fn launch<F>(&self, ordinal: usize, _dim: usize, setup: F) -> Result<(), crate::ForgeError>
    where
        F: FnOnce(&cudarc::driver::safe::CudaFunction, &Arc<cudarc::driver::safe::CudaStream>) -> Result<(), crate::ForgeError>,
    {
        let func = self.get_function(ordinal)?;
        let stream = crate::cuda::default_stream(ordinal);
        setup(&func, &stream)
    }
}

/// Compute grid/block dimensions for a 1D launch.
#[cfg(feature = "cuda")]
pub fn launch_config_1d(dim: usize) -> cudarc::driver::safe::LaunchConfig {
    let block_size = 256u32;
    let grid_size = ((dim as u32) + block_size - 1) / block_size;
    cudarc::driver::safe::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    }
}
