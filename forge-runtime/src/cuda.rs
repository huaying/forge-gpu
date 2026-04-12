//! CUDA context management via cudarc.

use cudarc::driver::safe::{CudaContext as CudarcContext, CudaStream};
use std::sync::{Arc, OnceLock};

/// Thread-safe CUDA contexts, lazily initialized.
static CUDA_CONTEXTS: OnceLock<Vec<Arc<CudarcContext>>> = OnceLock::new();

/// Initialize the CUDA driver (idempotent).
pub fn init() {
    let _ = CUDA_CONTEXTS.get_or_init(|| {
        let count = CudarcContext::device_count().unwrap_or(0) as usize;
        let mut contexts = Vec::with_capacity(count);
        for i in 0..count {
            match CudarcContext::new(i) {
                Ok(ctx) => contexts.push(ctx),
                Err(e) => eprintln!("Warning: failed to initialize CUDA device {}: {}", i, e),
            }
        }
        contexts
    });
}

/// Number of available CUDA devices.
pub fn device_count() -> usize {
    match CUDA_CONTEXTS.get() {
        Some(ctxs) => ctxs.len(),
        None => CudarcContext::device_count().unwrap_or(0) as usize,
    }
}

/// Get a handle to a CUDA context for a device.
pub fn get_context(ordinal: usize) -> Arc<CudarcContext> {
    init();
    let contexts = CUDA_CONTEXTS.get().expect("CUDA not initialized");
    contexts
        .get(ordinal)
        .unwrap_or_else(|| {
            panic!(
                "CUDA device {} not found (have {})",
                ordinal,
                contexts.len()
            )
        })
        .clone()
}

/// Get the default stream for a device.
pub fn default_stream(ordinal: usize) -> Arc<CudaStream> {
    get_context(ordinal).default_stream()
}

/// Synchronize a CUDA device (waits for all pending operations on default stream).
pub fn synchronize(ordinal: usize) {
    let stream = default_stream(ordinal);
    stream.synchronize().expect("CUDA synchronize failed");
}

/// Public re-export for advanced usage.
pub use cudarc::driver::safe::{
    CudaFunction, CudaModule, CudaSlice, CudaView, CudaViewMut, LaunchConfig,
};
pub use cudarc::driver::{DevicePtr, DevicePtrMut, DeviceRepr, DeviceSlice, PushKernelArg};
pub use cudarc::nvrtc::Ptx;

/// Re-exported context type.
pub type CudaContext = CudarcContext;
