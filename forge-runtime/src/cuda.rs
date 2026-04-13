//! CUDA context management via cudarc.

use cudarc::driver::safe::CudaContext as CudarcContext;
use std::sync::{Arc, OnceLock};

// Re-export CudaStream for public use
pub use cudarc::driver::safe::CudaStream;

/// Thread-safe CUDA contexts, lazily initialized.
static CUDA_CONTEXTS: OnceLock<Vec<Arc<CudarcContext>>> = OnceLock::new();

/// Non-default simulation stream (supports graph capture).
static SIM_STREAM: OnceLock<Arc<CudaStream>> = OnceLock::new();

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
    // Use the simulation stream (non-default, graph-capture compatible)
    SIM_STREAM.get_or_init(|| {
        let ctx = get_context(ordinal);
        // Disable event tracking — it inserts cross-stream events that break CUDA Graph capture
        unsafe { ctx.disable_event_tracking(); }
        ctx.new_stream().expect("Failed to create simulation stream")
    }).clone()
}

/// Synchronize a CUDA device (waits for all pending operations on default stream).
pub fn synchronize(ordinal: usize) {
    let stream = default_stream(ordinal);
    stream.synchronize().expect("CUDA synchronize failed");
}

/// Create a new non-default stream on the given device.
pub fn new_stream(ordinal: usize) -> Arc<CudaStream> {
    let ctx = get_context(ordinal);
    ctx.new_stream().expect("Failed to create CUDA stream")
}

/// Fork a child stream from an existing stream (for parallel work).
pub fn fork_stream(parent: &Arc<CudaStream>) -> Arc<CudaStream> {
    parent.fork().expect("Failed to fork CUDA stream")
}

/// Join a child stream back into a parent (wait for child to finish).
/// Records an event on the child stream, then makes the parent wait for it.
pub fn join_stream(parent: &Arc<CudaStream>, child: &Arc<CudaStream>) {
    let ctx = child.context();
    let event = ctx.new_event(None).expect("Failed to create event");
    event.record(child).expect("Failed to record event on child stream");
    parent.wait(&event).expect("Failed to wait on event");
}

/// Get device memory info (free, total) in bytes.
pub fn mem_info(ordinal: usize) -> (usize, usize) {
    let ctx = get_context(ordinal);
    ctx.mem_get_info().expect("Failed to get memory info")
}

/// Get device name.
pub fn device_name(ordinal: usize) -> String {
    let ctx = get_context(ordinal);
    ctx.name().expect("Failed to get device name")
}

/// Get device compute capability (major, minor).
pub fn compute_capability(ordinal: usize) -> (u32, u32) {
    let ctx = get_context(ordinal);
    let (major, minor) = ctx.compute_capability().expect("Failed to get compute capability");
    (major as u32, minor as u32)
}

/// Public re-export for advanced usage.
pub use cudarc::driver::safe::{
    CudaFunction, CudaModule, CudaSlice, CudaView, CudaViewMut, LaunchConfig, LaunchArgs,
    CudaGraph, CudaEvent,
};
pub use cudarc::driver::sys::{CUstreamCaptureMode, CUgraphInstantiate_flags};
pub use cudarc::driver::{DevicePtr, DevicePtrMut, DeviceRepr, DeviceSlice, PushKernelArg};
pub use cudarc::nvrtc::Ptx;

/// Re-exported context type.
pub type CudaContext = CudarcContext;
