//! Error types for Forge runtime.

use std::fmt;

/// Errors that can occur during Forge GPU operations.
#[derive(Debug, Clone)]
pub enum ForgeError {
    /// CUDA kernel compilation failed (nvrtc error).
    CompilationFailed(String),
    /// Failed to load PTX module onto device.
    ModuleLoadFailed(String),
    /// Failed to get a kernel function from the module.
    FunctionLoadFailed(String),
    /// Kernel launch failed.
    LaunchFailed(String),
    /// Device synchronization failed.
    SyncFailed(String),
    /// Array is not on the expected device.
    WrongDevice(String),
    /// CUDA device not found or not available.
    DeviceNotFound(String),
}

impl fmt::Display for ForgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ForgeError::CompilationFailed(msg) => write!(f, "compilation failed: {}", msg),
            ForgeError::ModuleLoadFailed(msg) => write!(f, "module load failed: {}", msg),
            ForgeError::FunctionLoadFailed(msg) => write!(f, "function load failed: {}", msg),
            ForgeError::LaunchFailed(msg) => write!(f, "kernel launch failed: {}", msg),
            ForgeError::SyncFailed(msg) => write!(f, "synchronization failed: {}", msg),
            ForgeError::WrongDevice(msg) => write!(f, "wrong device: {}", msg),
            ForgeError::DeviceNotFound(msg) => write!(f, "device not found: {}", msg),
        }
    }
}

impl std::error::Error for ForgeError {}
