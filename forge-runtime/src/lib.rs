//! # forge-runtime
//!
//! GPU runtime for Forge compute framework.
//!
//! Provides:
//! - Device discovery and management (CUDA, CPU fallback)
//! - GPU memory allocation and transfer (host ↔ device)
//! - `Array<T>` — typed GPU arrays with ownership semantics
//! - Kernel compilation (nvrtc) and launch dispatch
//!
//! Enable the `cuda` feature for GPU support. Without it, only CPU mode is available.

mod device;
mod array;
mod error;
#[cfg(feature = "cuda")]
pub mod cuda;
mod kernel;
mod tape;
mod hashgrid;

pub use device::*;
pub use array::*;
pub use error::*;
#[cfg(feature = "cuda")]
pub use cuda::CudaContext;
pub use kernel::*;
pub use tape::Tape;
pub use hashgrid::HashGrid;
