//! # Forge
//!
//! A Rust-native GPU compute framework built for the age of AI.
//!
//! Forge brings compile-time safety guarantees to GPU programming,
//! inspired by NVIDIA Warp but reimagined for a world where AI agents
//! write the code.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use forge::prelude::*;
//!
//! #[kernel]
//! fn add_one(data: &mut Array<f32>, n: i32) {
//!     let i = thread_id();
//!     if i < n {
//!         data[i] += 1.0;
//!     }
//! }
//! ```

pub use forge_core as core;
pub use forge_codegen as codegen;
pub use forge_runtime as runtime;

/// Re-export the `#[kernel]` proc macro.
pub use forge_macros::kernel;

/// Common imports for Forge users.
pub mod prelude {
    pub use forge_core::*;
    pub use forge_runtime::{Array, Device, Forge};
    pub use forge_macros::kernel;
}
