//! # forge-manifest — Declarative TOML simulation manifests.
//!
//! Parse a TOML file describing a simulation, compile it to GPU kernels, and run it.
//!
//! ## Supported Simulation Types
//!
//! - **particles**: N-body particle simulation with forces and constraints
//! - **springs**: Spring-mass system with elastic energy
//!
//! ## TOML Schema
//!
//! ```toml
//! [simulation]
//! name = "my-sim"
//! type = "particles"  # or "springs"
//! dt = 0.001
//! substeps = 4
//! duration = 5.0
//!
//! [[fields]]
//! name = "position"
//! type = "vec3f"
//! count = 100000
//! init = { type = "random", min = [-5, 10, -5], max = [5, 20, 5] }
//!
//! [[forces]]
//! type = "gravity"
//! value = [0, -9.81, 0]
//!
//! [[constraints]]
//! type = "ground_plane"
//! y = 0.0
//! restitution = 0.7
//! ```

mod schema;
mod runner;
mod validate;
pub mod modules;
pub mod expr;
pub mod serve;
pub mod phantom_bridge;
mod manifest_runner;

pub use schema::*;
pub use runner::*;
pub use validate::*;
pub use manifest_runner::*;
