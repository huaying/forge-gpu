//! TOML schema definitions for simulation manifests.

use serde::Deserialize;

/// Top-level manifest.
#[derive(Debug, Deserialize)]
pub struct SimManifest {
    pub simulation: SimConfig,
    #[serde(default)]
    pub fields: Vec<FieldDef>,
    #[serde(default)]
    pub forces: Vec<ForceDef>,
    #[serde(default)]
    pub constraints: Vec<ConstraintDef>,
    #[serde(default)]
    pub springs: Option<SpringConfig>,
    #[serde(default)]
    pub spatial: Option<SpatialConfig>,
    #[serde(default)]
    pub output: Option<OutputConfig>,
}

/// Simulation configuration.
#[derive(Debug, Deserialize)]
pub struct SimConfig {
    pub name: String,
    #[serde(default = "default_sim_type")]
    pub r#type: String,
    #[serde(default = "default_dt")]
    pub dt: f64,
    #[serde(default = "default_substeps")]
    pub substeps: u32,
    #[serde(default = "default_duration")]
    pub duration: f64,
    /// Number of particles (shorthand, can also use fields)
    #[serde(default)]
    pub count: Option<usize>,
}

fn default_sim_type() -> String { "particles".to_string() }
fn default_dt() -> f64 { 0.001 }
fn default_substeps() -> u32 { 1 }
fn default_duration() -> f64 { 1.0 }

/// Field definition (particle attribute).
#[derive(Debug, Deserialize)]
pub struct FieldDef {
    pub name: String,
    #[serde(default = "default_field_type")]
    pub r#type: String,
    #[serde(default)]
    pub count: Option<usize>,
    #[serde(default)]
    pub init: Option<InitDef>,
}

fn default_field_type() -> String { "vec3f".to_string() }

/// Initialization strategy for a field.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum InitDef {
    #[serde(rename = "zero")]
    Zero,
    #[serde(rename = "random")]
    Random {
        #[serde(default = "default_min")]
        min: Vec<f64>,
        #[serde(default = "default_max")]
        max: Vec<f64>,
    },
    #[serde(rename = "constant")]
    Constant { value: Vec<f64> },
    #[serde(rename = "grid")]
    Grid {
        #[serde(default = "default_grid_spacing")]
        spacing: f64,
        #[serde(default = "default_grid_origin")]
        origin: Vec<f64>,
    },
}

fn default_min() -> Vec<f64> { vec![-1.0, -1.0, -1.0] }
fn default_max() -> Vec<f64> { vec![1.0, 1.0, 1.0] }
fn default_grid_spacing() -> f64 { 0.1 }
fn default_grid_origin() -> Vec<f64> { vec![0.0, 0.0, 0.0] }

/// Force definition.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ForceDef {
    #[serde(rename = "gravity")]
    Gravity {
        #[serde(default = "default_gravity")]
        value: Vec<f64>,
    },
    #[serde(rename = "drag")]
    Drag {
        #[serde(default = "default_drag")]
        coefficient: f64,
    },
    #[serde(rename = "wind")]
    Wind {
        #[serde(default)]
        direction: Vec<f64>,
        #[serde(default = "default_wind_strength")]
        strength: f64,
    },
    #[serde(rename = "sph_density")]
    SphDensity {
        smoothing_radius: f64,
    },
    #[serde(rename = "sph_pressure")]
    SphPressure {
        gas_constant: f64,
        rest_density: f64,
    },
    #[serde(rename = "sph_viscosity")]
    SphViscosity {
        coefficient: f64,
    },
}

fn default_gravity() -> Vec<f64> { vec![0.0, -9.81, 0.0] }
fn default_drag() -> f64 { 0.01 }
fn default_wind_strength() -> f64 { 1.0 }

/// Constraint definition.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ConstraintDef {
    #[serde(rename = "ground_plane")]
    GroundPlane {
        #[serde(default)]
        y: f64,
        #[serde(default = "default_restitution")]
        restitution: f64,
    },
    #[serde(rename = "sphere")]
    Sphere {
        center: Vec<f64>,
        radius: f64,
        #[serde(default = "default_restitution")]
        restitution: f64,
    },
    #[serde(rename = "box")]
    Box {
        min: Vec<f64>,
        max: Vec<f64>,
        #[serde(default = "default_restitution")]
        restitution: f64,
    },
}

fn default_restitution() -> f64 { 0.5 }

/// Spring system configuration.
#[derive(Debug, Deserialize)]
pub struct SpringConfig {
    pub stiffness: f64,
    pub damping: f64,
    #[serde(default)]
    pub rest_length: Option<f64>,
    /// Spring connections as [[i, j], ...]
    #[serde(default)]
    pub connections: Vec<[u32; 2]>,
}

/// Spatial acceleration structure configuration.
#[derive(Debug, Deserialize)]
pub struct SpatialConfig {
    #[serde(default = "default_spatial_type")]
    pub r#type: String,
    pub cell_size: f64,
    #[serde(default = "default_grid_dims")]
    pub grid_dims: Vec<u32>,
}

fn default_spatial_type() -> String { "hashgrid".to_string() }
fn default_grid_dims() -> Vec<u32> { vec![32, 32, 32] }

/// Output configuration.
#[derive(Debug, Deserialize)]
pub struct OutputConfig {
    #[serde(default = "default_format")]
    pub format: String,
    #[serde(default = "default_fps")]
    pub fps: u32,
    #[serde(default)]
    pub path: Option<String>,
}

fn default_format() -> String { "json".to_string() }
fn default_fps() -> u32 { 60 }

impl SimManifest {
    /// Parse a TOML string into a manifest.
    pub fn from_toml(toml_str: &str) -> Result<Self, String> {
        toml::from_str(toml_str).map_err(|e| format!("TOML parse error: {}", e))
    }

    /// Parse from a file path.
    pub fn from_file(path: &str) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read '{}': {}", path, e))?;
        Self::from_toml(&content)
    }

    /// Get particle count (from simulation.count or first field's count).
    pub fn particle_count(&self) -> usize {
        if let Some(count) = self.simulation.count {
            return count;
        }
        for field in &self.fields {
            if let Some(count) = field.count {
                return count;
            }
        }
        1000 // default
    }
}
