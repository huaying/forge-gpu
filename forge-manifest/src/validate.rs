//! Manifest validation.

use crate::schema::*;

/// Validation error.
#[derive(Debug)]
pub struct ValidationError {
    pub message: String,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// Validate a manifest for completeness and correctness.
pub fn validate(manifest: &SimManifest) -> Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();

    // Must have a name
    if manifest.simulation.name.is_empty() {
        errors.push(ValidationError {
            message: "simulation.name is required".to_string(),
        });
    }

    // dt must be positive
    if manifest.simulation.dt <= 0.0 {
        errors.push(ValidationError {
            message: format!("simulation.dt must be > 0, got {}", manifest.simulation.dt),
        });
    }

    // duration must be positive
    if manifest.simulation.duration <= 0.0 {
        errors.push(ValidationError {
            message: format!("simulation.duration must be > 0, got {}", manifest.simulation.duration),
        });
    }

    // Particle sim needs a count
    if manifest.simulation.r#type == "particles" && manifest.particle_count() == 0 {
        errors.push(ValidationError {
            message: "particle simulation needs count > 0".to_string(),
        });
    }

    // Validate forces
    for (i, force) in manifest.forces.iter().enumerate() {
        match force {
            ForceDef::Gravity { value } => {
                if value.len() != 3 {
                    errors.push(ValidationError {
                        message: format!("forces[{}].gravity.value must have 3 components", i),
                    });
                }
            }
            _ => {}
        }
    }

    // Validate constraints
    for (i, constraint) in manifest.constraints.iter().enumerate() {
        match constraint {
            ConstraintDef::Sphere { center, .. } => {
                if center.len() != 3 {
                    errors.push(ValidationError {
                        message: format!("constraints[{}].sphere.center must have 3 components", i),
                    });
                }
            }
            _ => {}
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
