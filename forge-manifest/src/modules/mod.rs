//! # Module system — the core abstraction for declarative simulations.
//!
//! Every simulation is a pipeline of modules operating on a shared FieldSet.

use forge_runtime::{Array, Device, ForgeError};
use forge_core::Vec3f;
use std::collections::HashMap;

pub mod builtins;

/// A named collection of GPU arrays that modules read/write.
pub struct FieldSet {
    pub f32_fields: HashMap<String, Array<f32>>,
    pub vec3_fields: HashMap<String, Array<Vec3f>>,
    pub i32_fields: HashMap<String, Array<i32>>,
    pub u32_fields: HashMap<String, Array<u32>>,
    /// Index pairs for springs/edges (stored as flat i32: [i0,j0, i1,j1, ...])
    pub index_pairs: HashMap<String, Vec<[u32; 2]>>,
    pub particle_count: usize,
    pub device: Device,
}

impl FieldSet {
    pub fn new(particle_count: usize, device: Device) -> Self {
        Self {
            f32_fields: HashMap::new(),
            vec3_fields: HashMap::new(),
            i32_fields: HashMap::new(),
            u32_fields: HashMap::new(),
            index_pairs: HashMap::new(),
            particle_count,
            device,
        }
    }

    pub fn add_f32(&mut self, name: &str, data: Vec<f32>) {
        self.f32_fields.insert(name.to_string(), Array::from_vec(data, self.device));
    }

    pub fn add_f32_zeros(&mut self, name: &str, len: usize) {
        self.f32_fields.insert(name.to_string(), Array::<f32>::zeros(len, self.device));
    }

    pub fn add_vec3(&mut self, name: &str, data: Vec<Vec3f>) {
        self.vec3_fields.insert(name.to_string(), Array::from_vec(data, self.device));
    }

    pub fn add_vec3_zeros(&mut self, name: &str, len: usize) {
        self.vec3_fields.insert(name.to_string(), Array::from_vec(vec![Vec3f::zero(); len], self.device));
    }

    pub fn get_f32(&self, name: &str) -> Option<&Array<f32>> {
        self.f32_fields.get(name)
    }

    pub fn get_f32_mut(&mut self, name: &str) -> Option<&mut Array<f32>> {
        self.f32_fields.get_mut(name)
    }

    pub fn get_vec3(&self, name: &str) -> Option<&Array<Vec3f>> {
        self.vec3_fields.get(name)
    }

    pub fn get_vec3_mut(&mut self, name: &str) -> Option<&mut Array<Vec3f>> {
        self.vec3_fields.get_mut(name)
    }
}

/// A simulation module — reads some fields, writes some fields.
pub trait SimModule: Send + Sync {
    fn name(&self) -> &str;
    fn execute(&self, fields: &mut FieldSet, dt: f32) -> Result<(), ForgeError>;
}

/// A pipeline of modules executed in order each timestep.
pub struct Pipeline {
    modules: Vec<Box<dyn SimModule>>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self { modules: Vec::new() }
    }

    pub fn add(&mut self, module: Box<dyn SimModule>) {
        self.modules.push(module);
    }

    /// Execute all modules in order.
    pub fn step(&self, fields: &mut FieldSet, dt: f32) -> Result<(), ForgeError> {
        for module in &self.modules {
            module.execute(fields, dt)?;
        }
        Ok(())
    }

    /// Run for N steps.
    pub fn run(&self, fields: &mut FieldSet, dt: f32, steps: usize) -> Result<(), ForgeError> {
        for _ in 0..steps {
            self.step(fields, dt)?;
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.modules.len()
    }
}
