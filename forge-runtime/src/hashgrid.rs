//! # HashGrid — Spatial hash grid for particle neighbor queries.
//!
//! A uniform grid that partitions space into cells of fixed size.
//! Each particle maps to a cell based on its position.
//! Neighbor queries find all particles in adjacent cells.
//!
//! ## GPU Design
//!
//! 1. **Build**: Hash each particle position → cell index, sort by cell
//! 2. **Query**: For a position, check the 3x3x3 neighborhood of cells
//!
//! Uses a counting sort approach:
//! - Pass 1: Count particles per cell
//! - Pass 2: Prefix sum for cell start indices
//! - Pass 3: Scatter particles into sorted order

use crate::{Array, Device, ForgeError};
use forge_core::Vec3f;

/// A spatial hash grid for fast neighbor queries.
pub struct HashGrid {
    /// Cell size (same in all dimensions)
    pub cell_size: f32,
    /// Grid dimensions
    pub dim_x: u32,
    pub dim_y: u32,
    pub dim_z: u32,
    /// Origin of the grid (lower corner)
    pub origin: Vec3f,
    /// Cell start indices (prefix sum). Length = num_cells + 1
    pub cell_start: Array<u32>,
    /// Particle indices sorted by cell. Length = num_particles
    pub sorted_indices: Array<u32>,
    /// Cell index for each particle. Length = num_particles
    pub cell_indices: Array<u32>,
    /// Number of particles
    pub num_particles: usize,
    /// Device
    device: Device,
}

impl HashGrid {
    /// Create a new hash grid.
    ///
    /// - `cell_size`: Width of each cell (should be >= query radius)
    /// - `grid_dims`: (nx, ny, nz) number of cells in each dimension
    /// - `origin`: Lower corner of the grid
    /// - `max_particles`: Maximum number of particles
    /// - `device`: GPU device
    pub fn new(
        cell_size: f32,
        grid_dims: (u32, u32, u32),
        origin: Vec3f,
        max_particles: usize,
        device: Device,
    ) -> Self {
        let (dim_x, dim_y, dim_z) = grid_dims;
        let num_cells = (dim_x * dim_y * dim_z) as usize;

        Self {
            cell_size,
            dim_x,
            dim_y,
            dim_z,
            origin,
            cell_start: Array::<u32>::zeros(num_cells + 1, device),
            sorted_indices: Array::<u32>::zeros(max_particles, device),
            cell_indices: Array::<u32>::zeros(max_particles, device),
            num_particles: 0,
            device,
        }
    }

    /// Number of cells in the grid.
    pub fn num_cells(&self) -> usize {
        (self.dim_x * self.dim_y * self.dim_z) as usize
    }

    /// Build the hash grid from particle positions (CPU implementation).
    ///
    /// This is a simple CPU build for correctness. GPU build can be added later.
    pub fn build(&mut self, positions: &[Vec3f]) -> Result<(), ForgeError> {
        let n = positions.len();
        self.num_particles = n;

        let num_cells = self.num_cells();

        // Step 1: Compute cell index for each particle
        let mut cell_idx = vec![0u32; n];
        for i in 0..n {
            cell_idx[i] = self.position_to_cell(positions[i]);
        }

        // Step 2: Count particles per cell
        let mut counts = vec![0u32; num_cells];
        for &c in &cell_idx {
            if (c as usize) < num_cells {
                counts[c as usize] += 1;
            }
        }

        // Step 3: Prefix sum → cell_start
        let mut starts = vec![0u32; num_cells + 1];
        for i in 0..num_cells {
            starts[i + 1] = starts[i] + counts[i];
        }

        // Step 4: Scatter particles into sorted order
        let mut offsets = starts[..num_cells].to_vec();
        let mut sorted = vec![0u32; n];
        for i in 0..n {
            let c = cell_idx[i] as usize;
            if c < num_cells {
                sorted[offsets[c] as usize] = i as u32;
                offsets[c] += 1;
            }
        }

        // Upload to GPU
        self.cell_start = Array::from_vec(starts, self.device);
        self.sorted_indices = Array::from_vec(sorted, self.device);
        self.cell_indices = Array::from_vec(cell_idx, self.device);

        Ok(())
    }

    /// Convert a position to a cell index.
    pub fn position_to_cell(&self, pos: Vec3f) -> u32 {
        let cx = ((pos.x - self.origin.x) / self.cell_size).floor() as i32;
        let cy = ((pos.y - self.origin.y) / self.cell_size).floor() as i32;
        let cz = ((pos.z - self.origin.z) / self.cell_size).floor() as i32;

        // Clamp to grid bounds
        let cx = cx.max(0).min(self.dim_x as i32 - 1) as u32;
        let cy = cy.max(0).min(self.dim_y as i32 - 1) as u32;
        let cz = cz.max(0).min(self.dim_z as i32 - 1) as u32;

        cx + cy * self.dim_x + cz * self.dim_x * self.dim_y
    }

    /// Query neighbors of a position within the given radius.
    ///
    /// Returns indices of particles within `radius` of `pos`.
    /// CPU implementation for correctness testing.
    pub fn query_neighbors(
        &self,
        pos: Vec3f,
        radius: f32,
        positions: &[Vec3f],
    ) -> Vec<u32> {
        let mut result = Vec::new();
        let radius_sq = radius * radius;

        // Check 3x3x3 neighborhood
        let cx = ((pos.x - self.origin.x) / self.cell_size).floor() as i32;
        let cy = ((pos.y - self.origin.y) / self.cell_size).floor() as i32;
        let cz = ((pos.z - self.origin.z) / self.cell_size).floor() as i32;

        let starts = self.cell_start.to_vec();
        let sorted = self.sorted_indices.to_vec();

        for dz in -1..=1i32 {
            for dy in -1..=1i32 {
                for dx in -1..=1i32 {
                    let nx = cx + dx;
                    let ny = cy + dy;
                    let nz = cz + dz;

                    if nx < 0 || ny < 0 || nz < 0 { continue; }
                    if nx >= self.dim_x as i32 || ny >= self.dim_y as i32 || nz >= self.dim_z as i32 { continue; }

                    let cell = nx as u32 + ny as u32 * self.dim_x + nz as u32 * self.dim_x * self.dim_y;
                    let start = starts[cell as usize] as usize;
                    let end = starts[cell as usize + 1] as usize;

                    for si in start..end {
                        let pi = sorted[si] as usize;
                        let diff = Vec3f::new(
                            positions[pi].x - pos.x,
                            positions[pi].y - pos.y,
                            positions[pi].z - pos.z,
                        );
                        let dist_sq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                        if dist_sq <= radius_sq {
                            result.push(pi as u32);
                        }
                    }
                }
            }
        }

        result
    }
}
