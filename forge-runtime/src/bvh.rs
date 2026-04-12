//! # BVH — Bounding Volume Hierarchy for spatial queries.
//!
//! A binary tree where each node has an axis-aligned bounding box (AABB).
//! Leaf nodes contain primitives (points, triangles).
//! Used for fast ray casting, closest point, and overlap queries.
//!
//! ## Build Strategy
//!
//! Uses a top-down median split approach:
//! 1. Compute AABB of all primitives
//! 2. Split along the longest axis at the median
//! 3. Recurse on each half
//!
//! This gives O(n log n) build, O(log n) query.

use forge_core::Vec3f;

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vec3f,
    pub max: Vec3f,
}

impl Aabb {
    /// Create an empty (inverted) AABB.
    pub fn empty() -> Self {
        Self {
            min: Vec3f::new(f32::MAX, f32::MAX, f32::MAX),
            max: Vec3f::new(f32::MIN, f32::MIN, f32::MIN),
        }
    }

    /// Expand AABB to include a point.
    pub fn expand_point(&mut self, p: Vec3f) {
        self.min.x = self.min.x.min(p.x);
        self.min.y = self.min.y.min(p.y);
        self.min.z = self.min.z.min(p.z);
        self.max.x = self.max.x.max(p.x);
        self.max.y = self.max.y.max(p.y);
        self.max.z = self.max.z.max(p.z);
    }

    /// Expand AABB to include another AABB.
    pub fn expand_aabb(&mut self, other: &Aabb) {
        self.min.x = self.min.x.min(other.min.x);
        self.min.y = self.min.y.min(other.min.y);
        self.min.z = self.min.z.min(other.min.z);
        self.max.x = self.max.x.max(other.max.x);
        self.max.y = self.max.y.max(other.max.y);
        self.max.z = self.max.z.max(other.max.z);
    }

    /// Center of the AABB.
    pub fn center(&self) -> Vec3f {
        Vec3f::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5,
        )
    }

    /// Size along each axis.
    pub fn extents(&self) -> Vec3f {
        Vec3f::new(
            self.max.x - self.min.x,
            self.max.y - self.min.y,
            self.max.z - self.min.z,
        )
    }

    /// Longest axis (0=x, 1=y, 2=z).
    pub fn longest_axis(&self) -> usize {
        let e = self.extents();
        if e.x >= e.y && e.x >= e.z { 0 }
        else if e.y >= e.z { 1 }
        else { 2 }
    }

    /// Test if a point is inside the AABB (with optional padding).
    pub fn contains(&self, p: Vec3f, padding: f32) -> bool {
        p.x >= self.min.x - padding && p.x <= self.max.x + padding &&
        p.y >= self.min.y - padding && p.y <= self.max.y + padding &&
        p.z >= self.min.z - padding && p.z <= self.max.z + padding
    }

    /// Test if a sphere overlaps the AABB.
    pub fn overlaps_sphere(&self, center: Vec3f, radius: f32) -> bool {
        // Find closest point on AABB to sphere center
        let cx = center.x.max(self.min.x).min(self.max.x);
        let cy = center.y.max(self.min.y).min(self.max.y);
        let cz = center.z.max(self.min.z).min(self.max.z);
        let dx = cx - center.x;
        let dy = cy - center.y;
        let dz = cz - center.z;
        dx * dx + dy * dy + dz * dz <= radius * radius
    }

    /// Test ray-AABB intersection. Returns (hit, t_min, t_max).
    pub fn intersect_ray(&self, origin: Vec3f, inv_dir: Vec3f) -> (bool, f32, f32) {
        let t1 = (self.min.x - origin.x) * inv_dir.x;
        let t2 = (self.max.x - origin.x) * inv_dir.x;
        let t3 = (self.min.y - origin.y) * inv_dir.y;
        let t4 = (self.max.y - origin.y) * inv_dir.y;
        let t5 = (self.min.z - origin.z) * inv_dir.z;
        let t6 = (self.max.z - origin.z) * inv_dir.z;

        let tmin = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
        let tmax = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));

        (tmax >= tmin.max(0.0), tmin, tmax)
    }
}

/// A BVH node. Can be internal (two children) or leaf (primitives).
#[derive(Debug)]
enum BvhNode {
    Internal {
        bounds: Aabb,
        left: Box<BvhNode>,
        right: Box<BvhNode>,
    },
    Leaf {
        bounds: Aabb,
        /// Indices into the original primitive array.
        indices: Vec<u32>,
    },
}

/// Bounding Volume Hierarchy for point/triangle collections.
pub struct Bvh {
    root: Option<BvhNode>,
    /// Total number of primitives.
    pub num_primitives: usize,
}

impl Bvh {
    /// Create an empty BVH.
    pub fn new() -> Self {
        Self {
            root: None,
            num_primitives: 0,
        }
    }

    /// Build a BVH from point positions.
    ///
    /// Each point becomes a leaf primitive. The BVH can then be used
    /// for sphere overlap queries, closest point, or ray cast.
    pub fn build_points(&mut self, positions: &[Vec3f]) {
        self.num_primitives = positions.len();
        if positions.is_empty() {
            self.root = None;
            return;
        }

        let indices: Vec<u32> = (0..positions.len() as u32).collect();
        self.root = Some(Self::build_recursive(positions, indices, 4));
    }

    fn build_recursive(positions: &[Vec3f], indices: Vec<u32>, max_leaf: usize) -> BvhNode {
        // Compute bounds
        let mut bounds = Aabb::empty();
        for &i in &indices {
            bounds.expand_point(positions[i as usize]);
        }

        // Leaf condition
        if indices.len() <= max_leaf {
            return BvhNode::Leaf { bounds, indices };
        }

        // Split along longest axis at median
        let axis = bounds.longest_axis();
        let mut sorted = indices.clone();
        sorted.sort_by(|&a, &b| {
            let pa = positions[a as usize];
            let pb = positions[b as usize];
            let va = match axis { 0 => pa.x, 1 => pa.y, _ => pa.z };
            let vb = match axis { 0 => pb.x, 1 => pb.y, _ => pb.z };
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mid = sorted.len() / 2;
        let left_indices = sorted[..mid].to_vec();
        let right_indices = sorted[mid..].to_vec();

        let left = Self::build_recursive(positions, left_indices, max_leaf);
        let right = Self::build_recursive(positions, right_indices, max_leaf);

        BvhNode::Internal {
            bounds,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Query all points within `radius` of `center`.
    pub fn query_sphere(&self, center: Vec3f, radius: f32, positions: &[Vec3f]) -> Vec<u32> {
        let mut result = Vec::new();
        if let Some(ref root) = self.root {
            Self::query_sphere_recursive(root, center, radius, positions, &mut result);
        }
        result
    }

    fn query_sphere_recursive(
        node: &BvhNode,
        center: Vec3f,
        radius: f32,
        positions: &[Vec3f],
        result: &mut Vec<u32>,
    ) {
        match node {
            BvhNode::Leaf { bounds, indices } => {
                if !bounds.overlaps_sphere(center, radius) { return; }
                let r2 = radius * radius;
                for &i in indices {
                    let p = positions[i as usize];
                    let dx = p.x - center.x;
                    let dy = p.y - center.y;
                    let dz = p.z - center.z;
                    if dx*dx + dy*dy + dz*dz <= r2 {
                        result.push(i);
                    }
                }
            }
            BvhNode::Internal { bounds, left, right } => {
                if !bounds.overlaps_sphere(center, radius) { return; }
                Self::query_sphere_recursive(left, center, radius, positions, result);
                Self::query_sphere_recursive(right, center, radius, positions, result);
            }
        }
    }

    /// Find the closest point to `query` and return (index, distance²).
    pub fn closest_point(&self, query: Vec3f, positions: &[Vec3f]) -> Option<(u32, f32)> {
        if self.root.is_none() { return None; }
        let mut best = (0u32, f32::MAX);
        Self::closest_recursive(self.root.as_ref().unwrap(), query, positions, &mut best);
        if best.1 < f32::MAX { Some(best) } else { None }
    }

    fn closest_recursive(
        node: &BvhNode,
        query: Vec3f,
        positions: &[Vec3f],
        best: &mut (u32, f32),
    ) {
        match node {
            BvhNode::Leaf { bounds, indices } => {
                // Early out: if closest possible point on AABB is farther than best, skip
                if !bounds.overlaps_sphere(query, best.1.sqrt()) { return; }
                for &i in indices {
                    let p = positions[i as usize];
                    let dx = p.x - query.x;
                    let dy = p.y - query.y;
                    let dz = p.z - query.z;
                    let d2 = dx*dx + dy*dy + dz*dz;
                    if d2 < best.1 {
                        *best = (i, d2);
                    }
                }
            }
            BvhNode::Internal { bounds, left, right } => {
                if !bounds.overlaps_sphere(query, best.1.sqrt()) { return; }
                // Visit closer child first for better pruning
                Self::closest_recursive(left, query, positions, best);
                Self::closest_recursive(right, query, positions, best);
            }
        }
    }

    /// Ray cast: find the closest point hit by a ray.
    /// Returns (index, t) where t is the distance along the ray.
    pub fn ray_cast(
        &self,
        origin: Vec3f,
        direction: Vec3f,
        max_t: f32,
        positions: &[Vec3f],
        radius: f32,
    ) -> Option<(u32, f32)> {
        if self.root.is_none() { return None; }
        let inv_dir = Vec3f::new(
            if direction.x.abs() > 1e-10 { 1.0 / direction.x } else { 1e10 },
            if direction.y.abs() > 1e-10 { 1.0 / direction.y } else { 1e10 },
            if direction.z.abs() > 1e-10 { 1.0 / direction.z } else { 1e10 },
        );
        let mut best = (0u32, max_t);
        Self::ray_cast_recursive(
            self.root.as_ref().unwrap(),
            origin, direction, inv_dir, radius, positions, &mut best,
        );
        if best.1 < max_t { Some(best) } else { None }
    }

    fn ray_cast_recursive(
        node: &BvhNode,
        origin: Vec3f,
        direction: Vec3f,
        inv_dir: Vec3f,
        radius: f32,
        positions: &[Vec3f],
        best: &mut (u32, f32),
    ) {
        let bounds = match node {
            BvhNode::Leaf { bounds, .. } => bounds,
            BvhNode::Internal { bounds, .. } => bounds,
        };

        // Expand AABB by radius for sphere-sweep test
        let expanded = Aabb {
            min: Vec3f::new(bounds.min.x - radius, bounds.min.y - radius, bounds.min.z - radius),
            max: Vec3f::new(bounds.max.x + radius, bounds.max.y + radius, bounds.max.z + radius),
        };
        let (hit, tmin, _) = expanded.intersect_ray(origin, inv_dir);
        if !hit || tmin > best.1 { return; }

        match node {
            BvhNode::Leaf { indices, .. } => {
                for &i in indices {
                    let p = positions[i as usize];
                    // Point-ray distance test (sphere around point)
                    let oc = Vec3f::new(origin.x - p.x, origin.y - p.y, origin.z - p.z);
                    let b = oc.x * direction.x + oc.y * direction.y + oc.z * direction.z;
                    let c = oc.x*oc.x + oc.y*oc.y + oc.z*oc.z - radius*radius;
                    let disc = b*b - c;
                    if disc >= 0.0 {
                        let t = -b - disc.sqrt();
                        if t > 0.0 && t < best.1 {
                            *best = (i, t);
                        }
                    }
                }
            }
            BvhNode::Internal { left, right, .. } => {
                Self::ray_cast_recursive(left, origin, direction, inv_dir, radius, positions, best);
                Self::ray_cast_recursive(right, origin, direction, inv_dir, radius, positions, best);
            }
        }
    }

    /// Get the root bounding box.
    pub fn bounds(&self) -> Option<Aabb> {
        match &self.root {
            Some(BvhNode::Internal { bounds, .. }) => Some(*bounds),
            Some(BvhNode::Leaf { bounds, .. }) => Some(*bounds),
            None => None,
        }
    }
}

impl Default for Bvh {
    fn default() -> Self {
        Self::new()
    }
}
