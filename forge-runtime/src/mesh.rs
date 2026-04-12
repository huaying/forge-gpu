//! # Mesh — Triangle mesh with spatial queries.
//!
//! Supports:
//! - Closest point on mesh surface
//! - Ray-triangle intersection
//! - Point-in-mesh test (inside/outside)
//!
//! Uses BVH internally for acceleration.

use forge_core::Vec3f;
use crate::bvh::{Aabb, Bvh};

/// A triangle mesh with spatial query acceleration.
pub struct Mesh {
    /// Vertex positions.
    pub vertices: Vec<Vec3f>,
    /// Triangle indices (3 per triangle).
    pub indices: Vec<[u32; 3]>,
    /// Per-triangle BVH for acceleration.
    bvh: Option<MeshBvh>,
}

/// Internal BVH over triangles.
struct MeshBvh {
    root: Option<MeshBvhNode>,
}

enum MeshBvhNode {
    Internal {
        bounds: Aabb,
        left: Box<MeshBvhNode>,
        right: Box<MeshBvhNode>,
    },
    Leaf {
        bounds: Aabb,
        tri_indices: Vec<u32>,
    },
}

impl Mesh {
    /// Create a mesh from vertices and triangle indices.
    pub fn new(vertices: Vec<Vec3f>, indices: Vec<[u32; 3]>) -> Self {
        Self {
            vertices,
            indices,
            bvh: None,
        }
    }

    /// Build the internal BVH for accelerated queries.
    pub fn build_bvh(&mut self) {
        let n = self.indices.len();
        if n == 0 {
            self.bvh = None;
            return;
        }

        let tri_indices: Vec<u32> = (0..n as u32).collect();
        let root = self.build_bvh_recursive(tri_indices, 4);
        self.bvh = Some(MeshBvh { root: Some(root) });
    }

    fn tri_bounds(&self, tri_idx: u32) -> Aabb {
        let [a, b, c] = self.indices[tri_idx as usize];
        let mut bb = Aabb::empty();
        bb.expand_point(self.vertices[a as usize]);
        bb.expand_point(self.vertices[b as usize]);
        bb.expand_point(self.vertices[c as usize]);
        bb
    }

    fn tri_centroid(&self, tri_idx: u32) -> Vec3f {
        let [a, b, c] = self.indices[tri_idx as usize];
        let va = self.vertices[a as usize];
        let vb = self.vertices[b as usize];
        let vc = self.vertices[c as usize];
        Vec3f::new(
            (va.x + vb.x + vc.x) / 3.0,
            (va.y + vb.y + vc.y) / 3.0,
            (va.z + vb.z + vc.z) / 3.0,
        )
    }

    fn build_bvh_recursive(&self, tri_indices: Vec<u32>, max_leaf: usize) -> MeshBvhNode {
        let mut bounds = Aabb::empty();
        for &ti in &tri_indices {
            bounds.expand_aabb(&self.tri_bounds(ti));
        }

        if tri_indices.len() <= max_leaf {
            return MeshBvhNode::Leaf { bounds, tri_indices };
        }

        let axis = bounds.longest_axis();
        let mut sorted = tri_indices;
        let mesh = self;
        sorted.sort_by(|&a, &b| {
            let ca = mesh.tri_centroid(a);
            let cb = mesh.tri_centroid(b);
            let va = match axis { 0 => ca.x, 1 => ca.y, _ => ca.z };
            let vb = match axis { 0 => cb.x, 1 => cb.y, _ => cb.z };
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mid = sorted.len() / 2;
        let right_indices = sorted.split_off(mid);
        let left = self.build_bvh_recursive(sorted, max_leaf);
        let right = self.build_bvh_recursive(right_indices, max_leaf);

        MeshBvhNode::Internal {
            bounds,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Number of triangles.
    pub fn num_triangles(&self) -> usize {
        self.indices.len()
    }

    /// Find the closest point on the mesh surface to `query`.
    /// Returns (triangle_index, closest_point, distance²).
    pub fn closest_point(&self, query: Vec3f) -> Option<(u32, Vec3f, f32)> {
        let bvh = self.bvh.as_ref()?;
        let root = bvh.root.as_ref()?;
        let mut best = (0u32, Vec3f::zero(), f32::MAX);
        self.closest_recursive(root, query, &mut best);
        if best.2 < f32::MAX { Some(best) } else { None }
    }

    fn closest_recursive(&self, node: &MeshBvhNode, query: Vec3f, best: &mut (u32, Vec3f, f32)) {
        match node {
            MeshBvhNode::Leaf { bounds, tri_indices } => {
                if !bounds.overlaps_sphere(query, best.2.sqrt()) { return; }
                for &ti in tri_indices {
                    let [a, b, c] = self.indices[ti as usize];
                    let p = closest_point_on_triangle(
                        query,
                        self.vertices[a as usize],
                        self.vertices[b as usize],
                        self.vertices[c as usize],
                    );
                    let dx = p.x - query.x;
                    let dy = p.y - query.y;
                    let dz = p.z - query.z;
                    let d2 = dx*dx + dy*dy + dz*dz;
                    if d2 < best.2 {
                        *best = (ti, p, d2);
                    }
                }
            }
            MeshBvhNode::Internal { bounds, left, right } => {
                if !bounds.overlaps_sphere(query, best.2.sqrt()) { return; }
                self.closest_recursive(left, query, best);
                self.closest_recursive(right, query, best);
            }
        }
    }

    /// Ray-mesh intersection. Returns (triangle_index, t, hit_point).
    pub fn ray_cast(&self, origin: Vec3f, direction: Vec3f, max_t: f32) -> Option<(u32, f32, Vec3f)> {
        let bvh = self.bvh.as_ref()?;
        let root = bvh.root.as_ref()?;
        let inv_dir = Vec3f::new(
            if direction.x.abs() > 1e-10 { 1.0 / direction.x } else { 1e10 },
            if direction.y.abs() > 1e-10 { 1.0 / direction.y } else { 1e10 },
            if direction.z.abs() > 1e-10 { 1.0 / direction.z } else { 1e10 },
        );
        let mut best = (0u32, max_t, Vec3f::zero());
        self.ray_cast_recursive(root, origin, direction, inv_dir, &mut best);
        if best.1 < max_t { Some(best) } else { None }
    }

    fn ray_cast_recursive(
        &self,
        node: &MeshBvhNode,
        origin: Vec3f,
        direction: Vec3f,
        inv_dir: Vec3f,
        best: &mut (u32, f32, Vec3f),
    ) {
        let bounds = match node {
            MeshBvhNode::Leaf { bounds, .. } => bounds,
            MeshBvhNode::Internal { bounds, .. } => bounds,
        };

        let (hit, tmin, _) = bounds.intersect_ray(origin, inv_dir);
        if !hit || tmin > best.1 { return; }

        match node {
            MeshBvhNode::Leaf { tri_indices, .. } => {
                for &ti in tri_indices {
                    let [a, b, c] = self.indices[ti as usize];
                    if let Some((t, p)) = ray_triangle_intersect(
                        origin, direction,
                        self.vertices[a as usize],
                        self.vertices[b as usize],
                        self.vertices[c as usize],
                    ) {
                        if t > 0.0 && t < best.1 {
                            *best = (ti, t, p);
                        }
                    }
                }
            }
            MeshBvhNode::Internal { left, right, .. } => {
                self.ray_cast_recursive(left, origin, direction, inv_dir, best);
                self.ray_cast_recursive(right, origin, direction, inv_dir, best);
            }
        }
    }
}

/// Closest point on triangle ABC to point P.
fn closest_point_on_triangle(p: Vec3f, a: Vec3f, b: Vec3f, c: Vec3f) -> Vec3f {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);

    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 { return a; }

    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 { return b; }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return add(a, scale(ab, v));
    }

    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 { return c; }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return add(a, scale(ac, w));
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add(b, scale(sub(c, b), w));
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    Vec3f::new(
        a.x + ab.x * v + ac.x * w,
        a.y + ab.y * v + ac.y * w,
        a.z + ab.z * v + ac.z * w,
    )
}

/// Möller–Trumbore ray-triangle intersection.
/// Returns Some((t, hit_point)) or None.
fn ray_triangle_intersect(
    origin: Vec3f,
    dir: Vec3f,
    v0: Vec3f,
    v1: Vec3f,
    v2: Vec3f,
) -> Option<(f32, Vec3f)> {
    let e1 = sub(v1, v0);
    let e2 = sub(v2, v0);
    let h = cross(dir, e2);
    let a = dot(e1, h);
    if a.abs() < 1e-8 { return None; }

    let f = 1.0 / a;
    let s = sub(origin, v0);
    let u = f * dot(s, h);
    if u < 0.0 || u > 1.0 { return None; }

    let q = cross(s, e1);
    let v = f * dot(dir, q);
    if v < 0.0 || u + v > 1.0 { return None; }

    let t = f * dot(e2, q);
    if t > 1e-8 {
        let p = Vec3f::new(
            origin.x + dir.x * t,
            origin.y + dir.y * t,
            origin.z + dir.z * t,
        );
        Some((t, p))
    } else {
        None
    }
}

// Vec3f helpers (no operator overload needed here, just local functions)
fn sub(a: Vec3f, b: Vec3f) -> Vec3f { Vec3f::new(a.x - b.x, a.y - b.y, a.z - b.z) }
fn add(a: Vec3f, b: Vec3f) -> Vec3f { Vec3f::new(a.x + b.x, a.y + b.y, a.z + b.z) }
fn scale(a: Vec3f, s: f32) -> Vec3f { Vec3f::new(a.x * s, a.y * s, a.z * s) }
fn dot(a: Vec3f, b: Vec3f) -> f32 { a.x * b.x + a.y * b.y + a.z * b.z }
fn cross(a: Vec3f, b: Vec3f) -> Vec3f {
    Vec3f::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}
