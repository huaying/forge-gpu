//! Tests for Mesh spatial queries.

use forge_core::Vec3f;
use forge_runtime::Mesh;

fn make_floor_quad() -> Mesh {
    // Two triangles forming a 10x10 floor at y=0
    let vertices = vec![
        Vec3f::new(-5.0, 0.0, -5.0), // 0
        Vec3f::new( 5.0, 0.0, -5.0), // 1
        Vec3f::new( 5.0, 0.0,  5.0), // 2
        Vec3f::new(-5.0, 0.0,  5.0), // 3
    ];
    let indices = vec![[0, 1, 2], [0, 2, 3]];
    let mut mesh = Mesh::new(vertices, indices);
    mesh.build_bvh();
    mesh
}

#[test]
fn test_mesh_closest_point() {
    let mesh = make_floor_quad();

    // Point above the floor → closest should be directly below
    let (tri, cp, d2) = mesh.closest_point(Vec3f::new(0.0, 3.0, 0.0)).unwrap();
    assert!((cp.y - 0.0).abs() < 1e-4, "closest y = {}, expected 0", cp.y);
    assert!((cp.x - 0.0).abs() < 1e-4, "closest x = {}, expected 0", cp.x);
    assert!((d2 - 9.0).abs() < 1e-3, "d² = {}, expected 9.0", d2);

    eprintln!("✅ Mesh closest point: tri={}, point=({:.2}, {:.2}, {:.2}), d²={:.2}",
        tri, cp.x, cp.y, cp.z, d2);
}

#[test]
fn test_mesh_closest_point_edge() {
    let mesh = make_floor_quad();

    // Point outside the floor boundary → should snap to edge
    let (_, cp, _) = mesh.closest_point(Vec3f::new(10.0, 0.0, 0.0)).unwrap();
    assert!((cp.x - 5.0).abs() < 1e-3, "should snap to edge at x=5, got {}", cp.x);

    eprintln!("✅ Mesh closest point edge: ({:.2}, {:.2}, {:.2})", cp.x, cp.y, cp.z);
}

#[test]
fn test_mesh_ray_cast() {
    let mesh = make_floor_quad();

    // Ray pointing down from above → should hit floor
    let result = mesh.ray_cast(
        Vec3f::new(0.0, 5.0, 0.0),   // origin above
        Vec3f::new(0.0, -1.0, 0.0),  // direction down
        100.0,
    );
    assert!(result.is_some(), "should hit the floor");
    let (tri, t, hit) = result.unwrap();
    assert!((t - 5.0).abs() < 1e-4, "t = {}, expected 5.0", t);
    assert!((hit.y).abs() < 1e-4, "hit.y = {}, expected 0", hit.y);

    eprintln!("✅ Mesh ray cast: tri={}, t={:.2}, hit=({:.2}, {:.2}, {:.2})", tri, t, hit.x, hit.y, hit.z);
}

#[test]
fn test_mesh_ray_miss() {
    let mesh = make_floor_quad();

    // Ray parallel to floor → should miss
    let result = mesh.ray_cast(
        Vec3f::new(0.0, 5.0, 0.0),
        Vec3f::new(1.0, 0.0, 0.0), // horizontal
        100.0,
    );
    assert!(result.is_none(), "horizontal ray should miss floor");

    // Ray pointing up → should miss
    let result = mesh.ray_cast(
        Vec3f::new(0.0, 5.0, 0.0),
        Vec3f::new(0.0, 1.0, 0.0), // up
        100.0,
    );
    assert!(result.is_none(), "upward ray should miss floor");

    eprintln!("✅ Mesh ray miss: correct");
}

#[test]
fn test_mesh_cube() {
    // Build a simple cube mesh (12 triangles)
    let v = vec![
        Vec3f::new(0.0, 0.0, 0.0), Vec3f::new(1.0, 0.0, 0.0),
        Vec3f::new(1.0, 1.0, 0.0), Vec3f::new(0.0, 1.0, 0.0),
        Vec3f::new(0.0, 0.0, 1.0), Vec3f::new(1.0, 0.0, 1.0),
        Vec3f::new(1.0, 1.0, 1.0), Vec3f::new(0.0, 1.0, 1.0),
    ];
    let idx = vec![
        [0,1,2],[0,2,3], // front
        [4,6,5],[4,7,6], // back
        [0,4,5],[0,5,1], // bottom
        [2,6,7],[2,7,3], // top
        [0,3,7],[0,7,4], // left
        [1,5,6],[1,6,2], // right
    ];
    let mut mesh = Mesh::new(v, idx);
    mesh.build_bvh();
    assert_eq!(mesh.num_triangles(), 12);

    // Point inside cube → closest should be on surface
    let (_, cp, d2) = mesh.closest_point(Vec3f::new(0.5, 0.5, 0.5)).unwrap();
    let dist = d2.sqrt();
    assert!(dist < 0.51, "point inside cube should be close to surface, dist={}", dist);

    // Ray from outside hitting cube
    let result = mesh.ray_cast(
        Vec3f::new(-5.0, 0.5, 0.5),
        Vec3f::new(1.0, 0.0, 0.0),
        100.0,
    );
    assert!(result.is_some(), "ray should hit cube");
    let (_, t, _) = result.unwrap();
    assert!((t - 5.0).abs() < 1e-3, "t = {}, expected ~5.0", t);

    eprintln!("✅ Mesh cube: 12 triangles, closest dist={:.3}, ray t={:.2}", dist, t);
}
