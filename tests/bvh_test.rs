//! Tests for BVH spatial data structure.

use forge_core::Vec3f;
use forge_runtime::Bvh;

#[test]
fn test_bvh_sphere_query() {
    let mut bvh = Bvh::new();

    let positions = vec![
        Vec3f::new(0.0, 0.0, 0.0),   // 0
        Vec3f::new(0.5, 0.0, 0.0),   // 1
        Vec3f::new(1.0, 0.0, 0.0),   // 2
        Vec3f::new(5.0, 5.0, 5.0),   // 3 — far away
        Vec3f::new(0.3, 0.3, 0.0),   // 4 — close to origin
    ];

    bvh.build_points(&positions);

    // Sphere query: radius 0.6 from origin
    let result = bvh.query_sphere(Vec3f::new(0.0, 0.0, 0.0), 0.6, &positions);
    assert!(result.contains(&0), "should find origin");
    assert!(result.contains(&1), "should find (0.5, 0, 0)");
    assert!(result.contains(&4), "should find (0.3, 0.3, 0)");
    assert!(!result.contains(&2), "should NOT find (1, 0, 0) — dist=1.0");
    assert!(!result.contains(&3), "should NOT find (5, 5, 5)");

    eprintln!("✅ BVH sphere query: found {} points within radius 0.6", result.len());
}

#[test]
fn test_bvh_closest_point() {
    let mut bvh = Bvh::new();

    let positions = vec![
        Vec3f::new(0.0, 0.0, 0.0),
        Vec3f::new(3.0, 0.0, 0.0),
        Vec3f::new(0.0, 4.0, 0.0),
        Vec3f::new(10.0, 10.0, 10.0),
    ];

    bvh.build_points(&positions);

    // Closest to (1, 0, 0) should be (0, 0, 0) at dist=1.0
    let (idx, d2) = bvh.closest_point(Vec3f::new(1.0, 0.0, 0.0), &positions).unwrap();
    assert_eq!(idx, 0);
    assert!((d2 - 1.0).abs() < 1e-6, "d² = {}, expected 1.0", d2);

    // Closest to (2.5, 0, 0) should be (3, 0, 0) at dist=0.5
    let (idx, d2) = bvh.closest_point(Vec3f::new(2.5, 0.0, 0.0), &positions).unwrap();
    assert_eq!(idx, 1);
    assert!((d2 - 0.25).abs() < 1e-6, "d² = {}, expected 0.25", d2);

    eprintln!("✅ BVH closest point: correct");
}

#[test]
fn test_bvh_ray_cast() {
    let mut bvh = Bvh::new();

    let positions = vec![
        Vec3f::new(5.0, 0.0, 0.0),   // 0 — on the x-axis
        Vec3f::new(10.0, 0.0, 0.0),  // 1 — further on x
        Vec3f::new(0.0, 5.0, 0.0),   // 2 — on y-axis, shouldn't be hit
    ];

    bvh.build_points(&positions);

    // Ray from origin along +x, sphere radius 0.5
    let result = bvh.ray_cast(
        Vec3f::new(0.0, 0.0, 0.0),
        Vec3f::new(1.0, 0.0, 0.0),
        100.0,
        &positions,
        0.5,
    );

    assert!(result.is_some(), "should hit a point");
    let (idx, t) = result.unwrap();
    assert_eq!(idx, 0, "should hit closest point (5, 0, 0)");
    assert!((t - 4.5).abs() < 1e-4, "t = {}, expected ~4.5", t);

    eprintln!("✅ BVH ray cast: hit point {} at t={:.2}", idx, t);
}

#[test]
fn test_bvh_large() {
    let mut bvh = Bvh::new();

    let n = 10000;
    let positions: Vec<Vec3f> = (0..n)
        .map(|i| {
            let x = (i % 100) as f32;
            let y = ((i / 100) % 100) as f32;
            let z = (i / 10000) as f32;
            Vec3f::new(x, y, z)
        })
        .collect();

    bvh.build_points(&positions);
    assert_eq!(bvh.num_primitives, n);

    // Sphere query at center
    let result = bvh.query_sphere(Vec3f::new(50.0, 50.0, 0.0), 2.0, &positions);
    assert!(result.len() > 0, "should find some points near center");

    // Verify all results are within radius
    for &idx in &result {
        let p = positions[idx as usize];
        let dx = p.x - 50.0;
        let dy = p.y - 50.0;
        let dz = p.z;
        let dist = (dx*dx + dy*dy + dz*dz).sqrt();
        assert!(dist <= 2.0, "point {} at dist {} should be <= 2.0", idx, dist);
    }

    // Closest point
    let (idx, d2) = bvh.closest_point(Vec3f::new(50.5, 50.5, 0.0), &positions).unwrap();
    let p = positions[idx as usize];
    assert!((p.x - 50.0).abs() <= 1.0 && (p.y - 50.0).abs() <= 1.0);

    eprintln!("✅ BVH 10K points: sphere found {}, closest at d²={:.4}", result.len(), d2);
}
