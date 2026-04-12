//! Tests for HashGrid spatial queries.

use forge_core::Vec3f;
use forge_runtime::{Device, HashGrid};

#[test]
fn test_hashgrid_basic() {
    // 10x10x10 grid, cell size 1.0, origin at (0,0,0)
    let mut grid = HashGrid::new(
        1.0,
        (10, 10, 10),
        Vec3f::new(0.0, 0.0, 0.0),
        100,
        Device::Cuda(0),
    );

    // 4 particles in a cluster
    let positions = vec![
        Vec3f::new(5.0, 5.0, 5.0),  // 0
        Vec3f::new(5.1, 5.0, 5.0),  // 1 — very close to 0
        Vec3f::new(5.5, 5.5, 5.5),  // 2 — nearby
        Vec3f::new(9.0, 9.0, 9.0),  // 3 — far away
    ];

    forge_runtime::cuda::init();
    grid.build(&positions).unwrap();

    // Query neighbors of particle 0 within radius 1.0
    let neighbors = grid.query_neighbors(positions[0], 1.0, &positions);
    assert!(neighbors.contains(&0), "should find itself");
    assert!(neighbors.contains(&1), "should find particle 1 (dist=0.1)");
    assert!(neighbors.contains(&2), "should find particle 2 (dist≈0.87)");
    assert!(!neighbors.contains(&3), "should NOT find particle 3 (dist≈6.9)");

    eprintln!("✅ HashGrid: found {} neighbors for particle 0", neighbors.len());
}

#[test]
fn test_hashgrid_uniform_distribution() {
    forge_runtime::cuda::init();

    let mut grid = HashGrid::new(
        2.0,
        (5, 5, 5),
        Vec3f::new(0.0, 0.0, 0.0),
        1000,
        Device::Cuda(0),
    );

    // 1000 particles in a 10x10x10 box
    let n = 1000;
    let positions: Vec<Vec3f> = (0..n)
        .map(|i| {
            let x = (i % 10) as f32 + 0.5;
            let y = ((i / 10) % 10) as f32 + 0.5;
            let z = (i / 100) as f32 + 0.5;
            Vec3f::new(x, y, z)
        })
        .collect();

    grid.build(&positions).unwrap();

    // Query center point — should find 8 particles (2x2x2 block around center)
    let center = Vec3f::new(5.0, 5.0, 5.0);
    let neighbors = grid.query_neighbors(center, 1.5, &positions);

    assert!(
        neighbors.len() >= 4,
        "should find at least some particles near center, got {}",
        neighbors.len()
    );

    // Verify all returned neighbors are actually within radius
    for &idx in &neighbors {
        let p = positions[idx as usize];
        let dx = p.x - center.x;
        let dy = p.y - center.y;
        let dz = p.z - center.z;
        let dist = (dx*dx + dy*dy + dz*dz).sqrt();
        assert!(
            dist <= 1.5,
            "neighbor {} at dist {} should be <= 1.5",
            idx, dist
        );
    }

    eprintln!("✅ HashGrid 1000 particles: found {} neighbors near center", neighbors.len());
}
