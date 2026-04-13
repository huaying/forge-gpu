//! Tests for #[forge_struct] proc macro.

use forge_macros::forge_struct;

#[forge_struct]
#[derive(Clone, Copy)]
pub struct Particle {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub mass: f32,
}

#[forge_struct]
#[derive(Clone, Copy)]
pub struct Spring {
    pub rest_length: f32,
    pub stiffness: f32,
    pub idx_a: i32,
    pub idx_b: i32,
}

#[test]
fn test_particle_struct_cuda_def() {
    let cuda = Particle_forge_meta::CUDA_STRUCT_DEF;
    assert!(cuda.contains("struct Particle"));
    assert!(cuda.contains("float x;"));
    assert!(cuda.contains("float y;"));
    assert!(cuda.contains("float z;"));
    assert!(cuda.contains("float mass;"));
    // Should have operator overloads (all same type f32)
    assert!(cuda.contains("operator+"));
    assert!(cuda.contains("operator-"));
    assert!(cuda.contains("operator*"));
    println!("Particle CUDA def:\n{}", cuda);
}

#[test]
fn test_spring_struct_cuda_def() {
    let cuda = Spring_forge_meta::CUDA_STRUCT_DEF;
    assert!(cuda.contains("struct Spring"));
    assert!(cuda.contains("float rest_length;"));
    assert!(cuda.contains("float stiffness;"));
    assert!(cuda.contains("int idx_a;"));
    assert!(cuda.contains("int idx_b;"));
    // Mixed types → no operator overloads
    assert!(!cuda.contains("operator+"));
    println!("Spring CUDA def:\n{}", cuda);
}

#[test]
fn test_struct_meta_name() {
    assert_eq!(Particle_forge_meta::STRUCT_NAME, "Particle");
    assert_eq!(Spring_forge_meta::STRUCT_NAME, "Spring");
}
