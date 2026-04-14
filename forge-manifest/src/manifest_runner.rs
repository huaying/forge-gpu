//! Translate a SimManifest into a module Pipeline and execute it.

use crate::schema::*;
use crate::modules::{FieldSet, Pipeline, SimModule};
use crate::modules::builtins::*;
use forge_core::Vec3f;
use forge_runtime::{cuda, Device, ForgeError};
use rand::Rng;

/// Build and run a simulation from a manifest using the module system.
pub fn run_manifest(manifest: &SimManifest) -> Result<crate::SimResult, String> {
    crate::validate::validate(manifest)
        .map_err(|errs| errs.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("; "))?;

    cuda::init();
    if cuda::device_count() == 0 {
        return Err("No GPU".into());
    }

    let device = Device::Cuda(0);
    let n = manifest.particle_count();
    let dt = manifest.simulation.dt as f32;
    let substeps = manifest.simulation.substeps;
    let duration = manifest.simulation.duration as f32;
    let total_steps = ((duration / dt) as usize) / substeps.max(1) as usize;

    // Build fields
    let mut fields = FieldSet::new(n, device);
    init_fields(&mut fields, manifest);

    // Build pipeline from manifest
    let pipeline = build_pipeline(manifest)?;

    println!("  Pipeline: {} modules", pipeline.len());
    println!("  Fields: {} f32, {} vec3",
        fields.f32_fields.len(), fields.vec3_fields.len());
    println!();

    // Run
    let start = std::time::Instant::now();
    let mut step_count = 0;

    // Step 0: warm-up (compiles all kernels via OnceLock, allocates GPU buffers)
    for _sub in 0..substeps {
        pipeline.step(&mut fields, dt)
            .map_err(|e| format!("Warm-up step {}: {}", step_count, e))?;
        step_count += 1;
    }
    // Ensure all GPU work is done before capture
    let stream = cuda::default_stream(0);
    stream.synchronize().map_err(|e| format!("Pre-capture sync: {:?}", e))?;

    // Capture CUDA Graph from one step
    // Modules all use default_stream(0) which is our non-default simulation stream
    let stream = cuda::default_stream(0);

    stream.begin_capture(cuda::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .map_err(|e| format!("Graph begin_capture: {:?}", e))?;

    for _sub in 0..substeps {
        pipeline.step_async(&mut fields, dt)
            .map_err(|e| format!("Graph capture step: {}", e))?;
    }

    let graph = stream.end_capture(cuda::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .map_err(|e| format!("Graph end_capture: {:?}", e))?
        .ok_or_else(|| "Graph capture returned None".to_string())?;

    graph.upload().map_err(|e| format!("Graph upload: {:?}", e))?;

    eprintln!("  📊 CUDA Graph captured ({} substeps/launch)", substeps);

    // Replay graph for remaining steps
    for _step in 1..total_steps {
        graph.launch().map_err(|e| format!("Graph launch: {:?}", e))?;
        step_count += substeps as usize;
    }
    stream.synchronize().map_err(|e| format!("Final sync: {:?}", e))?;

    let elapsed = start.elapsed();

    // Gather results
    let px = fields.f32_fields.get("pos_x").map(|a| a.to_vec()).unwrap_or_default();
    let py = fields.f32_fields.get("pos_y").map(|a| a.to_vec()).unwrap_or_default();
    let pz = fields.f32_fields.get("pos_z").map(|a| a.to_vec()).unwrap_or_default();

    let mut positions = Vec::with_capacity(n * 3);
    for i in 0..n {
        positions.push(px.get(i).copied().unwrap_or(0.0));
        positions.push(py.get(i).copied().unwrap_or(0.0));
        positions.push(pz.get(i).copied().unwrap_or(0.0));
    }

    Ok(crate::SimResult {
        positions,
        count: n,
        steps: step_count,
        elapsed_secs: elapsed.as_secs_f64(),
        frames: vec![],
    })
}

/// Run a simulation with a frame callback for streaming.
/// The callback receives (frame_number, &[f32]) where the f32 slice is interleaved x,y,z.
/// `frame_interval` controls how often frames are emitted (every N substep groups).
pub fn run_manifest_streaming<F>(
    manifest: &SimManifest,
    frame_interval: usize,
    mut on_frame: F,
) -> Result<crate::SimResult, String>
where
    F: FnMut(u32, &[f32]),
{
    crate::validate::validate(manifest)
        .map_err(|errs| errs.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("; "))?;

    cuda::init();
    if cuda::device_count() == 0 {
        return Err("No GPU".into());
    }

    let device = Device::Cuda(0);
    let n = manifest.particle_count();
    let dt = manifest.simulation.dt as f32;
    let substeps = manifest.simulation.substeps;
    let duration = manifest.simulation.duration as f32;
    let total_steps = ((duration / dt) as usize) / substeps.max(1) as usize;

    let mut fields = FieldSet::new(n, device);
    init_fields(&mut fields, manifest);

    let pipeline = build_pipeline(manifest)?;

    println!("  Pipeline: {} modules", pipeline.len());
    println!("  Fields: {} f32, {} vec3",
        fields.f32_fields.len(), fields.vec3_fields.len());
    println!("  Streaming: frame every {} steps", frame_interval);
    println!();

    let start = std::time::Instant::now();
    let mut step_count = 0usize;
    let mut frame_num = 0u32;
    let interval = frame_interval.max(1);

    for step in 0..total_steps {
        for _sub in 0..substeps {
            pipeline.step(&mut fields, dt)
                .map_err(|e| format!("Step {}: {}", step_count, e))?;
            step_count += 1;
        }

        // Emit frame at interval
        if step % interval == 0 {
            let px = fields.f32_fields.get("pos_x").map(|a| a.to_vec()).unwrap_or_default();
            let py = fields.f32_fields.get("pos_y").map(|a| a.to_vec()).unwrap_or_default();
            let pz = fields.f32_fields.get("pos_z").map(|a| a.to_vec()).unwrap_or_default();

            let mut positions = Vec::with_capacity(n * 3);
            for i in 0..n {
                positions.push(px.get(i).copied().unwrap_or(0.0));
                positions.push(py.get(i).copied().unwrap_or(0.0));
                positions.push(pz.get(i).copied().unwrap_or(0.0));
            }

            on_frame(frame_num, &positions);
            frame_num += 1;
        }
    }

    let elapsed = start.elapsed();

    // Final positions
    let px = fields.f32_fields.get("pos_x").map(|a| a.to_vec()).unwrap_or_default();
    let py = fields.f32_fields.get("pos_y").map(|a| a.to_vec()).unwrap_or_default();
    let pz = fields.f32_fields.get("pos_z").map(|a| a.to_vec()).unwrap_or_default();
    let mut positions = Vec::with_capacity(n * 3);
    for i in 0..n {
        positions.push(px.get(i).copied().unwrap_or(0.0));
        positions.push(py.get(i).copied().unwrap_or(0.0));
        positions.push(pz.get(i).copied().unwrap_or(0.0));
    }

    Ok(crate::SimResult {
        positions,
        count: n,
        steps: step_count,
        elapsed_secs: elapsed.as_secs_f64(),
        frames: vec![],
    })
}

pub fn init_fields(fields: &mut FieldSet, manifest: &SimManifest) {
    let n = manifest.particle_count();

    // Initialize from field definitions
    for field_def in &manifest.fields {
        // Map common field names to standard internal names
        let base_name = match field_def.name.as_str() {
            "position" | "pos" => "pos",
            "velocity" | "vel" => "vel",
            other => other,
        };

        match field_def.r#type.as_str() {
            "vec3f" => {
                let data = init_vec3(n, &field_def.init);
                let px: Vec<f32> = data.iter().map(|v| v.x).collect();
                let py: Vec<f32> = data.iter().map(|v| v.y).collect();
                let pz: Vec<f32> = data.iter().map(|v| v.z).collect();
                fields.add_f32(&format!("{}_x", base_name), px);
                fields.add_f32(&format!("{}_y", base_name), py);
                fields.add_f32(&format!("{}_z", base_name), pz);
            }
            "f32" => {
                let data = init_f32(n, &field_def.init);
                fields.add_f32(&field_def.name, data);
            }
            _ => {}
        }
    }

    // Ensure vel exists
    if !fields.f32_fields.contains_key("vel_x") {
        fields.add_f32_zeros("vel_x", n);
        fields.add_f32_zeros("vel_y", n);
        fields.add_f32_zeros("vel_z", n);
    }
    // Ensure pos exists
    if !fields.f32_fields.contains_key("pos_x") {
        fields.add_f32_zeros("pos_x", n);
        fields.add_f32_zeros("pos_y", n);
        fields.add_f32_zeros("pos_z", n);
    }

    // Generate spring topology if defined
    if let Some(ref springs) = manifest.springs {
        if !springs.connections.is_empty() {
            fields.index_pairs.insert("springs".to_string(), springs.connections.clone());
        } else {
            // Auto-generate grid springs
            let side = (n as f32).sqrt().round() as usize;
            let mut pairs = Vec::new();
            for y in 0..side {
                for x in 0..side {
                    let idx = y * side + x;
                    // Structural: horizontal
                    if x + 1 < side {
                        pairs.push([idx as u32, (idx + 1) as u32]);
                    }
                    // Structural: vertical
                    if y + 1 < side {
                        pairs.push([idx as u32, (idx + side) as u32]);
                    }
                    // Shear: diagonal
                    if x + 1 < side && y + 1 < side {
                        pairs.push([idx as u32, (idx + side + 1) as u32]);
                    }
                    if x > 0 && y + 1 < side {
                        pairs.push([idx as u32, (idx + side - 1) as u32]);
                    }
                }
            }
            fields.index_pairs.insert("springs".to_string(), pairs);
        }
    }

    // Auto-generate grid springs from topology in pipeline config
    // (handled in build_pipeline)
}

fn init_vec3(n: usize, init: &Option<InitDef>) -> Vec<Vec3f> {
    match init {
        Some(InitDef::Random { min, max }) => {
            let mut rng = rand::thread_rng();
            let mn = [min.first().copied().unwrap_or(-1.0) as f32,
                       min.get(1).copied().unwrap_or(-1.0) as f32,
                       min.get(2).copied().unwrap_or(-1.0) as f32];
            let mx = [max.first().copied().unwrap_or(1.0) as f32,
                       max.get(1).copied().unwrap_or(1.0) as f32,
                       max.get(2).copied().unwrap_or(1.0) as f32];
            (0..n).map(|_| Vec3f::new(
                rng.gen_range(mn[0]..mx[0]),
                rng.gen_range(mn[1]..mx[1]),
                rng.gen_range(mn[2]..mx[2]),
            )).collect()
        }
        Some(InitDef::Constant { value }) => {
            let v = Vec3f::new(
                value.first().copied().unwrap_or(0.0) as f32,
                value.get(1).copied().unwrap_or(0.0) as f32,
                value.get(2).copied().unwrap_or(0.0) as f32,
            );
            vec![v; n]
        }
        Some(InitDef::Grid { spacing, origin }) => {
            let sp = *spacing as f32;
            let ox = origin.first().copied().unwrap_or(0.0) as f32;
            let oy = origin.get(1).copied().unwrap_or(0.0) as f32;
            let oz = origin.get(2).copied().unwrap_or(0.0) as f32;
            let side = (n as f32).sqrt().ceil() as usize;
            // Generate horizontal grid (X/Z plane at height Y=oy)
            // This is more natural for cloth/sheet simulations
            (0..n).map(|i| {
                let ix = i % side;
                let iz = i / side;
                Vec3f::new(ox + ix as f32 * sp, oy, oz + iz as f32 * sp)
            }).collect()
        }
        _ => vec![Vec3f::zero(); n],
    }
}

fn init_f32(n: usize, init: &Option<InitDef>) -> Vec<f32> {
    match init {
        Some(InitDef::Constant { value }) => {
            vec![value.first().copied().unwrap_or(0.0) as f32; n]
        }
        _ => vec![0.0; n],
    }
}

pub fn build_pipeline(manifest: &SimManifest) -> Result<Pipeline, String> {
    let mut pipeline = Pipeline::new();

    // If manifest has [[pipeline]] entries, use those
    // Otherwise, auto-build from forces/constraints

    // Check if we have explicit pipeline in the TOML
    // For now, build from forces → integrate → constraints pattern

    // ── Forces ──
    // Check for SPH fusion opportunity: if we have all three SPH forces,
    // replace them with a single fused module (2 neighbor passes instead of 3).
    let has_sph_density = manifest.forces.iter().any(|f| matches!(f, ForceDef::SphDensity { .. }));
    let has_sph_pressure = manifest.forces.iter().any(|f| matches!(f, ForceDef::SphPressure { .. }));
    let has_sph_viscosity = manifest.forces.iter().any(|f| matches!(f, ForceDef::SphViscosity { .. }));
    let fuse_sph = has_sph_density && has_sph_pressure && has_sph_viscosity;

    if fuse_sph {
        // Extract SPH params
        let smoothing_radius = manifest.forces.iter().find_map(|f| {
            if let ForceDef::SphDensity { smoothing_radius } = f {
                Some(*smoothing_radius as f32)
            } else { None }
        }).unwrap_or(0.025);
        let (gas_constant, rest_density) = manifest.forces.iter().find_map(|f| {
            if let ForceDef::SphPressure { gas_constant, rest_density } = f {
                Some((*gas_constant as f32, *rest_density as f32))
            } else { None }
        }).unwrap_or((2000.0, 1000.0));
        let viscosity_coefficient = manifest.forces.iter().find_map(|f| {
            if let ForceDef::SphViscosity { coefficient } = f {
                Some(*coefficient as f32)
            } else { None }
        }).unwrap_or(0.01);

        let spatial = manifest.spatial.as_ref();
        let cell_size = spatial.map(|s| s.cell_size as f32).unwrap_or(smoothing_radius * 2.0);
        let grid_dims = spatial.map(|s| {
            [
                *s.grid_dims.first().unwrap_or(&32),
                *s.grid_dims.get(1).unwrap_or(&32),
                *s.grid_dims.get(2).unwrap_or(&32),
            ]
        }).unwrap_or([32, 32, 32]);
        let n = manifest.particle_count();
        let particle_mass = rest_density / n as f32 * 0.001;

        println!("  ⚡ SPH kernel fusion: density + pressure + viscosity → 2 passes (was 3)");

        pipeline.add(Box::new(SphFusedModule {
            smoothing_radius,
            cell_size,
            grid_dims,
            particle_mass,
            gas_constant,
            rest_density,
            viscosity_coefficient,
        }));

        // Add non-SPH forces
        let mut expr_id = 0usize;
        for force in &manifest.forces {
            match force {
                ForceDef::SphDensity { .. } | ForceDef::SphPressure { .. } | ForceDef::SphViscosity { .. } => {
                    // already handled by fused module
                }
                ForceDef::Gravity { value } => {
                    pipeline.add(Box::new(GravityModule::new(
                        value.first().copied().unwrap_or(0.0) as f32,
                        value.get(1).copied().unwrap_or(-9.81) as f32,
                        value.get(2).copied().unwrap_or(0.0) as f32,
                    )));
                }
                ForceDef::Drag { coefficient } => {
                    pipeline.add(Box::new(DragModule { coefficient: *coefficient as f32 }));
                }
                ForceDef::Custom { expr } => {
                    pipeline.add(Box::new(ExprModule::new(expr, expr_id)));
                    expr_id += 1;
                }
                _ => {}
            }
        }
    } else {
        // No fusion — add forces individually
        for force in &manifest.forces {
            match force {
                ForceDef::Gravity { value } => {
                    pipeline.add(Box::new(GravityModule::new(
                        value.first().copied().unwrap_or(0.0) as f32,
                        value.get(1).copied().unwrap_or(-9.81) as f32,
                        value.get(2).copied().unwrap_or(0.0) as f32,
                    )));
                }
                ForceDef::Drag { coefficient } => {
                    pipeline.add(Box::new(DragModule { coefficient: *coefficient as f32 }));
                }
                ForceDef::SphDensity { smoothing_radius } => {
                    let spatial = manifest.spatial.as_ref();
                    let cell_size = spatial.map(|s| s.cell_size as f32).unwrap_or((*smoothing_radius * 2.0) as f32);
                    let grid_dims = spatial.map(|s| {
                        [
                            *s.grid_dims.first().unwrap_or(&32),
                            *s.grid_dims.get(1).unwrap_or(&32),
                            *s.grid_dims.get(2).unwrap_or(&32),
                        ]
                    }).unwrap_or([32, 32, 32]);
                    let n = manifest.particle_count();
                    let particle_mass = 1000.0 / n as f32 * 0.001;
                    pipeline.add(Box::new(SphDensityModule {
                        smoothing_radius: *smoothing_radius as f32,
                        cell_size,
                        grid_dims,
                        particle_mass,
                    }));
                }
                ForceDef::SphPressure { gas_constant, rest_density } => {
                    let spatial = manifest.spatial.as_ref();
                    let cell_size = spatial.map(|s| s.cell_size as f32).unwrap_or(0.05);
                    let grid_dims = spatial.map(|s| {
                        [
                            *s.grid_dims.first().unwrap_or(&32),
                            *s.grid_dims.get(1).unwrap_or(&32),
                            *s.grid_dims.get(2).unwrap_or(&32),
                        ]
                    }).unwrap_or([32, 32, 32]);
                    let smoothing_radius = manifest.forces.iter().find_map(|f| {
                        if let ForceDef::SphDensity { smoothing_radius } = f {
                            Some(*smoothing_radius as f32)
                        } else { None }
                    }).unwrap_or(0.025);
                    let n = manifest.particle_count();
                    let particle_mass = *rest_density as f32 / n as f32 * 0.001;
                    pipeline.add(Box::new(SphPressureModule {
                        gas_constant: *gas_constant as f32,
                        rest_density: *rest_density as f32,
                        smoothing_radius,
                        cell_size,
                        grid_dims,
                        particle_mass,
                    }));
                }
                ForceDef::SphViscosity { coefficient } => {
                    let spatial = manifest.spatial.as_ref();
                    let cell_size = spatial.map(|s| s.cell_size as f32).unwrap_or(0.05);
                    let grid_dims = spatial.map(|s| {
                        [
                            *s.grid_dims.first().unwrap_or(&32),
                            *s.grid_dims.get(1).unwrap_or(&32),
                            *s.grid_dims.get(2).unwrap_or(&32),
                        ]
                    }).unwrap_or([32, 32, 32]);
                    let smoothing_radius = manifest.forces.iter().find_map(|f| {
                        if let ForceDef::SphDensity { smoothing_radius } = f {
                            Some(*smoothing_radius as f32)
                        } else { None }
                    }).unwrap_or(0.025);
                    let n = manifest.particle_count();
                    let particle_mass = 1000.0 / n as f32 * 0.001;
                    pipeline.add(Box::new(SphViscosityModule {
                        coefficient: *coefficient as f32,
                        smoothing_radius,
                        cell_size,
                        grid_dims,
                        particle_mass,
                    }));
                }
                ForceDef::Custom { expr } => {
                    static EXPR_CTR: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(100);
                    let id = EXPR_CTR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    pipeline.add(Box::new(ExprModule::new(expr, id)));
                }
                _ => {}
            }
        }
    }

    // Springs (if defined)
    if let Some(ref springs) = manifest.springs {
        pipeline.add(Box::new(SpringModule {
            stiffness: springs.stiffness as f32,
            damping: springs.damping as f32,
        }));
    }

    // Integrate
    pipeline.add(Box::new(IntegrateModule));

    // Constraints
    for constraint in &manifest.constraints {
        match constraint {
            ConstraintDef::GroundPlane { y, restitution } => {
                pipeline.add(Box::new(GroundPlaneModule {
                    y: *y as f32,
                    restitution: *restitution as f32,
                }));
            }
            ConstraintDef::Sphere { center, radius, restitution } => {
                pipeline.add(Box::new(SphereColliderModule {
                    cx: center.first().copied().unwrap_or(0.0) as f32,
                    cy: center.get(1).copied().unwrap_or(0.0) as f32,
                    cz: center.get(2).copied().unwrap_or(0.0) as f32,
                    radius: *radius as f32,
                    restitution: *restitution as f32,
                }));
            }
            ConstraintDef::Box { min, max, restitution } => {
                pipeline.add(Box::new(BoxColliderModule {
                    min_x: min.first().copied().unwrap_or(0.0) as f32,
                    min_y: min.get(1).copied().unwrap_or(0.0) as f32,
                    min_z: min.get(2).copied().unwrap_or(0.0) as f32,
                    max_x: max.first().copied().unwrap_or(1.0) as f32,
                    max_y: max.get(1).copied().unwrap_or(1.0) as f32,
                    max_z: max.get(2).copied().unwrap_or(1.0) as f32,
                    restitution: *restitution as f32,
                }));
            }
            _ => {}
        }
    }

    Ok(pipeline)
}
