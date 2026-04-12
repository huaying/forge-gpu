//! Simulation runner — compiles and executes a manifest on GPU.

use forge_core::Vec3f;
use forge_runtime::{cuda, Array, Device};
use rand::Rng;

use crate::schema::*;

/// Result of running a simulation.
pub struct SimResult {
    /// Final positions (flattened: [x0,y0,z0, x1,y1,z1, ...])
    pub positions: Vec<f32>,
    /// Number of particles
    pub count: usize,
    /// Number of steps executed
    pub steps: usize,
    /// Wall-clock time in seconds
    pub elapsed_secs: f64,
    /// Per-frame snapshots (if output configured)
    pub frames: Vec<Vec<f32>>,
}

/// Run a simulation from a manifest.
pub fn run(manifest: &SimManifest) -> Result<SimResult, String> {
    // Validate
    crate::validate::validate(manifest)
        .map_err(|errs| errs.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("; "))?;

    cuda::init();
    if cuda::device_count() == 0 {
        return Err("No GPU available".to_string());
    }

    match manifest.simulation.r#type.as_str() {
        "particles" => run_particles(manifest),
        "springs" => run_springs(manifest),
        other => Err(format!("Unknown simulation type: '{}'", other)),
    }
}

fn run_particles(manifest: &SimManifest) -> Result<SimResult, String> {
    let n = manifest.particle_count();
    let dt = manifest.simulation.dt as f32;
    let duration = manifest.simulation.duration as f32;
    let substeps = manifest.simulation.substeps;
    let total_steps = ((duration / dt) as usize) / substeps.max(1) as usize;
    let frame_interval = if let Some(ref out) = manifest.output {
        let fps = out.fps.max(1);
        ((1.0 / fps as f64) / manifest.simulation.dt) as usize
    } else {
        0
    };

    // Initialize positions
    let mut pos = init_vec3_field(n, &manifest.fields, "position");
    let mut vel = init_vec3_field(n, &manifest.fields, "velocity");

    // Parse forces
    let gravity = extract_gravity(&manifest.forces);
    let drag = extract_drag(&manifest.forces);

    // Parse constraints
    let ground = extract_ground(&manifest.constraints);

    // Upload to GPU
    let (mut px, mut py, mut pz) = split_to_gpu(&pos, n);
    let (mut vx, mut vy, mut vz) = split_to_gpu(&vel, n);

    let mut frames = Vec::new();
    let start = std::time::Instant::now();
    let mut step_count = 0;

    for step in 0..total_steps {
        for _sub in 0..substeps {
            // Integrate on GPU
            integrate_particles_gpu(
                &mut px, &mut py, &mut pz,
                &mut vx, &mut vy, &mut vz,
                gravity, drag, ground, dt, n,
            ).map_err(|e| format!("GPU error at step {}: {}", step, e))?;
            step_count += 1;
        }

        // Capture frame if needed
        if frame_interval > 0 && step % frame_interval == 0 {
            let frame = gather_positions(&px, &py, &pz, n);
            frames.push(frame);
        }
    }

    let elapsed = start.elapsed();

    // Read back final positions
    let final_positions = gather_positions(&px, &py, &pz, n);

    Ok(SimResult {
        positions: final_positions,
        count: n,
        steps: step_count,
        elapsed_secs: elapsed.as_secs_f64(),
        frames,
    })
}

fn run_springs(manifest: &SimManifest) -> Result<SimResult, String> {
    // For now, springs use the same particle framework + spring energy
    // This is a placeholder for the full spring implementation
    run_particles(manifest)
}

// ── Helpers ──

fn init_vec3_field(n: usize, fields: &[FieldDef], name: &str) -> Vec<Vec3f> {
    let field = fields.iter().find(|f| f.name == name);
    match field {
        Some(FieldDef { init: Some(InitDef::Random { min, max }), .. }) => {
            let mut rng = rand::thread_rng();
            let mn = [
                *min.first().unwrap_or(&-1.0) as f32,
                *min.get(1).unwrap_or(&-1.0) as f32,
                *min.get(2).unwrap_or(&-1.0) as f32,
            ];
            let mx = [
                *max.first().unwrap_or(&1.0) as f32,
                *max.get(1).unwrap_or(&1.0) as f32,
                *max.get(2).unwrap_or(&1.0) as f32,
            ];
            (0..n).map(|_| Vec3f::new(
                rng.gen_range(mn[0]..mx[0]),
                rng.gen_range(mn[1]..mx[1]),
                rng.gen_range(mn[2]..mx[2]),
            )).collect()
        }
        Some(FieldDef { init: Some(InitDef::Constant { value }), .. }) => {
            let v = Vec3f::new(
                *value.first().unwrap_or(&0.0) as f32,
                *value.get(1).unwrap_or(&0.0) as f32,
                *value.get(2).unwrap_or(&0.0) as f32,
            );
            vec![v; n]
        }
        _ => vec![Vec3f::zero(); n],
    }
}

fn split_to_gpu(vecs: &[Vec3f], n: usize) -> (Array<f32>, Array<f32>, Array<f32>) {
    let px: Vec<f32> = vecs.iter().map(|v| v.x).collect();
    let py: Vec<f32> = vecs.iter().map(|v| v.y).collect();
    let pz: Vec<f32> = vecs.iter().map(|v| v.z).collect();
    (
        Array::from_vec(px, Device::Cuda(0)),
        Array::from_vec(py, Device::Cuda(0)),
        Array::from_vec(pz, Device::Cuda(0)),
    )
}

fn gather_positions(px: &Array<f32>, py: &Array<f32>, pz: &Array<f32>, n: usize) -> Vec<f32> {
    let x = px.to_vec();
    let y = py.to_vec();
    let z = pz.to_vec();
    let mut out = Vec::with_capacity(n * 3);
    for i in 0..n {
        out.push(x[i]);
        out.push(y[i]);
        out.push(z[i]);
    }
    out
}

fn extract_gravity(forces: &[ForceDef]) -> [f32; 3] {
    for f in forces {
        if let ForceDef::Gravity { value } = f {
            return [
                *value.first().unwrap_or(&0.0) as f32,
                *value.get(1).unwrap_or(&-9.81) as f32,
                *value.get(2).unwrap_or(&0.0) as f32,
            ];
        }
    }
    [0.0, -9.81, 0.0] // default gravity
}

fn extract_drag(forces: &[ForceDef]) -> f32 {
    for f in forces {
        if let ForceDef::Drag { coefficient } = f {
            return *coefficient as f32;
        }
    }
    0.0
}

fn extract_ground(constraints: &[ConstraintDef]) -> Option<(f32, f32)> {
    for c in constraints {
        if let ConstraintDef::GroundPlane { y, restitution } = c {
            return Some((*y as f32, *restitution as f32));
        }
    }
    None
}

/// GPU particle integration using compiled CUDA kernel.
fn integrate_particles_gpu(
    px: &mut Array<f32>, py: &mut Array<f32>, pz: &mut Array<f32>,
    vx: &mut Array<f32>, vy: &mut Array<f32>, vz: &mut Array<f32>,
    gravity: [f32; 3],
    drag: f32,
    ground: Option<(f32, f32)>,
    dt: f32,
    n: usize,
) -> Result<(), forge_runtime::ForgeError> {
    use std::sync::OnceLock;

    static KERNEL: OnceLock<forge_runtime::CompiledKernel> = OnceLock::new();

    let kernel = KERNEL.get_or_init(|| {
        let source = r#"
extern "C" __global__ void integrate_particles(
    float* px, float* py, float* pz,
    float* vx, float* vy, float* vz,
    float gx, float gy, float gz,
    float drag, float ground_y, float restitution,
    int has_ground, float dt, int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // Apply gravity
        vx[tid] += gx * dt;
        vy[tid] += gy * dt;
        vz[tid] += gz * dt;

        // Apply drag
        if (drag > 0.0f) {
            vx[tid] *= (1.0f - drag * dt);
            vy[tid] *= (1.0f - drag * dt);
            vz[tid] *= (1.0f - drag * dt);
        }

        // Integrate position
        px[tid] += vx[tid] * dt;
        py[tid] += vy[tid] * dt;
        pz[tid] += vz[tid] * dt;

        // Ground constraint
        if (has_ground && py[tid] < ground_y) {
            py[tid] = ground_y;
            vy[tid] = -vy[tid] * restitution;
        }
    }
}
"#;
        forge_runtime::CompiledKernel::compile(source, "integrate_particles")
            .expect("Failed to compile particle kernel")
    });

    let func = kernel.get_function(0)?;
    let stream = cuda::default_stream(0);
    let config = cuda::LaunchConfig::for_num_elems(n as u32);

    let (ground_y, restitution, has_ground) = match ground {
        Some((y, r)) => (y, r, 1i32),
        None => (0.0f32, 0.0f32, 0i32),
    };

    unsafe {
        use cuda::PushKernelArg;
        let n_i32 = n as i32;
        let mut builder = stream.launch_builder(&func);
        builder.arg(px.cuda_slice_mut().unwrap());
        builder.arg(py.cuda_slice_mut().unwrap());
        builder.arg(pz.cuda_slice_mut().unwrap());
        builder.arg(vx.cuda_slice_mut().unwrap());
        builder.arg(vy.cuda_slice_mut().unwrap());
        builder.arg(vz.cuda_slice_mut().unwrap());
        builder.arg(&gravity[0]);
        builder.arg(&gravity[1]);
        builder.arg(&gravity[2]);
        builder.arg(&drag);
        builder.arg(&ground_y);
        builder.arg(&restitution);
        builder.arg(&has_ground);
        builder.arg(&dt);
        builder.arg(&n_i32);
        builder.launch(config)
            .map_err(|e| forge_runtime::ForgeError::LaunchFailed(format!("{:?}", e)))?;
    }

    stream.synchronize()
        .map_err(|e| forge_runtime::ForgeError::SyncFailed(format!("{:?}", e)))?;

    Ok(())
}
