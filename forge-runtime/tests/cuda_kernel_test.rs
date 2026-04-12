//! End-to-end test: compile and launch a CUDA kernel via nvrtc.

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use forge_runtime::*;
    use cudarc::driver::PushKernelArg;

    const ADD_ONE_KERNEL: &str = r#"
extern "C" __global__ void add_one(float* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        data[tid] = data[tid] + 1.0f;
    }
}
"#;

    #[test]
    fn test_compile_and_launch_kernel() {
        forge_runtime::cuda::init();
        if forge_runtime::cuda::device_count() == 0 {
            eprintln!("Skipping: no GPU");
            return;
        }

        // 1. Compile the kernel
        let kernel = CompiledKernel::compile(ADD_ONE_KERNEL, "add_one")
            .expect("Failed to compile kernel");

        // 2. Create input data on GPU
        let n = 1024usize;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut arr = Array::from_vec(data, Device::Cuda(0));

        // 3. Launch the kernel
        let ordinal = 0;
        let func = kernel.get_function(ordinal).expect("Failed to get function");
        let stream = forge_runtime::cuda::default_stream(ordinal);

        let n_i32 = n as i32;
        let cfg = launch_config_1d(n);

        let cuda_slice = arr.cuda_slice_mut().unwrap();
        unsafe {
            stream
                .launch_builder(&func)
                .arg(cuda_slice)
                .arg(&n_i32)
                .launch(cfg)
                .expect("Kernel launch failed");
        }

        stream.synchronize().expect("Sync failed");

        // 4. Verify results
        let result = arr.to_vec();
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - (i as f32 + 1.0)).abs() < 1e-6,
                "Mismatch at {}: expected {}, got {}",
                i,
                i as f32 + 1.0,
                val
            );
        }
        eprintln!("✅ CUDA kernel add_one: {} elements processed correctly", n);
    }

    const PARTICLE_KERNEL: &str = r#"
extern "C" __global__ void integrate(
    float* pos_x, float* pos_y, float* pos_z,
    float* vel_x, float* vel_y, float* vel_z,
    float dt, float gravity, float ground_y, float restitution,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // Apply gravity
    vel_y[tid] = vel_y[tid] + gravity * dt;

    // Integrate position
    pos_x[tid] = pos_x[tid] + vel_x[tid] * dt;
    pos_y[tid] = pos_y[tid] + vel_y[tid] * dt;
    pos_z[tid] = pos_z[tid] + vel_z[tid] * dt;

    // Ground collision
    if (pos_y[tid] < ground_y) {
        pos_y[tid] = ground_y;
        vel_y[tid] = -vel_y[tid] * restitution;
    }
}
"#;

    #[test]
    fn test_particle_simulation_kernel() {
        forge_runtime::cuda::init();
        if forge_runtime::cuda::device_count() == 0 {
            return;
        }

        let n = 100_000usize;
        let dt = 1.0f32 / 60.0;
        let gravity = -9.81f32;
        let ground_y = 0.0f32;
        let restitution = 0.7f32;
        let steps = 300;

        // Compile
        let kernel =
            CompiledKernel::compile(PARTICLE_KERNEL, "integrate").expect("Failed to compile");
        let func = kernel.get_function(0).expect("get_function");
        let stream = forge_runtime::cuda::default_stream(0);

        // Init positions (spread in XZ, height 5-25)
        let pos_x_data: Vec<f32> = (0..n).map(|i| ((i * 7) % 200) as f32 / 10.0 - 10.0).collect();
        let pos_y_data: Vec<f32> = (0..n).map(|i| (i % 200) as f32 / 10.0 + 5.0).collect();
        let pos_z_data: Vec<f32> = (0..n).map(|i| ((i * 13) % 200) as f32 / 10.0 - 10.0).collect();

        let mut pos_x = Array::from_vec(pos_x_data, Device::Cuda(0));
        let mut pos_y = Array::from_vec(pos_y_data, Device::Cuda(0));
        let mut pos_z = Array::from_vec(pos_z_data, Device::Cuda(0));
        let mut vel_x = Array::<f32>::zeros(n, Device::Cuda(0));
        let mut vel_y = Array::<f32>::zeros(n, Device::Cuda(0));
        let mut vel_z = Array::<f32>::zeros(n, Device::Cuda(0));

        let n_i32 = n as i32;
        let cfg = launch_config_1d(n);

        // Simulate
        let start = std::time::Instant::now();
        for _ in 0..steps {
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(pos_x.cuda_slice_mut().unwrap())
                    .arg(pos_y.cuda_slice_mut().unwrap())
                    .arg(pos_z.cuda_slice_mut().unwrap())
                    .arg(vel_x.cuda_slice_mut().unwrap())
                    .arg(vel_y.cuda_slice_mut().unwrap())
                    .arg(vel_z.cuda_slice_mut().unwrap())
                    .arg(&dt)
                    .arg(&gravity)
                    .arg(&ground_y)
                    .arg(&restitution)
                    .arg(&n_i32)
                    .launch(cfg)
                    .expect("launch failed");
            }
        }
        stream.synchronize().expect("sync");
        let elapsed = start.elapsed();

        // Verify
        let final_y = pos_y.to_vec();
        assert!(
            final_y.iter().all(|&y| y >= ground_y - 0.001),
            "Some particles fell below ground!"
        );

        let throughput = (n * steps) as f64 / elapsed.as_secs_f64();
        eprintln!(
            "✅ Particle simulation: {}K particles × {} steps in {:.3}s ({:.0} particle-steps/s)",
            n / 1000,
            steps,
            elapsed.as_secs_f64(),
            throughput
        );
    }
}
