//! Tests for Multi-GPU API.

#[cfg(feature = "cuda")]
mod tests {
    use forge_runtime::{Array, Device};
    use forge_runtime::cuda;

    fn skip_if_no_gpu() -> bool {
        cuda::init();
        cuda::device_count() == 0
    }

    #[test]
    fn test_launch_on() {
        if skip_if_no_gpu() {
            eprintln!("Skipping: no GPU");
            return;
        }

        // launch_on returns a stream for the given device
        let stream = cuda::launch_on(0);
        // We can use it to launch kernels
        stream.synchronize().expect("sync failed");
        eprintln!("✅ launch_on: got stream for device 0");
    }

    #[test]
    fn test_copy_peer_single_device() {
        if skip_if_no_gpu() {
            eprintln!("Skipping: no GPU");
            return;
        }

        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let src = Array::from_vec(data.clone(), Device::Cuda(0));

        // Copy to the same device (trivial P2P)
        let dst = cuda::copy_peer(&src, 0);
        let result = dst.to_vec();
        assert_eq!(result, data);
        assert_eq!(dst.device(), Device::Cuda(0));
        eprintln!("✅ copy_peer single device: data preserved");
    }

    #[test]
    fn test_copy_peer_preserves_shape() {
        if skip_if_no_gpu() {
            eprintln!("Skipping: no GPU");
            return;
        }

        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let mut src = Array::from_vec(data.clone(), Device::Cuda(0));
        src.reshape(forge_runtime::Shape::new_2d(4, 6));

        let dst = cuda::copy_peer(&src, 0);
        assert_eq!(dst.ndim(), 2);
        assert_eq!(dst.shape().dims[0], 4);
        assert_eq!(dst.shape().dims[1], 6);
        assert_eq!(dst.to_vec(), data);
        eprintln!("✅ copy_peer preserves shape");
    }

    #[test]
    fn test_enable_peer_access_same_device() {
        if skip_if_no_gpu() {
            eprintln!("Skipping: no GPU");
            return;
        }

        // Enabling P2P between a device and itself should return false (no-op)
        let result = cuda::enable_peer_access(0, 0);
        assert_eq!(result.unwrap(), false);
        eprintln!("✅ enable_peer_access same device: no-op");
    }

    #[test]
    fn test_can_access_peer_same_device() {
        if skip_if_no_gpu() {
            eprintln!("Skipping: no GPU");
            return;
        }

        // A device can always "access" itself
        assert!(cuda::can_access_peer(0, 0));
        eprintln!("✅ can_access_peer same device: true");
    }

    #[test]
    fn test_launch_on_with_kernel() {
        if skip_if_no_gpu() {
            eprintln!("Skipping: no GPU");
            return;
        }

        // Compile a simple kernel and launch on device 0
        let kernel = forge_runtime::CompiledKernel::compile(
            r#"extern "C" __global__ void fill_42(float* data, int n) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) data[i] = 42.0f;
            }"#,
            "fill_42",
        ).expect("compile failed");

        let n = 1024;
        let mut data = Array::<f32>::zeros(n, Device::Cuda(0));

        // Use launch_on to get stream, then launch
        let stream = cuda::launch_on(0);
        let func = kernel.get_function(0).expect("get_function failed");
        let config = cuda::LaunchConfig::for_num_elems(n as u32);
        let n_i32 = n as i32;

        unsafe {
            use forge_runtime::cuda::PushKernelArg;
            let mut builder = stream.launch_builder(&func);
            builder.arg(data.cuda_slice_mut().unwrap());
            builder.arg(&n_i32);
            builder.launch(config).expect("launch failed");
        }
        stream.synchronize().expect("sync failed");

        let result = data.to_vec();
        assert!(result.iter().all(|&x| (x - 42.0).abs() < 1e-6));
        eprintln!("✅ launch_on with kernel: {} elements filled correctly", n);
    }

    #[test]
    fn test_multi_device_info() {
        if skip_if_no_gpu() {
            eprintln!("Skipping: no GPU");
            return;
        }

        let count = cuda::device_count();
        eprintln!("Multi-GPU info: {} device(s)", count);

        for i in 0..count {
            let name = cuda::device_name(i);
            let (major, minor) = cuda::compute_capability(i);
            let (free, total) = cuda::mem_info(i);
            eprintln!("  Device {}: {} (SM {}.{}) - {:.0} MB free / {:.0} MB total",
                i, name, major, minor,
                free as f64 / 1024.0 / 1024.0,
                total as f64 / 1024.0 / 1024.0);
        }

        // If multiple GPUs exist, test P2P
        if count > 1 {
            let can = cuda::can_access_peer(0, 1);
            eprintln!("  P2P 0↔1: {}", if can { "available" } else { "not available" });

            if can {
                let enabled = cuda::enable_peer_access(0, 1).unwrap();
                eprintln!("  P2P enabled: {}", enabled);
            }
        }
    }
}
