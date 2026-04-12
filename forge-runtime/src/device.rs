//! Device abstraction.

/// Represents a compute device (GPU or CPU).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    /// CPU fallback (uses Rayon for parallelism in the future).
    Cpu,
    /// CUDA GPU device with the given ordinal.
    Cuda(usize),
}

impl Device {
    /// Returns the best available device (prefers CUDA over CPU).
    pub fn best_available() -> Self {
        #[cfg(feature = "cuda")]
        {
            if crate::cuda::device_count() > 0 {
                return Device::Cuda(0);
            }
        }
        Device::Cpu
    }

    /// True if this is a CUDA device.
    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(i) => write!(f, "cuda:{}", i),
        }
    }
}

/// Top-level Forge context. Initializes the runtime.
pub struct Forge {
    device: Device,
}

impl Forge {
    /// Initialize the Forge runtime with the best available device.
    pub fn init() -> Self {
        #[cfg(feature = "cuda")]
        {
            crate::cuda::init();
        }
        let device = Device::best_available();
        eprintln!("🔥 Forge initialized on {}", device);
        Self { device }
    }

    /// Initialize with a specific device.
    pub fn with_device(device: Device) -> Self {
        #[cfg(feature = "cuda")]
        {
            crate::cuda::init();
        }
        eprintln!("🔥 Forge initialized on {}", device);
        Self { device }
    }

    /// Get the current device.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Synchronize all pending operations on the current device.
    pub fn synchronize(&self) {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(ordinal) = self.device {
            crate::cuda::synchronize(ordinal);
        }
    }
}
