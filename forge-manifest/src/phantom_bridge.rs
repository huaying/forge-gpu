//! Forge → Phantom streaming bridge.
//!
//! Implements `phantom_stream::StreamSource` for Forge simulations.
//! Renders particles to a CPU framebuffer (2D projection) and streams
//! via Phantom's NVENC pipeline.
//!
//! Gated behind the `phantom` feature flag.

#[cfg(feature = "phantom")]
pub mod phantom_bridge {
    use crate::schema::SimManifest;
    use crate::modules::{FieldSet, Pipeline};
    use crate::manifest_runner::{build_pipeline, init_fields};
    use forge_runtime::{cuda, Device};
    use phantom_stream::{StreamSource, StreamFrame, CpuFrame, GpuPixelFormat};
    use std::time::Instant;

    /// Forge simulation as a StreamSource for Phantom.
    pub struct ForgeStreamSource {
        fields: FieldSet,
        pipeline: Pipeline,
        dt: f32,
        substeps: u32,
        width: u32,
        height: u32,
        framebuffer: Vec<u8>, // RGBA
        step: usize,
        total_steps: usize,
    }

    impl ForgeStreamSource {
        pub fn new(manifest: &SimManifest, width: u32, height: u32) -> Result<Self, String> {
            crate::validate::validate(manifest)
                .map_err(|errs| errs.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("; "))?;

            cuda::init();
            let device = Device::Cuda(0);
            let n = manifest.particle_count();
            let dt = manifest.simulation.dt as f32;
            let substeps = manifest.simulation.substeps;
            let duration = manifest.simulation.duration as f32;
            let total_steps = ((duration / dt) as usize) / substeps.max(1) as usize;

            let mut fields = FieldSet::new(n, device);
            init_fields(&mut fields, manifest);

            let pipeline = build_pipeline(manifest)?;

            let fb_size = (width * height * 4) as usize;

            Ok(Self {
                fields,
                pipeline,
                dt,
                substeps,
                width,
                height,
                framebuffer: vec![0u8; fb_size],
                step: 0,
                total_steps,
            })
        }

        /// Render particles to the RGBA framebuffer (simple 2D projection).
        fn render(&mut self) {
            let w = self.width as usize;
            let h = self.height as usize;

            // Clear to dark background
            for pixel in self.framebuffer.chunks_exact_mut(4) {
                pixel[0] = 15;  // R
                pixel[1] = 15;  // G
                pixel[2] = 20;  // B
                pixel[3] = 255; // A
            }

            // Get positions from GPU
            let px = self.fields.f32_fields.get("pos_x").map(|a| a.to_vec()).unwrap_or_default();
            let py = self.fields.f32_fields.get("pos_y").map(|a| a.to_vec()).unwrap_or_default();
            let pz = self.fields.f32_fields.get("pos_z").map(|a| a.to_vec()).unwrap_or_default();

            let n = px.len();

            // Simple orthographic projection: XZ plane from above, Y=height as color
            // Camera: looking down Y axis, X maps to screen X, Z maps to screen Y
            // Scale: auto-fit to [-5, 5] range
            let scale = (w.min(h) as f32) / 10.0;
            let cx = w as f32 / 2.0;
            let cy = h as f32 / 2.0;

            for i in 0..n {
                let sx = (px[i] * scale + cx) as i32;
                let sy = (cy - pz[i] * scale) as i32; // flip Z for screen coords

                if sx >= 0 && sx < w as i32 && sy >= 0 && sy < h as i32 {
                    let idx = (sy as usize * w + sx as usize) * 4;

                    // Color based on height (Y)
                    let height_norm = ((py[i] / 4.0).max(0.0).min(1.0) * 255.0) as u8;
                    self.framebuffer[idx]     = 50 + height_norm / 2;       // R
                    self.framebuffer[idx + 1] = 120 + height_norm / 2;      // G
                    self.framebuffer[idx + 2] = 200 + (255 - height_norm) / 4; // B
                    self.framebuffer[idx + 3] = 255;                        // A
                }
            }
        }
    }

    impl StreamSource for ForgeStreamSource {
        fn resolution(&self) -> (u32, u32) {
            (self.width, self.height)
        }

        fn next_frame(&mut self) -> anyhow::Result<Option<StreamFrame>> {
            if self.step >= self.total_steps {
                return Ok(None);
            }

            // Run simulation substeps
            for _ in 0..self.substeps {
                self.pipeline.step(&mut self.fields, self.dt)
                    .map_err(|e| anyhow::anyhow!("sim error: {}", e))?;
            }
            self.step += 1;

            // Render to framebuffer
            self.render();

            Ok(Some(StreamFrame::Cpu(CpuFrame {
                data: self.framebuffer.clone(),
                width: self.width,
                height: self.height,
                format: GpuPixelFormat::Rgba8,
                timestamp: Instant::now(),
            })))
        }
    }
}
