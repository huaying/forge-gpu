use crate::modules::{FieldSet, SimModule};
use forge_runtime::ForgeError;

pub struct PinModule {
    /// Particle indices to pin
    pub indices: Vec<u32>,
    /// Fixed positions (x,y,z) for each pinned particle
    pub positions: Vec<[f32; 3]>,
}

impl SimModule for PinModule {
    fn name(&self) -> &str { "pin" }

    fn execute(&self, fields: &mut FieldSet, _dt: f32) -> Result<(), ForgeError> {
        // Read back, modify pinned particles, write back
        // For small pin counts this is fine; for large counts we'd use a GPU kernel
        let mut px = fields.f32_fields.get("pos_x")
            .ok_or_else(|| ForgeError::LaunchFailed("missing pos_x".into()))?.to_vec();
        let mut py = fields.f32_fields.get("pos_y")
            .ok_or_else(|| ForgeError::LaunchFailed("missing pos_y".into()))?.to_vec();
        let mut pz = fields.f32_fields.get("pos_z")
            .ok_or_else(|| ForgeError::LaunchFailed("missing pos_z".into()))?.to_vec();
        let mut vx = fields.f32_fields.get("vel_x")
            .ok_or_else(|| ForgeError::LaunchFailed("missing vel_x".into()))?.to_vec();
        let mut vy = fields.f32_fields.get("vel_y")
            .ok_or_else(|| ForgeError::LaunchFailed("missing vel_y".into()))?.to_vec();
        let mut vz = fields.f32_fields.get("vel_z")
            .ok_or_else(|| ForgeError::LaunchFailed("missing vel_z".into()))?.to_vec();

        for (k, &idx) in self.indices.iter().enumerate() {
            let i = idx as usize;
            if i < px.len() {
                px[i] = self.positions[k][0];
                py[i] = self.positions[k][1];
                pz[i] = self.positions[k][2];
                vx[i] = 0.0;
                vy[i] = 0.0;
                vz[i] = 0.0;
            }
        }

        // Write back
        fields.add_f32("pos_x", px);
        fields.add_f32("pos_y", py);
        fields.add_f32("pos_z", pz);
        fields.add_f32("vel_x", vx);
        fields.add_f32("vel_y", vy);
        fields.add_f32("vel_z", vz);

        Ok(())
    }
}
