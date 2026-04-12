//! # Tape — Record kernel launches for automatic backward pass.
//!
//! The `Tape` records a sequence of kernel launches (forward pass)
//! and can replay them in reverse order (backward pass) to compute gradients.
//!
//! ## Usage
//!
//! ```rust,ignore
//! let tape = Tape::new();
//!
//! // Record forward operations
//! tape.record_launch("kernel_name", forward_fn, backward_fn);
//!
//! // Replay backward
//! tape.backward();
//! ```

use std::sync::Mutex;

use crate::ForgeError;

/// A recorded operation: forward was already executed, backward is deferred.
struct TapeEntry {
    /// Name for debugging
    name: String,
    /// Backward function — called during tape.backward()
    backward: Box<dyn FnOnce() -> Result<(), ForgeError> + Send>,
}

/// Tape records kernel launches and replays them backward.
///
/// This is the simplest possible tape: it records backward closures
/// and replays them in reverse order.
pub struct Tape {
    entries: Mutex<Vec<TapeEntry>>,
}

impl Tape {
    /// Create a new empty tape.
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
        }
    }

    /// Record a backward operation.
    ///
    /// The forward kernel should already be launched before calling this.
    /// Pass a closure that calls `launch_adjoint()` with the appropriate arrays.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // 1. Run forward
    /// my_kernel::launch(&input, &mut output, n, 0)?;
    ///
    /// // 2. Record backward
    /// tape.record("my_kernel", move || {
    ///     my_kernel::launch_adjoint(&input, &mut output, &mut adj_input, &mut adj_output, n, 0)
    /// });
    /// ```
    pub fn record<F>(&self, name: &str, backward: F)
    where
        F: FnOnce() -> Result<(), ForgeError> + Send + 'static,
    {
        let mut entries = self.entries.lock().unwrap();
        entries.push(TapeEntry {
            name: name.to_string(),
            backward: Box::new(backward),
        });
    }

    /// Run all recorded backward operations in reverse order.
    ///
    /// This consumes all entries from the tape (tape is empty after calling).
    pub fn backward(&self) -> Result<(), ForgeError> {
        let mut entries = self.entries.lock().unwrap();
        let ops: Vec<TapeEntry> = entries.drain(..).collect();

        // Execute in reverse order
        for entry in ops.into_iter().rev() {
            (entry.backward)().map_err(|e| {
                ForgeError::LaunchFailed(format!(
                    "backward pass for '{}' failed: {}",
                    entry.name, e
                ))
            })?;
        }

        Ok(())
    }

    /// Number of recorded operations.
    pub fn len(&self) -> usize {
        self.entries.lock().unwrap().len()
    }

    /// Whether the tape is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all recorded operations without running them.
    pub fn clear(&self) {
        self.entries.lock().unwrap().clear();
    }
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}
