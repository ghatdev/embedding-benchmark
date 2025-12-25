//! Resource monitoring for embedding benchmarks
//!
//! Tracks memory usage, GPU metrics, and throughput during model loading and embedding.

use crate::embedders::traits::ResourceMetrics;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use sysinfo::{Pid, ProcessesToUpdate, System};

/// Sampling interval for peak memory tracking
const SAMPLE_INTERVAL_MS: u64 = 50;

/// Monitor for tracking resource usage during embedding benchmarks
pub struct ResourceMonitor {
    system: System,
    pid: Pid,
    baseline_memory_mb: f64,
    model_memory_mb: f64,
    model_load_time: Option<Duration>,
    peak_memory_mb: Arc<AtomicU64>,
    sampling_active: Arc<AtomicBool>,
    is_unified_memory: bool,
}

impl ResourceMonitor {
    /// Create a new resource monitor
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();

        let pid = Pid::from_u32(std::process::id());

        // Detect if we're on Apple Silicon (unified memory)
        let is_unified_memory = Self::detect_unified_memory();

        Self {
            system,
            pid,
            baseline_memory_mb: 0.0,
            model_memory_mb: 0.0,
            model_load_time: None,
            peak_memory_mb: Arc::new(AtomicU64::new(0)),
            sampling_active: Arc::new(AtomicBool::new(false)),
            is_unified_memory,
        }
    }

    /// Detect if running on Apple Silicon with unified memory
    fn detect_unified_memory() -> bool {
        #[cfg(target_os = "macos")]
        {
            // Check for Apple Silicon via architecture
            #[cfg(target_arch = "aarch64")]
            {
                return true;
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                return false;
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }

    /// Get current process memory usage in MB
    fn get_process_memory_mb(&mut self) -> f64 {
        // Refresh only our process (sysinfo 0.32+ API)
        self.system.refresh_processes(ProcessesToUpdate::Some(&[self.pid]), true);
        if let Some(process) = self.system.process(self.pid) {
            // Use RSS (Resident Set Size) - actual physical memory used
            process.memory() as f64 / (1024.0 * 1024.0)
        } else {
            0.0
        }
    }

    /// Snapshot baseline memory before model load
    pub fn snapshot_baseline(&mut self) {
        self.baseline_memory_mb = self.get_process_memory_mb();
        self.peak_memory_mb
            .store(self.baseline_memory_mb.to_bits(), Ordering::SeqCst);
        tracing::debug!(
            "Baseline memory: {:.1} MB",
            self.baseline_memory_mb
        );
    }

    /// Record memory after model is loaded
    pub fn record_model_loaded(&mut self, load_duration: Duration) {
        let current_memory = self.get_process_memory_mb();
        self.model_memory_mb = current_memory - self.baseline_memory_mb;
        self.model_load_time = Some(load_duration);

        // Update peak if current is higher
        self.update_peak(current_memory);

        tracing::debug!(
            "Model loaded: {:.1} MB (delta: {:.1} MB) in {:?}",
            current_memory,
            self.model_memory_mb,
            load_duration
        );
    }

    /// Update peak memory if current is higher
    fn update_peak(&self, current_mb: f64) {
        let current_bits = current_mb.to_bits();
        loop {
            let peak_bits = self.peak_memory_mb.load(Ordering::SeqCst);
            let peak_mb = f64::from_bits(peak_bits);
            if current_mb <= peak_mb {
                break;
            }
            if self
                .peak_memory_mb
                .compare_exchange(peak_bits, current_bits, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                break;
            }
        }
    }

    /// Start background memory sampling
    ///
    /// Returns a handle that must be kept alive until sampling should stop.
    pub fn start_sampling(&self) -> SamplingHandle {
        self.sampling_active.store(true, Ordering::SeqCst);

        let peak_memory = Arc::clone(&self.peak_memory_mb);
        let sampling_active = Arc::clone(&self.sampling_active);
        let pid = self.pid;

        // Spawn a background thread for sampling (not async to avoid blocking)
        let handle = std::thread::spawn(move || {
            let mut system = System::new();

            while sampling_active.load(Ordering::SeqCst) {
                // Refresh only our process (sysinfo 0.32+ API)
                system.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
                if let Some(process) = system.process(pid) {
                    let current_mb = process.memory() as f64 / (1024.0 * 1024.0);
                    let current_bits = current_mb.to_bits();

                    // Update peak atomically
                    loop {
                        let peak_bits = peak_memory.load(Ordering::SeqCst);
                        let peak_mb = f64::from_bits(peak_bits);
                        if current_mb <= peak_mb {
                            break;
                        }
                        if peak_memory
                            .compare_exchange(
                                peak_bits,
                                current_bits,
                                Ordering::SeqCst,
                                Ordering::SeqCst,
                            )
                            .is_ok()
                        {
                            break;
                        }
                    }
                }

                std::thread::sleep(Duration::from_millis(SAMPLE_INTERVAL_MS));
            }
        });

        SamplingHandle {
            sampling_active: Arc::clone(&self.sampling_active),
            _thread: Some(handle),
        }
    }

    /// Try to get GPU utilization (best effort, may not be available)
    fn try_get_gpu_utilization(&self) -> Option<f64> {
        // On macOS, GPU utilization requires powermetrics (sudo) or IOKit
        // This is a placeholder - would need platform-specific implementation
        #[cfg(target_os = "macos")]
        {
            // Could try parsing output of:
            // sudo powermetrics --samplers gpu_power -n 1
            // But requires elevated permissions
            None
        }
        #[cfg(not(target_os = "macos"))]
        {
            // On Linux with NVIDIA, could use nvml
            // On Windows, could use DXGI
            None
        }
    }

    /// Try to get GPU memory usage (best effort)
    fn try_get_gpu_memory_mb(&self) -> Option<f64> {
        if self.is_unified_memory {
            // On Apple Silicon, GPU uses unified memory
            // The process memory already includes GPU allocations
            return None;
        }

        // For discrete GPUs, would need:
        // - NVIDIA: nvml
        // - AMD: rocm-smi
        // - Intel: similar tooling
        None
    }

    /// Finalize metrics after embedding is complete
    pub fn finalize(
        mut self,
        embed_duration: Duration,
        embed_count: usize,
    ) -> ResourceMetrics {
        // Stop sampling
        self.sampling_active.store(false, Ordering::SeqCst);

        // Get final memory reading
        let final_memory = self.get_process_memory_mb();
        self.update_peak(final_memory);

        let peak_mb = f64::from_bits(self.peak_memory_mb.load(Ordering::SeqCst));
        let model_load_secs = self
            .model_load_time
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let embed_secs = embed_duration.as_secs_f64();
        let throughput = if embed_secs > 0.0 {
            embed_count as f64 / embed_secs
        } else {
            0.0
        };

        // Try to get GPU metrics (best effort)
        let gpu_utilization = self.try_get_gpu_utilization();
        let gpu_memory = self.try_get_gpu_memory_mb();

        // Generate notes about any limitations
        let notes = if self.is_unified_memory {
            Some("Apple Silicon unified memory - GPU memory included in process RAM".to_string())
        } else if gpu_utilization.is_none() && gpu_memory.is_none() {
            Some("GPU metrics unavailable - platform not supported".to_string())
        } else {
            None
        };

        ResourceMetrics {
            model_load_time_secs: model_load_secs,
            total_embed_time_secs: embed_secs,
            throughput_per_sec: throughput,
            baseline_memory_mb: self.baseline_memory_mb,
            model_memory_mb: self.model_memory_mb,
            peak_memory_mb: peak_mb,
            gpu_utilization_percent: gpu_utilization,
            gpu_memory_mb: gpu_memory,
            notes,
        }
    }
}

impl Default for ResourceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle for background memory sampling
///
/// Sampling continues while this handle is held. Drop to stop sampling.
pub struct SamplingHandle {
    sampling_active: Arc<AtomicBool>,
    _thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for SamplingHandle {
    fn drop(&mut self) {
        // Signal thread to stop
        self.sampling_active.store(false, Ordering::SeqCst);
        // Note: We don't join the thread here to avoid blocking
        // The thread will exit on next iteration when it sees sampling_active is false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_monitor_basic() {
        let mut monitor = ResourceMonitor::new();
        monitor.snapshot_baseline();

        assert!(monitor.baseline_memory_mb > 0.0);

        // Simulate model load
        monitor.record_model_loaded(Duration::from_millis(100));

        // Simulate embedding
        let _handle = monitor.start_sampling();
        std::thread::sleep(Duration::from_millis(200));

        let metrics = monitor.finalize(Duration::from_secs(1), 100);

        assert!(metrics.baseline_memory_mb > 0.0);
        assert!(metrics.peak_memory_mb >= metrics.baseline_memory_mb);
        assert_eq!(metrics.throughput_per_sec, 100.0);
    }
}
