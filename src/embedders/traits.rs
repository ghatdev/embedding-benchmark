//! Embedder trait abstraction
//!
//! Defines a common interface for all embedding backends, enabling fair benchmarking.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Resource usage metrics collected during embedding benchmark
///
/// Tracks memory, GPU, and throughput metrics for comparing model efficiency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// Time to load model into memory (seconds)
    pub model_load_time_secs: f64,

    /// Total time for all embeddings (seconds)
    pub total_embed_time_secs: f64,

    /// Embeddings generated per second
    pub throughput_per_sec: f64,

    /// Process memory before model load (MB)
    pub baseline_memory_mb: f64,

    /// Memory used by the model (after load - before load) (MB)
    pub model_memory_mb: f64,

    /// Peak process memory during embedding (MB)
    pub peak_memory_mb: f64,

    /// GPU utilization percentage during embedding (if available)
    /// Note: On macOS, requires elevated permissions - may be None
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_utilization_percent: Option<f64>,

    /// GPU memory usage in MB (for discrete GPUs)
    /// Note: Apple Silicon uses unified memory - this will be None
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_memory_mb: Option<f64>,

    /// Notes about resource monitoring limitations
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

impl ResourceMetrics {
    /// Format a summary line for display
    pub fn format_summary(&self) -> String {
        format!(
            "Load: {:.1}s | Embed: {:.1}s | Throughput: {:.1}/s | Peak RAM: {:.0}MB (model: {:.0}MB)",
            self.model_load_time_secs,
            self.total_embed_time_secs,
            self.throughput_per_sec,
            self.peak_memory_mb,
            self.model_memory_mb
        )
    }
}

/// Result of a single embedding operation with timing metadata
#[derive(Debug, Clone)]
pub struct EmbeddingResult {
    /// The embedding vector
    pub embedding: Vec<f32>,
    /// Time taken to generate the embedding
    pub duration: Duration,
}

/// Result of a batch embedding operation
#[derive(Debug, Clone)]
pub struct BatchEmbeddingResult {
    /// The embedding vectors
    pub embeddings: Vec<Vec<f32>>,
    /// Total time taken for the batch
    pub duration: Duration,
    /// Average time per embedding
    pub avg_duration: Duration,
}

/// Configuration for an embedder backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedderConfig {
    /// Human-readable name for this configuration
    pub name: String,
    /// Backend type (fastembed, mistralrs)
    pub backend: String,
    /// Model identifier
    pub model: String,
    /// Vector dimensions
    pub dimensions: usize,
    /// Quantization level (if applicable)
    pub quantization: Option<String>,
    /// Device (cpu, metal, cuda)
    pub device: String,
    /// Additional notes
    pub notes: Option<String>,
}

/// Unified trait for embedding backends
///
/// All embedding implementations must implement this trait to participate
/// in the benchmark. This ensures fair comparison across different backends.
#[async_trait::async_trait]
pub trait EmbedderBackend: Send + Sync {
    /// Get the configuration for this embedder
    fn config(&self) -> &EmbedderConfig;

    /// Get the name of this embedder configuration
    fn name(&self) -> &str {
        &self.config().name
    }

    /// Get the vector dimensions produced by this embedder
    fn dimensions(&self) -> usize {
        self.config().dimensions
    }

    /// Generate embedding for a single text
    ///
    /// Returns the embedding vector along with timing information.
    async fn embed(&self, text: &str) -> Result<EmbeddingResult>;

    /// Generate embeddings for a batch of texts
    ///
    /// Returns embedding vectors along with timing information.
    /// Implementations should optimize for batch processing where possible.
    async fn embed_batch(&self, texts: &[String]) -> Result<BatchEmbeddingResult>;

    /// Warm up the model (load weights, allocate buffers, etc.)
    ///
    /// Called before benchmarking to ensure cold start is measured separately.
    async fn warmup(&self) -> Result<Duration>;

    /// Get memory usage after model is loaded
    fn memory_usage_mb(&self) -> Option<f64> {
        None
    }

    /// Get model load duration
    fn load_duration(&self) -> Duration {
        Duration::ZERO
    }
}

/// Helper to measure duration of an async operation
pub async fn measure_async<F, T>(f: F) -> (T, Duration)
where
    F: std::future::Future<Output = T>,
{
    let start = std::time::Instant::now();
    let result = f.await;
    let duration = start.elapsed();
    (result, duration)
}

/// Helper to measure duration of a sync operation
pub fn measure_sync<F, T>(f: F) -> (T, Duration)
where
    F: FnOnce() -> T,
{
    let start = std::time::Instant::now();
    let result = f();
    let duration = start.elapsed();
    (result, duration)
}
