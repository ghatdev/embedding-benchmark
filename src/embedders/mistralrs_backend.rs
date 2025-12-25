//! mistral.rs backend implementation
//!
//! Wraps the mistral.rs library for Candle-based embedding models with Metal support.

use anyhow::{Context, Result};
use mistralrs::{EmbeddingModelBuilder, EmbeddingRequest, IsqType, Model};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

use super::traits::{BatchEmbeddingResult, EmbedderBackend, EmbedderConfig, EmbeddingResult};

/// Supported mistral.rs embedding models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MistralRsModel {
    /// Google EmbeddingGemma 300M
    EmbeddingGemma300M,
    /// Qwen3 Embedding 0.6B
    Qwen3Embedding06B,
    /// Qwen3 Embedding 4B
    Qwen3Embedding4B,
    /// Qwen3 Embedding 8B
    Qwen3Embedding8B,
}

impl MistralRsModel {
    /// Get the HuggingFace model ID
    pub fn model_id(&self) -> &'static str {
        match self {
            Self::EmbeddingGemma300M => "google/embeddinggemma-300m",
            Self::Qwen3Embedding06B => "Qwen/Qwen3-Embedding-0.6B",
            Self::Qwen3Embedding4B => "Qwen/Qwen3-Embedding-4B",
            Self::Qwen3Embedding8B => "Qwen/Qwen3-Embedding-8B",
        }
    }

    /// Format text as a query for embedding (with instruction prefix)
    /// This is crucial for models that are instruction-aware!
    pub fn format_query(&self, text: &str) -> String {
        match self {
            // EmbeddingGemma: "task: search result | query: {text}"
            Self::EmbeddingGemma300M => {
                format!("task: search result | query: {}", text)
            }
            // Qwen3-Embedding: "Instruct: {task}\nQuery: {text}"
            Self::Qwen3Embedding06B | Self::Qwen3Embedding4B | Self::Qwen3Embedding8B => {
                format!(
                    "Instruct: Given a code search query, retrieve relevant code snippets that answer the query\nQuery: {}",
                    text
                )
            }
        }
    }

    /// Format text as a document for embedding
    pub fn format_document(&self, text: &str) -> String {
        match self {
            // EmbeddingGemma: "title: none | text: {text}"
            Self::EmbeddingGemma300M => {
                format!("title: none | text: {}", text)
            }
            // Qwen3-Embedding: documents don't need instruction prefix
            // Using a simple format for passages
            Self::Qwen3Embedding06B | Self::Qwen3Embedding4B | Self::Qwen3Embedding8B => {
                text.to_string()
            }
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::EmbeddingGemma300M => "EmbeddingGemma-300M",
            Self::Qwen3Embedding06B => "Qwen3-Embedding-0.6B",
            Self::Qwen3Embedding4B => "Qwen3-Embedding-4B",
            Self::Qwen3Embedding8B => "Qwen3-Embedding-8B",
        }
    }

    /// Get default vector dimensions
    pub fn dimensions(&self) -> usize {
        match self {
            Self::EmbeddingGemma300M => 768,
            Self::Qwen3Embedding06B => 1024,
            Self::Qwen3Embedding4B => 2560,
            Self::Qwen3Embedding8B => 4096,
        }
    }

    /// Get approximate model size in GB
    pub fn size_gb(&self) -> f32 {
        match self {
            Self::EmbeddingGemma300M => 0.6,
            Self::Qwen3Embedding06B => 1.2,
            Self::Qwen3Embedding4B => 8.0,
            Self::Qwen3Embedding8B => 16.0,
        }
    }

    /// List all available models
    pub fn all() -> Vec<Self> {
        vec![
            Self::EmbeddingGemma300M,
            Self::Qwen3Embedding06B,
            Self::Qwen3Embedding4B,
            Self::Qwen3Embedding8B,
        ]
    }

    /// List recommended models for benchmark (excludes very large models)
    pub fn recommended() -> Vec<Self> {
        vec![
            Self::EmbeddingGemma300M,
            Self::Qwen3Embedding06B,
        ]
    }
}

/// Quantization options for mistral.rs models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MistralRsQuantization {
    /// No quantization (F16 or BF16)
    None,
    /// 8-bit quantization
    Q8_0,
    /// 4-bit quantization (Q4K)
    Q4K,
    /// 4-bit quantization (Q4_0)
    Q4_0,
}

impl MistralRsQuantization {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "F16",
            Self::Q8_0 => "Q8_0",
            Self::Q4K => "Q4K",
            Self::Q4_0 => "Q4_0",
        }
    }

    /// Convert to IsqType (In-Situ Quantization)
    pub fn to_isq_type(&self) -> Option<IsqType> {
        match self {
            Self::None => None,
            Self::Q8_0 => Some(IsqType::Q8_0),
            Self::Q4K => Some(IsqType::Q4K),
            Self::Q4_0 => Some(IsqType::Q4_0),
        }
    }
}

/// Device options for mistral.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MistralRsDevice {
    /// CPU only
    Cpu,
    /// Metal GPU (Apple Silicon)
    Metal,
}

impl MistralRsDevice {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Metal => "Metal",
        }
    }
}

/// Maximum tokens for embedding models (2048 - buffer for instruction prefix)
const MAX_TOKENS: usize = 1800;

/// Approximate characters per token for code (conservative estimate)
const CHARS_PER_TOKEN: usize = 4;

/// Maximum characters to allow (based on token estimate)
const MAX_CHARS: usize = MAX_TOKENS * CHARS_PER_TOKEN;

/// Truncate text to fit within token limits
///
/// Uses a conservative character-based estimate (4 chars/token for code).
/// Actual tokenization would be more accurate but this is sufficient for chunked code.
fn truncate_to_limit(text: &str) -> String {
    if text.len() <= MAX_CHARS {
        return text.to_string();
    }

    // Truncate at character limit, try to break at line boundary
    let truncated = &text[..MAX_CHARS];

    // Find last newline to avoid cutting mid-line
    if let Some(last_newline) = truncated.rfind('\n') {
        if last_newline > MAX_CHARS / 2 {
            return truncated[..last_newline].to_string();
        }
    }

    truncated.to_string()
}

/// mistral.rs backend for Candle-based embeddings
pub struct MistralRsBackend {
    model: Arc<Mutex<Model>>,
    config: EmbedderConfig,
    model_type: MistralRsModel,
    quantization: MistralRsQuantization,
    device: MistralRsDevice,
    load_duration: Duration,
}

impl MistralRsBackend {
    /// Create a new mistral.rs backend with the specified configuration
    pub async fn new(
        model_type: MistralRsModel,
        quantization: MistralRsQuantization,
        device: MistralRsDevice,
    ) -> Result<Self> {
        tracing::info!(
            "Initializing mistral.rs model: {} ({} on {})",
            model_type.name(),
            quantization.name(),
            device.name()
        );

        let start = std::time::Instant::now();

        // Build the model with optional ISQ
        let mut builder = EmbeddingModelBuilder::new(model_type.model_id())
            .with_logging();

        // Apply quantization if specified
        if let Some(isq_type) = quantization.to_isq_type() {
            builder = builder.with_isq(isq_type);
        }

        let model = builder
            .build()
            .await
            .context(format!(
                "Failed to initialize mistral.rs model: {}",
                model_type.name()
            ))?;

        let load_duration = start.elapsed();

        tracing::info!(
            "mistral.rs model {} loaded in {:?}",
            model_type.name(),
            load_duration
        );

        let config = EmbedderConfig {
            name: format!(
                "mistralrs-{}-{}-{}",
                model_type.name(),
                quantization.name(),
                device.name()
            ),
            backend: "mistralrs".to_string(),
            model: model_type.model_id().to_string(),
            dimensions: model_type.dimensions(),
            quantization: match quantization {
                MistralRsQuantization::None => None,
                q => Some(q.name().to_string()),
            },
            device: device.name().to_string(),
            notes: Some(format!(
                "Candle runtime, load time: {:?}, ~{:.1}GB",
                load_duration,
                model_type.size_gb()
            )),
        };

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            config,
            model_type,
            quantization,
            device,
            load_duration,
        })
    }

    /// Get the model type
    pub fn model_type(&self) -> MistralRsModel {
        self.model_type
    }

    /// Get the quantization level
    pub fn quantization(&self) -> MistralRsQuantization {
        self.quantization
    }

    /// Get the device
    pub fn device(&self) -> MistralRsDevice {
        self.device
    }

    /// Get the model load duration
    pub fn load_duration(&self) -> Duration {
        self.load_duration
    }
}

#[async_trait::async_trait]
impl EmbedderBackend for MistralRsBackend {
    fn config(&self) -> &EmbedderConfig {
        &self.config
    }

    /// Embed a query (uses instruction prefix for better retrieval)
    async fn embed(&self, text: &str) -> Result<EmbeddingResult> {
        let start = std::time::Instant::now();

        // Truncate to fit within token limits before formatting
        let truncated_text = truncate_to_limit(text);

        // Format as query with instruction prefix
        let formatted_text = self.model_type.format_query(&truncated_text);

        let model = self.model.lock().await;
        let request = EmbeddingRequest::builder().add_prompt(&formatted_text);

        let embeddings = model
            .generate_embeddings(request)
            .await
            .context("Failed to generate embedding")?;

        let duration = start.elapsed();

        let embedding = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No embedding returned"))?;

        Ok(EmbeddingResult {
            embedding,
            duration,
        })
    }

    /// Embed documents/passages (uses document prefix)
    async fn embed_batch(&self, texts: &[String]) -> Result<BatchEmbeddingResult> {
        if texts.is_empty() {
            return Ok(BatchEmbeddingResult {
                embeddings: vec![],
                duration: Duration::ZERO,
                avg_duration: Duration::ZERO,
            });
        }

        let start = std::time::Instant::now();
        let count = texts.len();

        let model = self.model.lock().await;

        // Build request with all texts formatted as documents
        let mut request = EmbeddingRequest::builder();
        for text in texts {
            let formatted_text = self.model_type.format_document(text);
            request = request.add_prompt(&formatted_text);
        }

        let embeddings = model
            .generate_embeddings(request)
            .await
            .map_err(|e| {
                // Log the full error for debugging
                eprintln!("    ERROR: mistralrs embedding failed: {:?}", e);
                eprintln!("    Batch size: {}, first text len: {} chars",
                    texts.len(),
                    texts.first().map(|t| t.len()).unwrap_or(0));
                anyhow::anyhow!("Failed to generate batch embeddings: {}", e)
            })?;

        let duration = start.elapsed();
        let avg_duration = duration / count as u32;

        Ok(BatchEmbeddingResult {
            embeddings,
            duration,
            avg_duration,
        })
    }

    async fn warmup(&self) -> Result<Duration> {
        // Embed a simple text to warm up the model
        let result = self.embed("warmup text for model initialization").await?;
        Ok(result.duration)
    }

    fn load_duration(&self) -> Duration {
        self.load_duration
    }
}

/// Configuration presets for common benchmark scenarios
pub struct MistralRsPresets;

impl MistralRsPresets {
    /// All configurations to benchmark
    pub fn all_configs() -> Vec<(MistralRsModel, MistralRsQuantization, MistralRsDevice)> {
        vec![
            // EmbeddingGemma configurations
            (MistralRsModel::EmbeddingGemma300M, MistralRsQuantization::None, MistralRsDevice::Cpu),
            (MistralRsModel::EmbeddingGemma300M, MistralRsQuantization::None, MistralRsDevice::Metal),
            (MistralRsModel::EmbeddingGemma300M, MistralRsQuantization::Q8_0, MistralRsDevice::Metal),
            (MistralRsModel::EmbeddingGemma300M, MistralRsQuantization::Q4K, MistralRsDevice::Metal),

            // Qwen3-Embedding-0.6B configurations
            (MistralRsModel::Qwen3Embedding06B, MistralRsQuantization::None, MistralRsDevice::Cpu),
            (MistralRsModel::Qwen3Embedding06B, MistralRsQuantization::None, MistralRsDevice::Metal),
            (MistralRsModel::Qwen3Embedding06B, MistralRsQuantization::Q8_0, MistralRsDevice::Metal),
            (MistralRsModel::Qwen3Embedding06B, MistralRsQuantization::Q4K, MistralRsDevice::Metal),

            // Large models (optional, may require significant RAM)
            // (MistralRsModel::Qwen3Embedding4B, MistralRsQuantization::Q4K, MistralRsDevice::Metal),
            // (MistralRsModel::Qwen3Embedding8B, MistralRsQuantization::Q4K, MistralRsDevice::Metal),
        ]
    }

    /// Minimal set for quick testing
    pub fn quick_configs() -> Vec<(MistralRsModel, MistralRsQuantization, MistralRsDevice)> {
        vec![
            (MistralRsModel::EmbeddingGemma300M, MistralRsQuantization::None, MistralRsDevice::Metal),
            (MistralRsModel::Qwen3Embedding06B, MistralRsQuantization::Q4K, MistralRsDevice::Metal),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires model download
    async fn test_mistralrs_embedding_gemma() {
        let backend = MistralRsBackend::new(
            MistralRsModel::EmbeddingGemma300M,
            MistralRsQuantization::None,
            MistralRsDevice::Metal,
        )
        .await
        .unwrap();

        assert_eq!(backend.dimensions(), 768);

        let result = backend.embed("Hello, world!").await.unwrap();
        assert_eq!(result.embedding.len(), 768);
    }

    #[tokio::test]
    #[ignore] // Requires model download
    async fn test_mistralrs_qwen3_embedding() {
        let backend = MistralRsBackend::new(
            MistralRsModel::Qwen3Embedding06B,
            MistralRsQuantization::Q4K,
            MistralRsDevice::Metal,
        )
        .await
        .unwrap();

        assert_eq!(backend.dimensions(), 1024);

        let result = backend.embed("Hello, world!").await.unwrap();
        assert_eq!(result.embedding.len(), 1024);
    }
}
