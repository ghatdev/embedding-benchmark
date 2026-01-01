//! FastEmbed backend implementation
//!
//! Wraps the fastembed-rs library for ONNX-based embedding models.

use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::traits::{BatchEmbeddingResult, EmbedderBackend, EmbedderConfig, EmbeddingResult, measure_sync};

/// Supported fastembed models for benchmarking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastEmbedModel {
    /// BAAI/bge-small-en-v1.5 (384 dims)
    BgeSmallEnV15,
    /// BAAI/bge-small-en-v1.5 quantized (384 dims)
    BgeSmallEnV15Q,
    /// BAAI/bge-base-en-v1.5 (768 dims)
    BgeBaseEnV15,
    /// BAAI/bge-large-en-v1.5 (1024 dims)
    BgeLargeEnV15,
    /// Google EmbeddingGemma 300M (768 dims)
    EmbeddingGemma300M,
    /// Jina embeddings v2 base code (768 dims)
    JinaEmbeddingsV2BaseCode,
    /// Alibaba GTE large en v1.5 (1024 dims)
    GteLargeEnV15,
    /// Nomic embed text v1.5 (768 dims)
    NomicEmbedTextV15,
    /// Snowflake Arctic Embed M (768 dims)
    SnowflakeArcticEmbedM,
    /// Snowflake Arctic Embed L (1024 dims) - larger version
    SnowflakeArcticEmbedL,
    /// intfloat/multilingual-e5-base (768 dims) - 100+ languages
    MultilingualE5Base,
    /// intfloat/multilingual-e5-large (1024 dims) - 100+ languages
    MultilingualE5Large,
}

impl FastEmbedModel {
    /// Convert to fastembed's EmbeddingModel enum
    pub fn to_fastembed_model(&self) -> EmbeddingModel {
        match self {
            Self::BgeSmallEnV15 => EmbeddingModel::BGESmallENV15,
            Self::BgeSmallEnV15Q => EmbeddingModel::BGESmallENV15Q,
            Self::BgeBaseEnV15 => EmbeddingModel::BGEBaseENV15,
            Self::BgeLargeEnV15 => EmbeddingModel::BGELargeENV15,
            Self::EmbeddingGemma300M => EmbeddingModel::EmbeddingGemma300M,
            Self::JinaEmbeddingsV2BaseCode => EmbeddingModel::JinaEmbeddingsV2BaseCode,
            Self::GteLargeEnV15 => EmbeddingModel::GTELargeENV15,
            Self::NomicEmbedTextV15 => EmbeddingModel::NomicEmbedTextV15,
            Self::SnowflakeArcticEmbedM => EmbeddingModel::SnowflakeArcticEmbedM,
            Self::SnowflakeArcticEmbedL => EmbeddingModel::SnowflakeArcticEmbedL,
            Self::MultilingualE5Base => EmbeddingModel::MultilingualE5Base,
            Self::MultilingualE5Large => EmbeddingModel::MultilingualE5Large,
        }
    }

    /// Format text as a query for embedding (with instruction prefix where needed)
    /// This is crucial for models that are instruction-aware!
    pub fn format_query(&self, text: &str) -> String {
        match self {
            // BGE models: "Represent this sentence for searching relevant passages: {text}"
            Self::BgeSmallEnV15 | Self::BgeSmallEnV15Q | Self::BgeBaseEnV15 | Self::BgeLargeEnV15 => {
                format!("Represent this sentence for searching relevant passages: {}", text)
            }
            // EmbeddingGemma: "task: search result | query: {text}"
            // Note: "search result" outperforms "code retrieval" in ablation tests
            Self::EmbeddingGemma300M => {
                format!("task: search result | query: {}", text)
            }
            // Nomic: "search_query: {text}"
            Self::NomicEmbedTextV15 => {
                format!("search_query: {}", text)
            }
            // Snowflake Arctic: same as BGE
            Self::SnowflakeArcticEmbedM | Self::SnowflakeArcticEmbedL => {
                format!("Represent this sentence for searching relevant passages: {}", text)
            }
            // Multilingual E5: "query: {text}"
            Self::MultilingualE5Base | Self::MultilingualE5Large => {
                format!("query: {}", text)
            }
            // Jina-code and GTE: no prefix needed
            Self::JinaEmbeddingsV2BaseCode | Self::GteLargeEnV15 => {
                text.to_string()
            }
        }
    }

    /// Format text as a document for embedding (with prefix where needed)
    pub fn format_document(&self, text: &str) -> String {
        match self {
            // EmbeddingGemma: "title: none | text: {text}"
            Self::EmbeddingGemma300M => {
                format!("title: none | text: {}", text)
            }
            // Nomic: "search_document: {text}"
            Self::NomicEmbedTextV15 => {
                format!("search_document: {}", text)
            }
            // Multilingual E5: "passage: {text}"
            Self::MultilingualE5Base | Self::MultilingualE5Large => {
                format!("passage: {}", text)
            }
            // All other models: no document prefix needed
            _ => text.to_string(),
        }
    }

    /// Get the vector dimensions for this model
    pub fn dimensions(&self) -> usize {
        match self {
            Self::BgeSmallEnV15 | Self::BgeSmallEnV15Q => 384,
            Self::BgeBaseEnV15 => 768,
            Self::BgeLargeEnV15 => 1024,
            Self::EmbeddingGemma300M => 768,
            Self::JinaEmbeddingsV2BaseCode => 768,
            Self::GteLargeEnV15 => 1024,
            Self::NomicEmbedTextV15 => 768,
            Self::SnowflakeArcticEmbedM => 768,
            Self::SnowflakeArcticEmbedL => 1024,
            Self::MultilingualE5Base => 768,
            Self::MultilingualE5Large => 1024,
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::BgeSmallEnV15 => "BGE-small-en-v1.5",
            Self::BgeSmallEnV15Q => "BGE-small-en-v1.5-Q",
            Self::BgeBaseEnV15 => "BGE-base-en-v1.5",
            Self::BgeLargeEnV15 => "BGE-large-en-v1.5",
            Self::EmbeddingGemma300M => "EmbeddingGemma-300M",
            Self::JinaEmbeddingsV2BaseCode => "Jina-embeddings-v2-base-code",
            Self::GteLargeEnV15 => "GTE-large-en-v1.5",
            Self::NomicEmbedTextV15 => "Nomic-embed-text-v1.5",
            Self::SnowflakeArcticEmbedM => "Snowflake-Arctic-Embed-M",
            Self::SnowflakeArcticEmbedL => "Snowflake-Arctic-Embed-L",
            Self::MultilingualE5Base => "Multilingual-E5-Base",
            Self::MultilingualE5Large => "Multilingual-E5-Large",
        }
    }

    /// Get model identifier (for config)
    pub fn model_id(&self) -> &'static str {
        match self {
            Self::BgeSmallEnV15 => "BAAI/bge-small-en-v1.5",
            Self::BgeSmallEnV15Q => "BAAI/bge-small-en-v1.5-Q",
            Self::BgeBaseEnV15 => "BAAI/bge-base-en-v1.5",
            Self::BgeLargeEnV15 => "BAAI/bge-large-en-v1.5",
            Self::EmbeddingGemma300M => "onnx-community/embeddinggemma-300m-ONNX",
            Self::JinaEmbeddingsV2BaseCode => "jinaai/jina-embeddings-v2-base-code",
            Self::GteLargeEnV15 => "Alibaba-NLP/gte-large-en-v1.5",
            Self::NomicEmbedTextV15 => "nomic-ai/nomic-embed-text-v1.5",
            Self::SnowflakeArcticEmbedM => "Snowflake/snowflake-arctic-embed-m",
            Self::SnowflakeArcticEmbedL => "snowflake/snowflake-arctic-embed-l",
            Self::MultilingualE5Base => "intfloat/multilingual-e5-base",
            Self::MultilingualE5Large => "Qdrant/multilingual-e5-large-onnx",
        }
    }

    /// Check if this is a quantized model
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::BgeSmallEnV15Q)
    }

    /// List all available models
    pub fn all() -> Vec<Self> {
        vec![
            Self::BgeSmallEnV15,
            Self::BgeSmallEnV15Q,
            Self::BgeBaseEnV15,
            Self::BgeLargeEnV15,
            Self::EmbeddingGemma300M,
            Self::JinaEmbeddingsV2BaseCode,
            Self::GteLargeEnV15,
            Self::NomicEmbedTextV15,
            Self::SnowflakeArcticEmbedM,
            Self::SnowflakeArcticEmbedL,
            Self::MultilingualE5Base,
            Self::MultilingualE5Large,
        ]
    }

    /// List recommended models for benchmark (smaller set)
    pub fn recommended() -> Vec<Self> {
        vec![
            Self::BgeSmallEnV15,
            Self::BgeSmallEnV15Q,
            Self::EmbeddingGemma300M,
            Self::JinaEmbeddingsV2BaseCode,
            Self::GteLargeEnV15,
        ]
    }
}

/// FastEmbed backend for ONNX-based embeddings
pub struct FastEmbedBackend {
    model: Arc<Mutex<TextEmbedding>>,
    config: EmbedderConfig,
    model_type: FastEmbedModel,
    load_duration: Duration,
}

impl FastEmbedBackend {
    /// Create a new FastEmbed backend with the specified model
    pub fn new(model_type: FastEmbedModel) -> Result<Self> {
        tracing::info!("Initializing FastEmbed model: {}", model_type.name());

        let (model, load_duration) = measure_sync(|| {
            let init_options = InitOptions::new(model_type.to_fastembed_model())
                .with_show_download_progress(true);

            TextEmbedding::try_new(init_options)
        });

        let model = model.context(format!("Failed to initialize FastEmbed model: {}", model_type.name()))?;

        tracing::info!(
            "FastEmbed model {} loaded in {:?}",
            model_type.name(),
            load_duration
        );

        let config = EmbedderConfig {
            name: format!("fastembed-{}", model_type.name()),
            backend: "fastembed".to_string(),
            model: model_type.model_id().to_string(),
            dimensions: model_type.dimensions(),
            quantization: if model_type.is_quantized() {
                Some("INT8".to_string())
            } else {
                None
            },
            device: "cpu".to_string(),
            notes: Some(format!("ONNX runtime, load time: {:?}", load_duration)),
        };

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            config,
            model_type,
            load_duration,
        })
    }

    /// Get the model type
    pub fn model_type(&self) -> FastEmbedModel {
        self.model_type
    }

    /// Get the model load duration
    pub fn load_duration(&self) -> Duration {
        self.load_duration
    }
}

#[async_trait::async_trait]
impl EmbedderBackend for FastEmbedBackend {
    fn config(&self) -> &EmbedderConfig {
        &self.config
    }

    /// Embed a query (uses instruction prefix for better retrieval)
    async fn embed(&self, text: &str) -> Result<EmbeddingResult> {
        // Format as query with instruction prefix
        let formatted_text = self.model_type.format_query(text);
        let model = Arc::clone(&self.model);

        // Run in blocking task since fastembed is synchronous
        let (result, duration) = tokio::task::spawn_blocking(move || {
            let start = std::time::Instant::now();
            let mut guard = model.lock().unwrap();
            let embeddings = guard.embed(vec![&formatted_text], None);
            let duration = start.elapsed();
            (embeddings, duration)
        })
        .await?;

        let embeddings = result.context("Failed to generate embedding")?;
        let embedding = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No embedding returned"))?;

        Ok(EmbeddingResult {
            embedding,
            duration,
        })
    }

    /// Embed documents/passages (uses document prefix where needed)
    async fn embed_batch(&self, texts: &[String]) -> Result<BatchEmbeddingResult> {
        if texts.is_empty() {
            return Ok(BatchEmbeddingResult {
                embeddings: vec![],
                duration: Duration::ZERO,
                avg_duration: Duration::ZERO,
            });
        }

        // Format all texts as documents
        let formatted_texts: Vec<String> = texts
            .iter()
            .map(|t| self.model_type.format_document(t))
            .collect();
        let count = formatted_texts.len();
        let model = Arc::clone(&self.model);

        // Run in blocking task since fastembed is synchronous
        let (result, duration) = tokio::task::spawn_blocking(move || {
            let start = std::time::Instant::now();
            let mut guard = model.lock().unwrap();
            let text_refs: Vec<&str> = formatted_texts.iter().map(|s| s.as_str()).collect();
            let embeddings = guard.embed(text_refs, None);
            let duration = start.elapsed();
            (embeddings, duration)
        })
        .await?;

        let embeddings = result.context("Failed to generate batch embeddings")?;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires model download
    async fn test_fastembed_bge_small() {
        let backend = FastEmbedBackend::new(FastEmbedModel::BgeSmallEnV15).unwrap();
        assert_eq!(backend.dimensions(), 384);

        let result = backend.embed("Hello, world!").await.unwrap();
        assert_eq!(result.embedding.len(), 384);
    }

    #[tokio::test]
    #[ignore] // Requires model download
    async fn test_fastembed_batch() {
        let backend = FastEmbedBackend::new(FastEmbedModel::BgeSmallEnV15).unwrap();

        let texts = vec![
            "First text".to_string(),
            "Second text".to_string(),
            "Third text".to_string(),
        ];

        let result = backend.embed_batch(&texts).await.unwrap();
        assert_eq!(result.embeddings.len(), 3);
    }
}
