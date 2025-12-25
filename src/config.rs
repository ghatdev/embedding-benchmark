//! Configuration for embedding benchmark
//!
//! Defines the models.yaml schema and Strategy enum.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Chunking strategy for the benchmark
///
/// - `Line`: Simple line-based chunking (baseline)
/// - `Semantic`: Structure-aware chunking with context preservation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Strategy {
    /// Line-based chunking (baseline)
    /// - Code: Fixed line count with overlap
    /// - Docs: Fixed line count with overlap
    /// - Other: Fixed line count with overlap
    #[default]
    Line,

    /// Semantic structure-aware chunking
    /// - Code: AST-based with class/function context injection
    /// - Docs: Header-based with breadcrumb injection
    /// - Other: Falls back to line-based
    Semantic,
}

impl Strategy {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Line => "line",
            Self::Semantic => "semantic",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "line" | "line-based" | "lines" => Some(Self::Line),
            "semantic" | "smart" | "ast" | "structure" => Some(Self::Semantic),
            _ => None,
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Line => "Line-based chunking with overlap (baseline)",
            Self::Semantic => "Structure-aware chunking with context preservation",
        }
    }
}

/// Models configuration loaded from YAML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    /// FastEmbed models to benchmark
    #[serde(default)]
    pub fastembed: Vec<FastEmbedModelConfig>,

    /// MistralRS models to benchmark
    #[serde(default)]
    pub mistralrs: Vec<MistralRsModelConfig>,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            fastembed: vec![
                FastEmbedModelConfig {
                    model: "BAAI/bge-small-en-v1.5".to_string(),
                    max_tokens: Some(512),
                },
            ],
            mistralrs: vec![],
        }
    }
}

impl ModelsConfig {
    /// Load config from TOML file
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read models config: {:?}", path))?;
        let config: Self = toml::from_str(&content)
            .with_context(|| format!("Failed to parse models config: {:?}", path))?;
        Ok(config)
    }

    /// Load from default location (./models.toml) or return defaults
    pub fn load_default() -> Result<Self> {
        let local_path = Path::new("models.toml");
        if local_path.exists() {
            return Self::load(local_path);
        }
        // Return default config
        Ok(Self::default())
    }

    /// Save config to TOML file
    pub fn save(&self, path: &Path) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Total number of models configured
    pub fn model_count(&self) -> usize {
        self.fastembed.len() + self.mistralrs.len()
    }

    /// Check if any models are configured
    pub fn is_empty(&self) -> bool {
        self.fastembed.is_empty() && self.mistralrs.is_empty()
    }
}

/// FastEmbed model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastEmbedModelConfig {
    /// Model ID (e.g., "BAAI/bge-small-en-v1.5")
    pub model: String,

    /// Maximum tokens (optional, defaults to model's max)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
}

/// MistralRS model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralRsModelConfig {
    /// Model ID or name
    pub model: String,

    /// Quantization level
    #[serde(default)]
    pub quantization: MistralRsQuantization,

    /// Device to use
    #[serde(default)]
    pub device: MistralRsDevice,
}

/// MistralRS quantization options
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MistralRsQuantization {
    /// No quantization (F16)
    #[default]
    None,
    /// 8-bit quantization
    Q8,
    /// 4-bit quantization (Q4K)
    Q4k,
}

/// MistralRS device options
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MistralRsDevice {
    /// CPU
    Cpu,
    /// Metal GPU (macOS)
    #[default]
    Metal,
    /// CUDA GPU
    Cuda,
}

/// Benchmark run configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRunConfig {
    /// Corpus path
    pub corpus: String,

    /// Queries file path
    pub queries: String,

    /// Chunking strategy
    pub strategy: Strategy,

    /// Maximum characters per chunk
    #[serde(default = "default_max_chars")]
    pub max_chars: usize,

    /// Context budget for semantic chunking (chars reserved for context)
    #[serde(default = "default_context_budget")]
    pub context_budget: usize,

    /// Line-based: lines per code chunk
    #[serde(default = "default_code_lines")]
    pub code_chunk_lines: usize,

    /// Line-based: lines per doc chunk
    #[serde(default = "default_doc_lines")]
    pub doc_chunk_lines: usize,

    /// Line-based: overlap ratio
    #[serde(default = "default_overlap")]
    pub overlap: f32,
}

fn default_max_chars() -> usize { 2000 }
fn default_context_budget() -> usize { 400 }
fn default_code_lines() -> usize { 40 }
fn default_doc_lines() -> usize { 60 }
fn default_overlap() -> f32 { 0.25 }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_from_str() {
        assert_eq!(Strategy::from_str("line"), Some(Strategy::Line));
        assert_eq!(Strategy::from_str("semantic"), Some(Strategy::Semantic));
        assert_eq!(Strategy::from_str("smart"), Some(Strategy::Semantic));
        assert_eq!(Strategy::from_str("invalid"), None);
    }

    #[test]
    fn test_models_config_default() {
        let config = ModelsConfig::default();
        assert!(!config.fastembed.is_empty());
        assert!(config.mistralrs.is_empty());
    }

    #[test]
    fn test_models_config_toml() {
        let toml_str = r#"
[[fastembed]]
model = "BAAI/bge-small-en-v1.5"
max_tokens = 512

[[fastembed]]
model = "thenlper/gte-small"

[[mistralrs]]
model = "embedding-gemma"
quantization = "q4k"
device = "metal"
"#;
        let config: ModelsConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.fastembed.len(), 2);
        assert_eq!(config.mistralrs.len(), 1);
    }
}
