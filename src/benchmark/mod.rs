//! Benchmark infrastructure
//!
//! Provides quality metrics for embedding evaluation.
//!
//! ## Usage
//!
//! Use the CLI for benchmarks (file-level and answer-level evaluation):
//!
//! ```bash
//! embedding-benchmark run --corpus ./path --queries ./queries.json --strategy semantic
//! ```
//!
//! ## Modules
//!
//! - `quality` - File-level and answer-level quality metrics (nDCG@10, MRR@10, Hit@K)

pub mod quality;

pub use quality::{
    // File-level metrics
    FileQualityMetrics, FileQueryMetrics, FileIndex,
    // Answer-level metrics
    AnswerMetrics, AnswerQueryMetrics,
    // Confidence metrics
    ModelConfidenceMetrics, QueryConfidenceMetrics, ConfidenceInterval,
    bootstrap_confidence_interval, std_dev, pearson_correlation,
    // Corpus stats
    CorpusStats,
    // Utilities
    cosine_similarity, aggregate_scored_chunks, normalize_path,
};
