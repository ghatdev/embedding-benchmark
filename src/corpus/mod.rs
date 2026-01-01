//! Corpus loading and chunking
//!
//! Loads code and documentation files for embedding benchmarks.
//!
//! ## Generic Usage (Recommended)
//!
//! ```rust,ignore
//! use corpus::{CorpusConfig, load_corpus_generic};
//! use crate::config::Strategy;
//!
//! let config = CorpusConfig::new()
//!     .with_extensions(vec!["rs", "py", "md"])
//!     .with_strategy(Strategy::Semantic);  // Use semantic chunking
//!
//! let corpus = load_corpus_generic(&path, "my-project", &config)?;
//! ```
//!
//! ## Chunking Strategies
//!
//! High-level strategies (recommended):
//! - **Line**: Simple line-count chunking (baseline)
//! - **Semantic**: Structure-aware chunking with context preservation
//!
//! The Semantic strategy uses:
//! - AstSemanticChunker for code (AST-based with class/function context injection)
//! - HeaderSemanticChunker for docs (header-based with breadcrumb injection)

pub mod loader;
pub mod chunker;
pub mod semantic_chunker;
pub mod bounded_chunker;

#[cfg(test)]
mod bounded_chunker_tests;

// Chunking constants

// Semantic chunkers (recommended)

// New generic API (recommended)
pub use loader::{
    Chunk, ContentType, CorpusConfig, ContextFormat,
    load_corpus_generic,
    load_sample_texts,
};

// Legacy API (deprecated, for backward compatibility)
