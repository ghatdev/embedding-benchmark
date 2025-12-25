//! Chunking constants
//!
//! Constants for chunk sizing. The actual chunking implementations are in:
//! - `semantic_chunker.rs` - Semantic chunkers with context injection (recommended)

/// Maximum characters per chunk (industry standard: 1500 chars ≈ 375 tokens)
pub const DEFAULT_MAX_CHARS: usize = 1500;

/// Minimum characters for a valid chunk
pub const MIN_CHUNK_CHARS: usize = 50;
