//! Corpus loader for embedding benchmarks
//!
//! Loads and chunks any codebase for benchmarking.
//!
//! ## Generic Corpus Loading
//!
//! ```rust,ignore
//! let config = CorpusConfig::new()
//!     .with_extensions(vec!["rs", "py", "md"])
//!     .with_chunk_lines(50)
//!     .with_overlap(0.25);
//!
//! let corpus = load_corpus_generic(&path, "my-project", &config)?;
//! ```

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::path::Path;

use super::bounded_chunker::{BoundedChunkerConfig, BoundedSemanticChunker};
use super::chunker::DEFAULT_MAX_CHARS;
use super::semantic_chunker::HeaderSemanticChunker;
use crate::config::Strategy;

/// A chunk of text for embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Unique identifier (hash of content)
    pub id: String,
    /// Source file path (relative)
    pub file_path: String,
    /// Content of the chunk
    pub content: String,
    /// Language (rust, markdown, etc.)
    pub language: String,
    /// Start line in source file
    pub start_line: usize,
    /// End line in source file
    pub end_line: usize,
    /// Content type (code or document)
    pub content_type: ContentType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContentType {
    Code,
    Document,
}

impl Chunk {
    /// Create a new chunk with auto-generated ID
    pub fn new(
        file_path: impl Into<String>,
        content: impl Into<String>,
        language: impl Into<String>,
        start_line: usize,
        end_line: usize,
        content_type: ContentType,
    ) -> Self {
        let content = content.into();
        let file_path = file_path.into();

        // Generate ID from content hash
        let mut hasher = Sha256::new();
        hasher.update(&file_path);
        hasher.update(&content);
        let id = format!("{:x}", hasher.finalize())[..16].to_string();

        Self {
            id,
            file_path,
            content,
            language: language.into(),
            start_line,
            end_line,
            content_type,
        }
    }
}

/// A corpus of chunks for embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Corpus {
    /// Name of the corpus
    pub name: String,
    /// Root directory
    pub root_path: String,
    /// All chunks
    pub chunks: Vec<Chunk>,
    /// Total files processed
    pub file_count: usize,
    /// Total lines of code/text
    pub total_lines: usize,
}

impl Corpus {
    /// Get code chunks only
    pub fn code_chunks(&self) -> Vec<&Chunk> {
        self.chunks
            .iter()
            .filter(|c| c.content_type == ContentType::Code)
            .collect()
    }

    /// Get document chunks only
    pub fn doc_chunks(&self) -> Vec<&Chunk> {
        self.chunks
            .iter()
            .filter(|c| c.content_type == ContentType::Document)
            .collect()
    }

    /// Get all chunk contents as strings (for embedding)
    pub fn texts(&self) -> Vec<String> {
        self.chunks.iter().map(|c| c.content.clone()).collect()
    }

    /// Sample N random chunks
    pub fn sample(&self, n: usize) -> Vec<&Chunk> {
        use std::collections::HashSet;

        let mut indices = HashSet::new();
        let max = self.chunks.len();

        while indices.len() < n.min(max) {
            indices.insert(rand_index(max));
        }

        indices.into_iter().map(|i| &self.chunks[i]).collect()
    }
}

fn rand_index(max: usize) -> usize {
    use std::time::SystemTime;
    let seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    (seed as usize) % max
}

// =============================================================================
// GENERIC CONFIGURATION (New: no hardcoding)
// =============================================================================

/// Configuration for corpus loading
///
/// Use the builder pattern to configure:
/// - Which file extensions to include
/// - Chunking parameters (lines, overlap)
/// - Chunking strategy (line-based, tree-sitter, header-based)
/// - Skip patterns
#[derive(Debug, Clone)]
pub struct CorpusConfig {
    /// File extensions to include (without dots, e.g., "rs", "py", "md")
    /// Empty means auto-detect based on common code extensions
    pub extensions: Vec<String>,

    /// Extensions treated as code (for metrics)
    pub code_extensions: HashSet<String>,

    /// Extensions treated as documentation
    pub doc_extensions: HashSet<String>,

    /// Chunking strategy: Line (baseline) or Semantic (context-aware)
    pub strategy: Strategy,

    /// Maximum lines per chunk for code files (line-based only)
    pub code_chunk_lines: usize,

    /// Maximum lines per chunk for documentation files (line-based only)
    pub doc_chunk_lines: usize,

    /// Maximum characters per chunk (semantic chunking)
    pub max_chunk_chars: usize,

    /// Context budget for semantic chunking (chars reserved for context injection)
    /// NOTE: Set to 0 for BoundedSemanticChunker (no context injection)
    pub context_budget: usize,

    /// Overlap ratio (0.0 to 0.5) for chunking
    pub overlap: f32,

    /// Minimum characters for a valid chunk
    pub min_chunk_chars: usize,

    /// Patterns to skip (substrings in path)
    pub skip_patterns: Vec<String>,

    /// Maximum tokens per chunk (for embedding model limits)
    /// Used by BoundedSemanticChunker to enforce hard limits
    pub max_tokens: usize,

    /// Whether to inject file path context into chunks (semantic chunking)
    pub inject_context: bool,

    /// Context format for injection
    pub context_format: ContextFormat,
}

/// Context injection format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum ContextFormat {
    /// No context injection
    #[default]
    None,
    /// Comment-style: `// path/to/file.rs`
    Comment,
    /// Bracketed: `[file: path/to/file.rs]`
    Bracket,
    /// Natural language: `File: path/to/file.rs`
    Natural,
    /// Full breadcrumb with function signature: `File: path.rs | fn foo()`
    Signature,
}

impl Default for CorpusConfig {
    fn default() -> Self {
        Self {
            extensions: vec![], // Empty = auto-detect
            code_extensions: ["rs", "py", "ts", "js", "tsx", "jsx", "go", "java", "c", "cpp", "h", "hpp", "rb", "php", "swift", "kt", "scala", "toml", "yaml", "yml", "json"]
                .into_iter().map(String::from).collect(),
            doc_extensions: ["md", "rst", "txt", "adoc"]
                .into_iter().map(String::from).collect(),
            strategy: Strategy::Line, // Default: line-based (baseline)
            code_chunk_lines: 50,
            doc_chunk_lines: 100,
            max_chunk_chars: DEFAULT_MAX_CHARS,
            context_budget: 0, // No context injection (research: improves retrieval)
            overlap: 0.25, // 25% overlap (match line chunker for fair comparison)
            min_chunk_chars: 50,
            skip_patterns: vec![
                "/target/".to_string(),
                "/node_modules/".to_string(),
                "/.git/".to_string(),
                "/__pycache__/".to_string(),
                "/.venv/".to_string(),
                "/venv/".to_string(),
                "/.tox/".to_string(),
                "/dist/".to_string(),
                "/build/".to_string(),
                "/.eggs/".to_string(),
                "/site-packages/".to_string(),
            ],
            max_tokens: 350, // Reduced for more chunks (research: 200-500 tokens optimal)
            inject_context: false, // Default: no context injection
            context_format: ContextFormat::None,
        }
    }
}

impl CorpusConfig {
    /// Create a new config with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set file extensions to include
    pub fn with_extensions(mut self, extensions: Vec<&str>) -> Self {
        self.extensions = extensions.into_iter().map(String::from).collect();
        self
    }

    /// Set chunk lines for code files
    pub fn with_code_chunk_lines(mut self, lines: usize) -> Self {
        self.code_chunk_lines = lines;
        self
    }

    /// Set chunk lines for documentation files
    pub fn with_doc_chunk_lines(mut self, lines: usize) -> Self {
        self.doc_chunk_lines = lines;
        self
    }

    /// Set overlap ratio (0.0 to 0.5)
    pub fn with_overlap(mut self, overlap: f32) -> Self {
        self.overlap = overlap.clamp(0.0, 0.5);
        self
    }

    /// Set maximum characters per chunk (for AST/header-based)
    pub fn with_max_chunk_chars(mut self, max_chars: usize) -> Self {
        self.max_chunk_chars = max_chars;
        self
    }

    /// Set context budget for semantic chunking
    /// NOTE: Set to 0 for best results (no context injection)
    pub fn with_context_budget(mut self, budget: usize) -> Self {
        self.context_budget = budget;
        self
    }

    /// Set maximum tokens per chunk (for embedding model limits)
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set whether to inject file path context into chunks
    pub fn with_inject_context(mut self, inject: bool) -> Self {
        self.inject_context = inject;
        self
    }

    /// Set context format for injection
    pub fn with_context_format(mut self, format: ContextFormat) -> Self {
        self.context_format = format;
        // Auto-enable inject_context when format is not None
        if format != ContextFormat::None {
            self.inject_context = true;
        }
        self
    }

    /// Set high-level strategy (Line or Semantic)
    ///
    /// When set, this overrides code_chunking/doc_chunking:
    /// - Line: Uses line-based chunking for both code and docs
    /// - Semantic: Uses AstSemanticChunker for code, HeaderSemanticChunker for docs
    pub fn with_strategy(mut self, strategy: Strategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Add skip patterns
    pub fn with_skip_patterns(mut self, patterns: Vec<&str>) -> Self {
        self.skip_patterns.extend(patterns.into_iter().map(String::from));
        self
    }

    /// Replace all skip patterns
    pub fn set_skip_patterns(mut self, patterns: Vec<&str>) -> Self {
        self.skip_patterns = patterns.into_iter().map(String::from).collect();
        self
    }

    /// Check if extension should be included
    pub fn should_include(&self, ext: &str) -> bool {
        if self.extensions.is_empty() {
            // Auto-detect: include if it's a known code or doc extension
            self.code_extensions.contains(ext) || self.doc_extensions.contains(ext)
        } else {
            self.extensions.iter().any(|e| e.eq_ignore_ascii_case(ext))
        }
    }

    /// Check if extension is code
    pub fn is_code(&self, ext: &str) -> bool {
        self.code_extensions.contains(ext)
    }

    /// Check if extension is documentation
    pub fn is_doc(&self, ext: &str) -> bool {
        self.doc_extensions.contains(ext)
    }

    /// Check if path should be skipped
    pub fn should_skip(&self, path_str: &str) -> bool {
        self.skip_patterns.iter().any(|pattern| path_str.contains(pattern))
    }
}

// =============================================================================
// LEGACY SUPPORT (Deprecated: for backward compatibility)
// =============================================================================

/// Corpus language type for loading appropriate files
///
/// **Deprecated**: Use `CorpusConfig` with `load_corpus_generic` instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorpusLanguage {
    /// Rust codebase (agentide, ripgrep)
    Rust,
    /// Python codebase (FastAPI)
    Python,
    /// Mixed languages
    Mixed,
}

/// Load a codebase as a corpus with language-specific file selection
pub fn load_corpus(root_path: &Path, name: &str, language: CorpusLanguage) -> Result<Corpus> {
    let mut chunks = Vec::new();
    let mut file_count = 0;
    let mut total_lines = 0;

    // Walk the codebase respecting .gitignore
    let walker = WalkBuilder::new(root_path)
        .hidden(true)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .build();

    for entry in walker {
        let entry = entry.context("Failed to read directory entry")?;
        let path = entry.path();

        if !path.is_file() {
            continue;
        }

        // Determine file type based on corpus language
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let (file_language, content_type) = match (language, ext) {
            // Rust codebase
            (CorpusLanguage::Rust, "rs") => ("rust", ContentType::Code),
            (CorpusLanguage::Rust, "md") => ("markdown", ContentType::Document),
            (CorpusLanguage::Rust, "toml") => ("toml", ContentType::Code),

            // Python codebase
            (CorpusLanguage::Python, "py") => ("python", ContentType::Code),
            (CorpusLanguage::Python, "md") => ("markdown", ContentType::Document),
            (CorpusLanguage::Python, "rst") => ("rst", ContentType::Document),
            (CorpusLanguage::Python, "txt") => ("text", ContentType::Document),

            // Mixed codebase (all supported types)
            (CorpusLanguage::Mixed, "rs") => ("rust", ContentType::Code),
            (CorpusLanguage::Mixed, "py") => ("python", ContentType::Code),
            (CorpusLanguage::Mixed, "md") => ("markdown", ContentType::Document),
            (CorpusLanguage::Mixed, "toml") => ("toml", ContentType::Code),
            (CorpusLanguage::Mixed, "json") => ("json", ContentType::Code),

            _ => continue, // Skip unsupported files
        };

        // Skip test fixtures, build dirs, etc.
        let path_str = path.to_string_lossy();
        if path_str.contains("/target/")
            || path_str.contains("/node_modules/")
            || path_str.contains("/.git/")
            || path_str.contains("/experiments/")
            || path_str.contains("/__pycache__/")
            || path_str.contains("/.venv/")
            || path_str.contains("/venv/")
            || path_str.contains("/.tox/")
            || path_str.contains("/dist/")
            || path_str.contains("/build/")
            || path_str.contains("/.eggs/")
            || path_str.contains("/site-packages/")
            // Skip test directories (don't benchmark on tests)
            || path_str.contains("/tests/")
            || path_str.contains("/test/")
            || path_str.contains("_test.py")
            || path_str.contains("_test.rs")
        {
            continue;
        }

        // Read file content
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue, // Skip binary/unreadable files
        };

        let lines: Vec<&str> = content.lines().collect();
        total_lines += lines.len();

        // Chunk the file
        let file_chunks = chunk_file(
            path.strip_prefix(root_path).unwrap_or(path),
            &content,
            file_language,
            content_type,
        );

        chunks.extend(file_chunks);
        file_count += 1;
    }

    Ok(Corpus {
        name: name.to_string(),
        root_path: root_path.to_string_lossy().to_string(),
        chunks,
        file_count,
        total_lines,
    })
}

/// Load the agentide codebase as a corpus (convenience function)
pub fn load_agentide_corpus(root_path: &Path) -> Result<Corpus> {
    load_corpus(root_path, "agentide", CorpusLanguage::Rust)
}

/// Load the ripgrep codebase as a corpus
pub fn load_ripgrep_corpus(root_path: &Path) -> Result<Corpus> {
    load_corpus(root_path, "ripgrep", CorpusLanguage::Rust)
}

/// Load the FastAPI codebase as a corpus
pub fn load_fastapi_corpus(root_path: &Path) -> Result<Corpus> {
    load_corpus(root_path, "fastapi", CorpusLanguage::Python)
}

// =============================================================================
// GENERIC LOADER (New: fully configurable)
// =============================================================================

/// Load any codebase as a corpus with full configuration
///
/// This is the recommended way to load a corpus. Use `CorpusConfig` to
/// configure file extensions, chunking, and skip patterns.
///
/// # Example
///
/// ```rust,ignore
/// let config = CorpusConfig::new()
///     .with_extensions(vec!["rs", "py", "md"])
///     .with_code_chunk_lines(50);
///
/// let corpus = load_corpus_generic(&path, "my-project", &config)?;
/// ```
pub fn load_corpus_generic(root_path: &Path, name: &str, config: &CorpusConfig) -> Result<Corpus> {
    let mut chunks = Vec::new();
    let mut file_count = 0;
    let mut total_lines = 0;

    // Walk the codebase respecting .gitignore
    let walker = WalkBuilder::new(root_path)
        .hidden(true)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .build();

    for entry in walker {
        let entry = entry.context("Failed to read directory entry")?;
        let path = entry.path();

        if !path.is_file() {
            continue;
        }

        let path_str = path.to_string_lossy();

        // Check skip patterns
        if config.should_skip(&path_str) {
            continue;
        }

        // Get extension
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        // Check if we should include this file
        if !config.should_include(ext) {
            continue;
        }

        // Determine content type
        let content_type = if config.is_doc(ext) {
            ContentType::Document
        } else {
            ContentType::Code
        };

        // Determine language for context extraction
        let file_language = extension_to_language(ext);

        // Read file content
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue, // Skip binary/unreadable files
        };

        let lines: Vec<&str> = content.lines().collect();
        total_lines += lines.len();

        // Chunk the file with config
        let file_chunks = chunk_file_with_config(
            path.strip_prefix(root_path).unwrap_or(path),
            &content,
            file_language,
            content_type,
            config,
        );

        chunks.extend(file_chunks);
        file_count += 1;
    }

    Ok(Corpus {
        name: name.to_string(),
        root_path: root_path.to_string_lossy().to_string(),
        chunks,
        file_count,
        total_lines,
    })
}

/// Map file extension to language name
fn extension_to_language(ext: &str) -> &'static str {
    match ext {
        "rs" => "rust",
        "py" => "python",
        "ts" | "tsx" => "typescript",
        "js" | "jsx" => "javascript",
        "go" => "go",
        "java" => "java",
        "c" | "h" => "c",
        "cpp" | "hpp" | "cc" => "cpp",
        "rb" => "ruby",
        "php" => "php",
        "swift" => "swift",
        "kt" => "kotlin",
        "scala" => "scala",
        "md" => "markdown",
        "rst" => "rst",
        "txt" => "text",
        "toml" => "toml",
        "yaml" | "yml" => "yaml",
        "json" => "json",
        _ => "unknown",
    }
}

/// Chunk a file with configurable parameters
fn chunk_file_with_config(
    file_path: &Path,
    content: &str,
    language: &str,
    content_type: ContentType,
    config: &CorpusConfig,
) -> Vec<Chunk> {
    let file_path_str = file_path.to_string_lossy().to_string();

    match config.strategy {
        Strategy::Line => {
            // Line-based chunking for everything (baseline)
            chunk_file_line_based(file_path, content, language, content_type, config)
        }
        Strategy::Semantic => {
            // Use BoundedSemanticChunker for code (research-based improvements)
            // - Configurable context injection (file path prefix)
            // - Hard token limit enforcement
            // - Configurable overlap
            // - AST-aware splitting with fallback chain
            match content_type {
                ContentType::Code => {
                    let chunker = BoundedSemanticChunker::new(BoundedChunkerConfig {
                        max_tokens: config.max_tokens,
                        target_tokens: (config.max_tokens as f32 * 0.8) as usize,
                        overlap_ratio: config.overlap,
                        inject_context: config.inject_context,
                        context_format: config.context_format,
                    });

                    // Use chunk_with_file_context when context format is not None
                    let bounded_chunks = if config.context_format != ContextFormat::None {
                        chunker.chunk_with_file_context(content, language, &file_path_str)
                    } else if config.inject_context {
                        // Legacy: inject_context flag uses Comment format
                        let legacy_chunker = BoundedSemanticChunker::new(BoundedChunkerConfig {
                            max_tokens: config.max_tokens,
                            target_tokens: (config.max_tokens as f32 * 0.8) as usize,
                            overlap_ratio: config.overlap,
                            inject_context: true,
                            context_format: ContextFormat::Comment,
                        });
                        legacy_chunker.chunk_with_file_context(content, language, &file_path_str)
                    } else {
                        chunker.chunk(content, language)
                    };

                    if bounded_chunks.is_empty() {
                        // Fall back to line-based if AST/semantic fails
                        tracing::debug!(
                            "BoundedSemanticChunker returned empty for {}, falling back to line-based",
                            file_path.display()
                        );
                        return chunk_file_line_based(file_path, content, language, content_type, config);
                    }

                    // Convert BoundedChunk to Chunk
                    bounded_chunks
                        .into_iter()
                        .map(|bc| Chunk::new(
                            &file_path_str,
                            bc.content,
                            language,
                            bc.start_line,
                            bc.end_line,
                            content_type,
                        ))
                        .collect()
                }
                ContentType::Document => {
                    // Use HeaderSemanticChunker for docs (header-based splitting)
                    if language == "markdown" || language == "rst" {
                        let chunker = HeaderSemanticChunker::new(
                            config.max_chunk_chars,
                            config.context_budget,
                        );
                        return chunker.create_chunks(file_path, content);
                    }
                    // Fall back to line-based for non-markdown docs
                    chunk_file_line_based(file_path, content, language, content_type, config)
                }
            }
        }
    }
}

/// Line-based chunking (legacy method)
fn chunk_file_line_based(
    file_path: &Path,
    content: &str,
    language: &str,
    content_type: ContentType,
    config: &CorpusConfig,
) -> Vec<Chunk> {
    let file_path_str = file_path.to_string_lossy().to_string();
    let lines: Vec<&str> = content.lines().collect();

    // Extract file-level context for better embedding
    let file_context = extract_file_context(content, language);

    // Build context prefix
    let context_prefix = match &file_context {
        Some(desc) => format!("[file: {}] {}\n\n", file_path_str, desc),
        None => format!("[file: {}]\n\n", file_path_str),
    };

    // Get chunk parameters from config
    let max_lines = match content_type {
        ContentType::Code => config.code_chunk_lines,
        ContentType::Document => config.doc_chunk_lines,
    };
    let overlap_lines = (max_lines as f32 * config.overlap) as usize;
    let min_chars = config.min_chunk_chars;

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < lines.len() {
        let end = (start + max_lines).min(lines.len());
        let chunk_lines = &lines[start..end];
        let raw_content = chunk_lines.join("\n");

        // Skip very short chunks
        if raw_content.len() >= min_chars {
            let chunk_content = format!("{}{}", context_prefix, raw_content);

            chunks.push(Chunk::new(
                &file_path_str,
                chunk_content,
                language,
                start + 1, // 1-indexed
                end,
                content_type,
            ));
        }

        // Move forward with overlap
        start += max_lines - overlap_lines;
    }

    // If no chunks were created, add the whole file as one chunk
    if chunks.is_empty() && content.len() >= min_chars {
        let chunk_content = format!("{}{}", context_prefix, content);
        chunks.push(Chunk::new(
            &file_path_str,
            chunk_content,
            language,
            1,
            lines.len(),
            content_type,
        ));
    }

    chunks
}

/// Extract module-level documentation from file content.
///
/// For Rust files, this extracts //! doc comments from the start.
/// For Python files, this extracts module docstrings (triple-quoted).
/// For Markdown, this extracts the first heading and paragraph.
fn extract_file_context(content: &str, language: &str) -> Option<String> {
    let lines: Vec<&str> = content.lines().collect();

    match language {
        "rust" => {
            // Extract //! module doc comments
            let doc_lines: Vec<&str> = lines
                .iter()
                .take_while(|line| line.starts_with("//!") || line.trim().is_empty())
                .filter(|line| line.starts_with("//!"))
                .map(|line| line.trim_start_matches("//!").trim())
                .collect();

            if doc_lines.is_empty() {
                None
            } else {
                Some(doc_lines.join(" "))
            }
        }
        "python" => {
            // Extract module docstring (first triple-quoted string after optional comments/imports)
            let mut in_docstring = false;
            let mut docstring_lines = Vec::new();

            for line in lines.iter().take(30) {
                let trimmed = line.trim();

                // Skip empty lines, comments, and encoding declarations at start
                if !in_docstring {
                    if trimmed.is_empty()
                        || trimmed.starts_with('#')
                        || trimmed.starts_with("from __future__")
                        || trimmed.starts_with("# -*-")
                    {
                        continue;
                    }

                    // Check for docstring start
                    if trimmed.starts_with("\"\"\"") || trimmed.starts_with("'''") {
                        in_docstring = true;
                        let quote = if trimmed.starts_with("\"\"\"") { "\"\"\"" } else { "'''" };
                        let after_start = trimmed.trim_start_matches(quote);

                        // Single-line docstring
                        if after_start.ends_with(quote) {
                            let content = after_start.trim_end_matches(quote).trim();
                            if !content.is_empty() {
                                return Some(content.to_string());
                            }
                            return None;
                        } else if !after_start.is_empty() {
                            docstring_lines.push(after_start);
                        }
                    } else {
                        // First non-docstring statement - no module docstring
                        break;
                    }
                } else {
                    // Inside docstring
                    if trimmed.ends_with("\"\"\"") || trimmed.ends_with("'''") {
                        let quote = if trimmed.ends_with("\"\"\"") { "\"\"\"" } else { "'''" };
                        let before_end = trimmed.trim_end_matches(quote).trim();
                        if !before_end.is_empty() {
                            docstring_lines.push(before_end);
                        }
                        break;
                    } else {
                        docstring_lines.push(trimmed);
                    }
                }
            }

            if docstring_lines.is_empty() {
                None
            } else {
                // Take first meaningful line(s) of docstring
                Some(
                    docstring_lines
                        .into_iter()
                        .take(3)
                        .collect::<Vec<_>>()
                        .join(" ")
                        .trim()
                        .to_string(),
                )
            }
        }
        "markdown" => {
            // Extract first heading
            lines
                .iter()
                .find(|line| line.starts_with('#'))
                .map(|line| line.trim_start_matches('#').trim().to_string())
        }
        _ => None,
    }
}

/// Chunk a file into smaller pieces for embedding
///
/// Each chunk is prefixed with file-level context (path + module description)
/// to help embedding models understand the chunk's purpose.
fn chunk_file(
    file_path: &Path,
    content: &str,
    language: &str,
    content_type: ContentType,
) -> Vec<Chunk> {
    let file_path_str = file_path.to_string_lossy().to_string();
    let lines: Vec<&str> = content.lines().collect();

    // Extract file-level context for better embedding
    let file_context = extract_file_context(content, language);

    // Build context prefix: [file: path/to/file.rs] Module description here
    let context_prefix = match &file_context {
        Some(desc) => format!("[file: {}] {}\n\n", file_path_str, desc),
        None => format!("[file: {}]\n\n", file_path_str),
    };

    // Configuration
    let max_lines = match content_type {
        ContentType::Code => 50,     // Smaller chunks for code
        ContentType::Document => 100, // Larger chunks for docs
    };
    let overlap_lines = max_lines / 4; // 25% overlap
    let min_chars = 50;

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < lines.len() {
        let end = (start + max_lines).min(lines.len());
        let chunk_lines = &lines[start..end];
        let raw_content = chunk_lines.join("\n");

        // Skip very short chunks
        if raw_content.len() >= min_chars {
            // Prepend context to chunk content for better embedding
            let chunk_content = format!("{}{}", context_prefix, raw_content);

            chunks.push(Chunk::new(
                &file_path_str,
                chunk_content,
                language,
                start + 1, // 1-indexed
                end,
                content_type,
            ));
        }

        // Move forward with overlap
        start += max_lines - overlap_lines;
    }

    // If no chunks were created, add the whole file as one chunk
    if chunks.is_empty() && content.len() >= min_chars {
        let chunk_content = format!("{}{}", context_prefix, content);
        chunks.push(Chunk::new(
            &file_path_str,
            chunk_content,
            language,
            1,
            lines.len(),
            content_type,
        ));
    }

    chunks
}

/// Load sample texts for latency benchmarks
pub fn load_sample_texts() -> Vec<String> {
    vec![
        "function to handle user authentication with JWT tokens".to_string(),
        "database connection pool management and configuration".to_string(),
        "error handling middleware for HTTP requests".to_string(),
        "JSON serialization and deserialization utilities".to_string(),
        "async runtime configuration with tokio".to_string(),
        "logging and tracing span management".to_string(),
        "unit test assertions and mocking".to_string(),
        "memory allocation and resource cleanup".to_string(),
        "file system operations and path handling".to_string(),
        "network socket communication protocol".to_string(),
        "configuration parsing from TOML files".to_string(),
        "command line argument parsing with clap".to_string(),
        "vector database embedding storage".to_string(),
        "semantic search implementation".to_string(),
        "LSP client server communication".to_string(),
        "MCP tool schema generation".to_string(),
        "HTTP route handler implementation".to_string(),
        "websocket connection management".to_string(),
        "SQL query builder and execution".to_string(),
        "cache invalidation strategies".to_string(),
        "rate limiting middleware".to_string(),
        "request validation and sanitization".to_string(),
        "response compression and streaming".to_string(),
        "session state management".to_string(),
        "background task scheduling".to_string(),
        "event sourcing and CQRS patterns".to_string(),
        "microservice communication patterns".to_string(),
        "circuit breaker implementation".to_string(),
        "distributed tracing correlation".to_string(),
        "health check endpoint implementation".to_string(),
        "metrics collection and reporting".to_string(),
        "feature flag evaluation".to_string(),
        "A/B testing framework".to_string(),
        "data migration scripts".to_string(),
        "index optimization strategies".to_string(),
        "query plan analysis".to_string(),
        "connection pooling tuning".to_string(),
        "memory leak detection".to_string(),
        "profiling and benchmarking".to_string(),
        "code coverage reporting".to_string(),
        "continuous integration pipeline".to_string(),
        "deployment automation scripts".to_string(),
        "infrastructure as code".to_string(),
        "container orchestration".to_string(),
        "service mesh configuration".to_string(),
        "API gateway routing".to_string(),
        "load balancer health checks".to_string(),
        "SSL certificate management".to_string(),
        "secrets management integration".to_string(),
        "audit logging implementation".to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_creation() {
        let chunk = Chunk::new("test.rs", "fn main() {}", "rust", 1, 1, ContentType::Code);
        assert!(!chunk.id.is_empty());
        assert_eq!(chunk.language, "rust");
    }

    #[test]
    fn test_sample_texts() {
        let texts = load_sample_texts();
        assert!(texts.len() >= 50);
    }
}
