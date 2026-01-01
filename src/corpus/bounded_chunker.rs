//! Bounded Semantic Chunker
//!
//! A semantic chunker that enforces hard token limits while preserving
//! code structure boundaries. Based on research from:
//! - cAST (EMNLP 2025): AST-based chunking with split-merge algorithm
//! - LangChain: Recursive fallback chain (AST → Line → Word → Truncate)
//! - OpenAI Cookbook: Weighted average for long text embeddings
//!
//! ## Key Features
//!
//! 1. **Hard Limit Enforcement**: Chunks NEVER exceed max_tokens
//! 2. **AST-Aware Splitting**: Prefers semantic boundaries (functions, classes)
//! 3. **Fallback Chain**: AST → Line → Word → Token truncation
//! 4. **No Context Injection**: Pure code, metadata stored separately
//! 5. **Configurable Overlap**: For boundary context preservation
//! 6. **Multi-Vector Support**: For monster lines without truncation

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tree_sitter::{Language, Node, Parser};

use super::loader::ContextFormat;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for the bounded semantic chunker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundedChunkerConfig {
    /// Hard maximum tokens - NEVER exceed this
    pub max_tokens: usize,

    /// Target tokens (typically 80% of max) - aim for this
    pub target_tokens: usize,

    /// Overlap ratio (0.0 to 0.5) between adjacent chunks
    pub overlap_ratio: f32,

    /// Whether to inject context prefixes (should be false for best results)
    pub inject_context: bool,

    /// Context format for injection
    #[serde(default)]
    pub context_format: ContextFormat,
}

impl Default for BoundedChunkerConfig {
    fn default() -> Self {
        // Reduced defaults based on research: 1000-1200 chars (~250-300 tokens)
        // yields better retrieval performance than larger chunks
        // 25% overlap matches line chunker for fair comparison
        Self {
            max_tokens: 350,     // ~1400 chars
            target_tokens: 280,  // ~1120 chars (80% of max)
            overlap_ratio: 0.25, // Match line chunker's 25% overlap
            inject_context: false,
            context_format: ContextFormat::None,
        }
    }
}

impl BoundedChunkerConfig {
    /// Create a small chunk configuration optimized for retrieval
    /// Based on research showing 200-500 tokens is optimal
    pub fn small() -> Self {
        Self {
            max_tokens: 350,
            target_tokens: 280,
            overlap_ratio: 0.25,
            inject_context: false,
            context_format: ContextFormat::None,
        }
    }

    /// Create an aggressive configuration for maximum chunk count
    /// Produces more, smaller chunks with high overlap - closest to line chunker behavior
    pub fn aggressive() -> Self {
        Self {
            max_tokens: 200,     // ~800 chars - force more splitting
            target_tokens: 160,  // ~640 chars (80% of max)
            overlap_ratio: 0.25, // Match line chunker
            inject_context: false,
            context_format: ContextFormat::None,
        }
    }
}

impl BoundedChunkerConfig {
    /// Create config for a specific embedding model
    pub fn for_model(model_max_tokens: usize) -> Self {
        Self {
            max_tokens: model_max_tokens,
            target_tokens: (model_max_tokens as f32 * 0.8) as usize,
            overlap_ratio: 0.25, // Match line chunker
            inject_context: false,
            context_format: ContextFormat::None,
        }
    }

    /// Set overlap ratio (clamped to 0.0-0.5)
    pub fn with_overlap(mut self, ratio: f32) -> Self {
        self.overlap_ratio = ratio.clamp(0.0, 0.5);
        self
    }
}

// ============================================================================
// CHUNK OUTPUT TYPES
// ============================================================================

/// A chunk of code with position information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundedChunk {
    /// Unique identifier
    pub id: String,

    /// The actual content (pure code, no injected context)
    pub content: String,

    /// Language of the content
    pub language: String,

    /// Start line in source file (1-indexed)
    pub start_line: usize,

    /// End line in source file (1-indexed)
    pub end_line: usize,

    /// Start byte offset in source
    pub start_byte: usize,

    /// End byte offset in source
    pub end_byte: usize,

    /// Optional metadata (stored separately from content)
    pub metadata: Option<ChunkMetadata>,

    /// Parent ID for multi-vector chunks
    pub parent_id: Option<String>,
}

impl BoundedChunk {
    fn new(content: String, language: &str, start_line: usize, end_line: usize, start_byte: usize, end_byte: usize) -> Self {
        let id = Self::generate_id(&content, start_byte);
        Self {
            id,
            content,
            language: language.to_string(),
            start_line,
            end_line,
            start_byte,
            end_byte,
            metadata: None,
            parent_id: None,
        }
    }

    fn generate_id(content: &str, offset: usize) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        hasher.update(offset.to_le_bytes());
        format!("{:x}", hasher.finalize())[..16].to_string()
    }

    fn with_metadata(mut self, metadata: ChunkMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    fn with_parent(mut self, parent_id: &str) -> Self {
        self.parent_id = Some(parent_id.to_string());
        self
    }
}

/// Metadata stored separately from chunk content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Source file path
    pub file_path: String,

    /// Function/method context (e.g., "impl Foo > fn bar")
    pub function_context: Option<String>,

    /// Class/module context
    pub class_context: Option<String>,

    /// AST node type that was chunked
    pub node_type: Option<String>,
}

/// Result of multi-vector chunking (for monster lines)
#[derive(Debug, Clone)]
pub struct MultiVectorResult {
    /// Individual chunks, all linked to parent
    pub chunks: Vec<BoundedChunk>,

    /// Parent ID that links all chunks
    pub parent_id: String,

    /// Original content hash for deduplication
    pub content_hash: String,
}

/// Result of weighted average embedding
#[derive(Debug, Clone)]
pub struct WeightedAverageResult {
    /// Combined embedding vector
    pub embedding: Vec<f32>,

    /// Number of chunks that were averaged
    pub chunk_count: usize,

    /// Weights used for each chunk
    pub weights: Vec<f32>,
}

// ============================================================================
// MAIN CHUNKER
// ============================================================================

/// Bounded semantic chunker with hard token limit enforcement
pub struct BoundedSemanticChunker {
    config: BoundedChunkerConfig,
}

impl BoundedSemanticChunker {
    /// Create a new chunker with the given configuration
    pub fn new(config: BoundedChunkerConfig) -> Self {
        Self { config }
    }

    /// Create a chunker for a specific embedding model's token limit
    pub fn for_model(model_max_tokens: usize) -> Self {
        Self::new(BoundedChunkerConfig::for_model(model_max_tokens))
    }

    // ========================================================================
    // MAIN CHUNKING API
    // ========================================================================

    /// Chunk content with hard limit enforcement
    ///
    /// Returns chunks that NEVER exceed max_tokens.
    /// Uses fallback chain: AST → Line → Word → Truncate
    pub fn chunk(&self, content: &str, language: &str) -> Vec<BoundedChunk> {
        // Handle empty/whitespace content
        let trimmed = content.trim();
        if trimmed.is_empty() {
            return Vec::new();
        }

        // Check if content fits in a single chunk
        let total_tokens = self.count_tokens(trimmed);
        if total_tokens <= self.config.target_tokens {
            let lines: Vec<&str> = trimmed.lines().collect();
            return vec![BoundedChunk::new(
                trimmed.to_string(),
                language,
                1,
                lines.len().max(1),
                0,
                trimmed.len(),
            )];
        }

        // Try AST-based splitting first
        if let Some(chunks) = self.split_at_ast(content, language) {
            if !chunks.is_empty() {
                let chunks = self.apply_overlap(chunks, content);
                return chunks;
            }
        }

        // Fallback to line-based splitting
        let chunks = self.split_at_lines(content, language);
        let chunks = self.apply_overlap(chunks, content);
        chunks
    }

    /// Chunk with explicit file path for metadata
    pub fn chunk_with_metadata(
        &self,
        content: &str,
        language: &str,
        file_path: &str,
    ) -> Vec<BoundedChunk> {
        let chunks = self.chunk(content, language);
        chunks
            .into_iter()
            .map(|c| {
                c.with_metadata(ChunkMetadata {
                    file_path: file_path.to_string(),
                    function_context: None,
                    class_context: None,
                    node_type: None,
                })
            })
            .collect()
    }

    /// Chunk with position tracking (alias for chunk, positions always tracked)
    pub fn chunk_with_positions(&self, content: &str, language: &str) -> Vec<BoundedChunk> {
        self.chunk(content, language)
    }

    /// Chunk with minimal file path context injection
    ///
    /// When `inject_context` is enabled, adds a comment with the file path
    /// at the start of each chunk. This provides minimal context without
    /// the overhead of full breadcrumb injection.
    ///
    /// Example output:
    /// ```text
    /// // src/auth.rs
    /// fn login() { ... }
    /// ```
    pub fn chunk_with_file_context(
        &self,
        content: &str,
        language: &str,
        file_path: &str,
    ) -> Vec<BoundedChunk> {
        let chunks = self.chunk(content, language);

        if !self.config.inject_context {
            // Context injection disabled, return pure code chunks
            return chunks
                .into_iter()
                .map(|c| {
                    c.with_metadata(ChunkMetadata {
                        file_path: file_path.to_string(),
                        function_context: None,
                        class_context: None,
                        node_type: None,
                    })
                })
                .collect();
        }

        // For Signature format, we need to extract signature per-chunk
        let use_signature = matches!(self.config.context_format, ContextFormat::Signature);

        chunks
            .into_iter()
            .map(|mut chunk| {
                // Extract signature if using Signature format
                let signature = if use_signature {
                    self.extract_signature(&chunk.content, language)
                } else {
                    None
                };

                // Format context with optional signature
                let context_line = self.format_context_with_signature(
                    language,
                    file_path,
                    signature.as_deref(),
                );
                let context_tokens = self.count_tokens(&context_line);

                // Check if adding context would exceed max
                let chunk_tokens = self.count_tokens(&chunk.content);
                if chunk_tokens + context_tokens <= self.config.max_tokens {
                    // Safe to add context
                    chunk.content = format!("{}{}", context_line, chunk.content);
                } else {
                    // Need to trim chunk to make room for context
                    let available_tokens = self.config.max_tokens.saturating_sub(context_tokens);
                    let trimmed_content = self.trim_to_tokens(&chunk.content, available_tokens);
                    chunk.content = format!("{}{}", context_line, trimmed_content);
                }

                chunk.with_metadata(ChunkMetadata {
                    file_path: file_path.to_string(),
                    function_context: signature.clone(),
                    class_context: None,
                    node_type: None,
                })
            })
            .collect()
    }

    /// Format context line based on configured format
    fn format_context_line(&self, language: &str, file_path: &str) -> String {
        self.format_context_with_signature(language, file_path, None)
    }

    /// Format context line with optional function signature
    fn format_context_with_signature(&self, language: &str, file_path: &str, signature: Option<&str>) -> String {
        match self.config.context_format {
            ContextFormat::None => String::new(),
            ContextFormat::Comment => {
                let comment_prefix = Self::get_comment_prefix(language);
                format!("{} {}\n", comment_prefix, file_path)
            }
            ContextFormat::Bracket => {
                format!("[file: {}]\n\n", file_path)
            }
            ContextFormat::Natural => {
                format!("File: {}\n\n", file_path)
            }
            ContextFormat::Signature => {
                match signature {
                    Some(sig) => format!("File: {} | {}\n\n", file_path, sig),
                    None => format!("File: {}\n\n", file_path),
                }
            }
        }
    }

    /// Extract function/class signature from chunk content using AST
    fn extract_signature(&self, content: &str, language: &str) -> Option<String> {
        let ts_language = Self::get_tree_sitter_language(language)?;

        let mut parser = Parser::new();
        parser.set_language(&ts_language).ok()?;

        let tree = parser.parse(content, None)?;
        let root = tree.root_node();

        // Find the first function/class/impl definition
        self.find_first_definition(root, content, language)
    }

    /// Find the first function/class/impl definition in AST
    fn find_first_definition(&self, node: Node, content: &str, language: &str) -> Option<String> {
        let kind = node.kind();

        // Check if this node is a definition we care about
        let signature = match language {
            "rust" => self.extract_rust_signature(node, content, kind),
            "python" | "py" => self.extract_python_signature(node, content, kind),
            "javascript" | "typescript" | "js" | "ts" | "jsx" | "tsx" =>
                self.extract_js_signature(node, content, kind),
            "go" => self.extract_go_signature(node, content, kind),
            _ => None,
        };

        if signature.is_some() {
            return signature;
        }

        // Recurse into children
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if let Some(sig) = self.find_first_definition(child, content, language) {
                return Some(sig);
            }
        }

        None
    }

    /// Extract Rust function/impl signature
    fn extract_rust_signature(&self, node: Node, content: &str, kind: &str) -> Option<String> {
        match kind {
            "function_item" => {
                // Find function name and params
                let mut name = None;
                let mut params = None;
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    match child.kind() {
                        "identifier" if name.is_none() => {
                            name = Some(&content[child.start_byte()..child.end_byte()]);
                        }
                        "parameters" => {
                            params = Some(&content[child.start_byte()..child.end_byte()]);
                        }
                        _ => {}
                    }
                }
                match (name, params) {
                    (Some(n), Some(p)) => Some(format!("fn {}{}", n, p)),
                    (Some(n), None) => Some(format!("fn {}", n)),
                    _ => None,
                }
            }
            "impl_item" => {
                // Extract "impl Type" or "impl Trait for Type"
                let text = &content[node.start_byte()..node.end_byte()];
                let first_line = text.lines().next()?;
                // Find up to opening brace
                let sig = first_line.split('{').next()?.trim();
                Some(sig.to_string())
            }
            "struct_item" | "enum_item" | "trait_item" => {
                let text = &content[node.start_byte()..node.end_byte()];
                let first_line = text.lines().next()?;
                let sig = first_line.split('{').next()?.trim();
                Some(sig.to_string())
            }
            _ => None,
        }
    }

    /// Extract Python function/class signature
    fn extract_python_signature(&self, node: Node, content: &str, kind: &str) -> Option<String> {
        match kind {
            "function_definition" => {
                let text = &content[node.start_byte()..node.end_byte()];
                // Get "def name(params):"
                let first_line = text.lines().next()?;
                Some(first_line.trim_end_matches(':').trim().to_string())
            }
            "class_definition" => {
                let text = &content[node.start_byte()..node.end_byte()];
                let first_line = text.lines().next()?;
                Some(first_line.trim_end_matches(':').trim().to_string())
            }
            _ => None,
        }
    }

    /// Extract JavaScript/TypeScript function/class signature
    fn extract_js_signature(&self, node: Node, content: &str, kind: &str) -> Option<String> {
        match kind {
            "function_declaration" | "function" => {
                let text = &content[node.start_byte()..node.end_byte()];
                let first_line = text.lines().next()?;
                let sig = first_line.split('{').next()?.trim();
                Some(sig.to_string())
            }
            "class_declaration" => {
                let text = &content[node.start_byte()..node.end_byte()];
                let first_line = text.lines().next()?;
                let sig = first_line.split('{').next()?.trim();
                Some(sig.to_string())
            }
            "method_definition" => {
                let text = &content[node.start_byte()..node.end_byte()];
                let first_line = text.lines().next()?;
                let sig = first_line.split('{').next()?.trim();
                Some(sig.to_string())
            }
            _ => None,
        }
    }

    /// Extract Go function signature
    fn extract_go_signature(&self, node: Node, content: &str, kind: &str) -> Option<String> {
        match kind {
            "function_declaration" | "method_declaration" => {
                let text = &content[node.start_byte()..node.end_byte()];
                let first_line = text.lines().next()?;
                let sig = first_line.split('{').next()?.trim();
                Some(sig.to_string())
            }
            "type_declaration" => {
                let text = &content[node.start_byte()..node.end_byte()];
                let first_line = text.lines().next()?;
                Some(first_line.trim().to_string())
            }
            _ => None,
        }
    }

    /// Get the comment prefix for a language
    fn get_comment_prefix(language: &str) -> &'static str {
        match language {
            "python" | "py" | "ruby" | "rb" | "shell" | "bash" | "sh" | "yaml" | "yml" | "toml" => "#",
            "html" | "xml" => "<!--",
            "css" => "/*",
            _ => "//", // Rust, JavaScript, TypeScript, Go, Java, C, C++, etc.
        }
    }

    /// Trim content to fit within token limit
    fn trim_to_tokens(&self, content: &str, max_tokens: usize) -> String {
        if self.count_tokens(content) <= max_tokens {
            return content.to_string();
        }

        // Trim line by line from the end
        let lines: Vec<&str> = content.lines().collect();
        let mut result_lines = Vec::new();
        let mut total_tokens = 0;

        for line in lines {
            let line_tokens = self.count_tokens(line);
            if total_tokens + line_tokens > max_tokens {
                break;
            }
            result_lines.push(line);
            total_tokens += line_tokens;
        }

        result_lines.join("\n")
    }

    // ========================================================================
    // SPECIAL CHUNKING MODES
    // ========================================================================

    /// Pattern-aware chunking for regex content
    pub fn chunk_regex_aware(&self, content: &str) -> Vec<BoundedChunk> {
        let chunks = RegexPatternSplitter::split(content, self.config.max_tokens, self);

        chunks
            .into_iter()
            .enumerate()
            .map(|(i, text)| {
                BoundedChunk::new(text, "regex", i + 1, i + 1, 0, 0)
            })
            .collect()
    }

    /// Multi-vector chunking for monster lines
    pub fn chunk_multi_vector(&self, content: &str, language: &str) -> MultiVectorResult {
        let parent_id = Self::generate_content_hash(content);
        let chunks = self.chunk(content, language);

        let chunks_with_parent: Vec<BoundedChunk> = chunks
            .into_iter()
            .map(|c| c.with_parent(&parent_id))
            .collect();

        MultiVectorResult {
            chunks: chunks_with_parent,
            parent_id: parent_id.clone(),
            content_hash: parent_id,
        }
    }

    /// Chunk and combine with weighted average embedding
    pub fn chunk_with_weighted_average<E: Embedder>(
        &self,
        content: &str,
        language: &str,
        embedder: &E,
    ) -> WeightedAverageResult {
        let chunks = self.chunk(content, language);

        if chunks.is_empty() {
            return WeightedAverageResult {
                embedding: vec![0.0; embedder.dimension()],
                chunk_count: 0,
                weights: vec![],
            };
        }

        // Embed each chunk and collect weights
        let mut embeddings: Vec<Vec<f32>> = Vec::new();
        let mut weights: Vec<f32> = Vec::new();

        for chunk in &chunks {
            let emb = embedder.embed(&chunk.content);
            let weight = self.count_tokens(&chunk.content) as f32;
            embeddings.push(emb);
            weights.push(weight);
        }

        // Normalize weights
        let total_weight: f32 = weights.iter().sum();
        let normalized_weights: Vec<f32> = weights.iter().map(|w| w / total_weight).collect();

        // Weighted average
        let dim = embedder.dimension();
        let mut combined = vec![0.0f32; dim];

        for (emb, weight) in embeddings.iter().zip(normalized_weights.iter()) {
            for (i, val) in emb.iter().enumerate() {
                combined[i] += val * weight;
            }
        }

        // L2 normalize
        let norm: f32 = combined.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut combined {
                *val /= norm;
            }
        }

        WeightedAverageResult {
            embedding: combined,
            chunk_count: chunks.len(),
            weights: normalized_weights,
        }
    }

    // ========================================================================
    // TOKEN COUNTING
    // ========================================================================

    /// Count tokens in text
    /// Uses heuristic: ~4 non-whitespace chars per token for code
    pub fn count_tokens(&self, text: &str) -> usize {
        self.approx_tokens(text)
    }

    /// Approximate token count (fast)
    pub fn approx_tokens(&self, text: &str) -> usize {
        let non_ws = text.chars().filter(|c| !c.is_whitespace()).count();
        // ~4 chars per token for code, minimum 1
        // Don't add +1 for small texts to match expected behavior
        if non_ws == 0 {
            return 0;
        }
        ((non_ws as f32) / 4.0).ceil() as usize
    }

    // ========================================================================
    // INTERNAL: AST-BASED SPLITTING
    // ========================================================================

    fn split_at_ast(&self, content: &str, language: &str) -> Option<Vec<BoundedChunk>> {
        let ts_language = Self::get_tree_sitter_language(language)?;

        let mut parser = Parser::new();
        parser.set_language(&ts_language).ok()?;

        let tree = parser.parse(content, None)?;
        let root = tree.root_node();

        let mut chunks = Vec::new();
        self.collect_ast_chunks(root, content, language, &mut chunks);

        if chunks.is_empty() {
            return None;
        }

        // Merge small siblings (cAST algorithm)
        let merged = self.merge_small_siblings(chunks);

        Some(merged)
    }

    fn collect_ast_chunks(
        &self,
        node: Node,
        content: &str,
        language: &str,
        chunks: &mut Vec<BoundedChunk>,
    ) {
        let node_text = &content[node.start_byte()..node.end_byte()];
        let node_tokens = self.count_tokens(node_text);

        // If node fits in target, keep it whole
        if node_tokens <= self.config.target_tokens {
            let start_line = node.start_position().row + 1;
            let end_line = node.end_position().row + 1;

            chunks.push(BoundedChunk::new(
                node_text.to_string(),
                language,
                start_line,
                end_line,
                node.start_byte(),
                node.end_byte(),
            ));
            return;
        }

        // Node too large - try to split at children
        let child_count = node.child_count();
        if child_count == 0 {
            // Leaf node too large - fall back to line splitting
            let line_chunks = self.split_text_at_lines(node_text, language, node.start_byte(), node.start_position().row + 1);
            chunks.extend(line_chunks);
            return;
        }

        // Collect children and decide how to handle
        let mut cursor = node.walk();
        let children: Vec<Node> = node.children(&mut cursor).collect();

        // Check if we should recurse into children or split at this level
        let mut has_splittable_children = false;
        for child in &children {
            let child_text = &content[child.start_byte()..child.end_byte()];
            let child_tokens = self.count_tokens(child_text);
            if child_tokens <= self.config.max_tokens {
                has_splittable_children = true;
                break;
            }
        }

        if has_splittable_children {
            // Recurse into children
            for child in children {
                if Self::is_significant_node(&child) {
                    self.collect_ast_chunks(child, content, language, chunks);
                }
            }
        } else {
            // All children too large, fall back to line splitting
            let line_chunks = self.split_text_at_lines(node_text, language, node.start_byte(), node.start_position().row + 1);
            chunks.extend(line_chunks);
        }
    }

    fn is_significant_node(node: &Node) -> bool {
        let kind = node.kind();
        // Skip trivial nodes
        !matches!(kind, "comment" | "line_comment" | "block_comment" | "{" | "}" | "(" | ")" | ";" | ",")
    }

    fn get_tree_sitter_language(language: &str) -> Option<Language> {
        match language {
            "rust" => Some(tree_sitter_rust::LANGUAGE.into()),
            "python" => Some(tree_sitter_python::LANGUAGE.into()),
            "javascript" | "js" => Some(tree_sitter_javascript::LANGUAGE.into()),
            "typescript" | "ts" => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
            "go" => Some(tree_sitter_go::LANGUAGE.into()),
            _ => None,
        }
    }

    // ========================================================================
    // INTERNAL: LINE-BASED SPLITTING
    // ========================================================================

    fn split_at_lines(&self, content: &str, language: &str) -> Vec<BoundedChunk> {
        self.split_text_at_lines(content, language, 0, 1)
    }

    fn split_text_at_lines(
        &self,
        content: &str,
        language: &str,
        base_byte_offset: usize,
        base_line: usize,
    ) -> Vec<BoundedChunk> {
        let lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() {
            return Vec::new();
        }

        let mut chunks = Vec::new();
        let mut current_lines: Vec<&str> = Vec::new();
        let mut current_tokens = 0;
        let mut chunk_start_line = 0;
        let mut chunk_start_byte = 0;
        let mut current_byte = 0;

        for (i, line) in lines.iter().enumerate() {
            let line_tokens = self.count_tokens(line);

            // If single line exceeds max, split at words
            if line_tokens > self.config.max_tokens {
                // Flush current chunk
                if !current_lines.is_empty() {
                    let chunk_content = current_lines.join("\n");
                    let end_byte = chunk_start_byte + chunk_content.len();
                    chunks.push(BoundedChunk::new(
                        chunk_content,
                        language,
                        base_line + chunk_start_line,
                        base_line + i - 1,
                        base_byte_offset + chunk_start_byte,
                        base_byte_offset + end_byte,
                    ));
                    current_lines.clear();
                    current_tokens = 0;
                }

                // Split long line at words
                let word_chunks = self.split_at_words(line, language, base_byte_offset + current_byte, base_line + i);
                chunks.extend(word_chunks);

                chunk_start_line = i + 1;
                chunk_start_byte = current_byte + line.len() + 1; // +1 for newline
                current_byte += line.len() + 1;
                continue;
            }

            // Check if adding this line would exceed target
            if current_tokens + line_tokens > self.config.target_tokens && !current_lines.is_empty() {
                // Flush current chunk
                let chunk_content = current_lines.join("\n");
                let end_byte = chunk_start_byte + chunk_content.len();
                chunks.push(BoundedChunk::new(
                    chunk_content,
                    language,
                    base_line + chunk_start_line,
                    base_line + i - 1,
                    base_byte_offset + chunk_start_byte,
                    base_byte_offset + end_byte,
                ));
                current_lines.clear();
                current_tokens = 0;
                chunk_start_line = i;
                chunk_start_byte = current_byte;
            }

            current_lines.push(line);
            current_tokens += line_tokens;
            current_byte += line.len() + 1; // +1 for newline
        }

        // Don't forget the last chunk
        if !current_lines.is_empty() {
            let chunk_content = current_lines.join("\n");
            let end_byte = chunk_start_byte + chunk_content.len();
            chunks.push(BoundedChunk::new(
                chunk_content,
                language,
                base_line + chunk_start_line,
                base_line + lines.len() - 1,
                base_byte_offset + chunk_start_byte,
                base_byte_offset + end_byte,
            ));
        }

        chunks
    }

    // ========================================================================
    // INTERNAL: WORD-BASED SPLITTING
    // ========================================================================

    fn split_at_words(
        &self,
        text: &str,
        language: &str,
        base_byte_offset: usize,
        line_num: usize,
    ) -> Vec<BoundedChunk> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Vec::new();
        }

        let mut chunks = Vec::new();
        let mut current_words: Vec<&str> = Vec::new();
        let mut current_tokens = 0;

        for word in words {
            let word_tokens = self.count_tokens(word);

            // If single word exceeds max, truncate it into multiple chunks
            if word_tokens > self.config.max_tokens {
                // Flush current
                if !current_words.is_empty() {
                    let chunk_content = current_words.join(" ");
                    chunks.push(BoundedChunk::new(
                        chunk_content,
                        language,
                        line_num,
                        line_num,
                        base_byte_offset,
                        base_byte_offset,
                    ));
                    current_words.clear();
                    current_tokens = 0;
                }

                // Truncate the monster word into multiple chunks
                let mut remaining = word;
                while !remaining.is_empty() {
                    let truncated = self.truncate_to_limit(remaining);
                    let truncated_len = truncated.len();
                    chunks.push(BoundedChunk::new(
                        truncated,
                        language,
                        line_num,
                        line_num,
                        base_byte_offset,
                        base_byte_offset,
                    ));
                    if truncated_len >= remaining.len() {
                        break;
                    }
                    remaining = &remaining[truncated_len..];
                }
                continue;
            }

            // Check if adding word exceeds target
            if current_tokens + word_tokens > self.config.target_tokens && !current_words.is_empty() {
                let chunk_content = current_words.join(" ");
                chunks.push(BoundedChunk::new(
                    chunk_content,
                    language,
                    line_num,
                    line_num,
                    base_byte_offset,
                    base_byte_offset,
                ));
                current_words.clear();
                current_tokens = 0;
            }

            current_words.push(word);
            current_tokens += word_tokens;
        }

        // Last chunk
        if !current_words.is_empty() {
            let chunk_content = current_words.join(" ");
            chunks.push(BoundedChunk::new(
                chunk_content,
                language,
                line_num,
                line_num,
                base_byte_offset,
                base_byte_offset,
            ));
        }

        chunks
    }

    // ========================================================================
    // INTERNAL: TRUNCATION (Last Resort)
    // ========================================================================

    fn truncate_to_limit(&self, text: &str) -> String {
        // Use binary search to find the maximum chars that fit in max_tokens
        let chars: Vec<char> = text.chars().collect();

        if self.count_tokens(text) <= self.config.max_tokens {
            return text.to_string();
        }

        // Binary search for the right length
        let mut low = 0;
        let mut high = chars.len();

        while low < high {
            let mid = (low + high + 1) / 2;
            let candidate: String = chars[..mid].iter().collect();
            if self.count_tokens(&candidate) <= self.config.max_tokens {
                low = mid;
            } else {
                high = mid - 1;
            }
        }

        chars[..low.max(1)].iter().collect()
    }

    // ========================================================================
    // INTERNAL: SIBLING MERGING (cAST algorithm)
    // ========================================================================

    fn merge_small_siblings(&self, chunks: Vec<BoundedChunk>) -> Vec<BoundedChunk> {
        if chunks.len() <= 1 {
            return chunks;
        }

        let mut merged = Vec::new();
        let mut current_content = String::new();
        let mut current_start_line = 0;
        let mut current_end_line = 0;
        let mut current_start_byte = 0;
        let mut current_end_byte = 0;
        let mut current_tokens = 0;
        let mut language = String::new();

        // Track overlap content from previous chunk for sibling overlap
        let mut overlap_content: Option<String> = None;

        for chunk in chunks {
            let chunk_tokens = self.count_tokens(&chunk.content);

            if current_content.is_empty() {
                // Starting a new chunk - prepend overlap from previous if available
                if let Some(ref overlap) = overlap_content {
                    let overlap_tokens = self.count_tokens(overlap);
                    if overlap_tokens + chunk_tokens <= self.config.max_tokens {
                        current_content = format!("{}\n{}", overlap, chunk.content);
                        current_tokens = overlap_tokens + chunk_tokens;
                    } else {
                        current_content = chunk.content;
                        current_tokens = chunk_tokens;
                    }
                } else {
                    current_content = chunk.content;
                    current_tokens = chunk_tokens;
                }
                current_start_line = chunk.start_line;
                current_end_line = chunk.end_line;
                current_start_byte = chunk.start_byte;
                current_end_byte = chunk.end_byte;
                language = chunk.language;
                continue;
            }

            // Can we merge?
            if current_tokens + chunk_tokens <= self.config.target_tokens {
                // Merge
                current_content.push('\n');
                current_content.push_str(&chunk.content);
                current_end_line = chunk.end_line;
                current_end_byte = chunk.end_byte;
                current_tokens += chunk_tokens;
            } else {
                // Flush current and start new
                // Calculate overlap for next chunk (15% of current chunk)
                overlap_content = self.extract_overlap_content(&current_content);

                merged.push(BoundedChunk::new(
                    current_content,
                    &language,
                    current_start_line,
                    current_end_line,
                    current_start_byte,
                    current_end_byte,
                ));

                // Start new chunk with overlap from previous
                if let Some(ref overlap) = overlap_content {
                    let overlap_tokens = self.count_tokens(overlap);
                    if overlap_tokens + chunk_tokens <= self.config.max_tokens {
                        current_content = format!("{}\n{}", overlap, chunk.content);
                        current_tokens = overlap_tokens + chunk_tokens;
                    } else {
                        current_content = chunk.content;
                        current_tokens = chunk_tokens;
                    }
                } else {
                    current_content = chunk.content;
                    current_tokens = chunk_tokens;
                }
                current_start_line = chunk.start_line;
                current_end_line = chunk.end_line;
                current_start_byte = chunk.start_byte;
                current_end_byte = chunk.end_byte;
                language = chunk.language;
            }
        }

        // Don't forget last one
        if !current_content.is_empty() {
            merged.push(BoundedChunk::new(
                current_content,
                &language,
                current_start_line,
                current_end_line,
                current_start_byte,
                current_end_byte,
            ));
        }

        merged
    }

    /// Extract overlap content from the end of a chunk (based on overlap_ratio)
    fn extract_overlap_content(&self, content: &str) -> Option<String> {
        if self.config.overlap_ratio <= 0.0 {
            return None;
        }

        let total_tokens = self.count_tokens(content);
        let overlap_tokens = (total_tokens as f32 * self.config.overlap_ratio) as usize;

        if overlap_tokens == 0 {
            return None;
        }

        // Get last N lines that fit within overlap_tokens
        let lines: Vec<&str> = content.lines().collect();
        let mut overlap_lines = Vec::new();
        let mut collected_tokens = 0;

        for line in lines.iter().rev() {
            let line_tokens = self.count_tokens(line);
            if collected_tokens + line_tokens > overlap_tokens {
                break;
            }
            overlap_lines.push(*line);
            collected_tokens += line_tokens;
        }

        if overlap_lines.is_empty() {
            return None;
        }

        overlap_lines.reverse();
        Some(overlap_lines.join("\n"))
    }

    // ========================================================================
    // INTERNAL: OVERLAP
    // ========================================================================

    fn apply_overlap(&self, chunks: Vec<BoundedChunk>, _original_content: &str) -> Vec<BoundedChunk> {
        if self.config.overlap_ratio <= 0.0 || chunks.len() <= 1 {
            return chunks;
        }

        let mut result = Vec::new();

        for (i, chunk) in chunks.iter().enumerate() {
            if i == 0 {
                result.push(chunk.clone());
                continue;
            }

            // Calculate overlap from previous chunk
            let prev_chunk = &chunks[i - 1];
            let overlap_tokens = (self.count_tokens(&prev_chunk.content) as f32 * self.config.overlap_ratio) as usize;

            // Get last N tokens worth of content from previous chunk
            let prev_lines: Vec<&str> = prev_chunk.content.lines().collect();
            let mut overlap_content = String::new();
            let mut overlap_token_count = 0;

            for line in prev_lines.iter().rev() {
                let line_tokens = self.count_tokens(line);
                if overlap_token_count + line_tokens > overlap_tokens {
                    break;
                }
                if !overlap_content.is_empty() {
                    overlap_content = format!("{}\n{}", line, overlap_content);
                } else {
                    overlap_content = line.to_string();
                }
                overlap_token_count += line_tokens;
            }

            // Combine overlap with current chunk
            let combined = if overlap_content.is_empty() {
                chunk.content.clone()
            } else {
                format!("{}\n{}", overlap_content, chunk.content)
            };

            // Ensure we don't exceed max after adding overlap
            let combined_tokens = self.count_tokens(&combined);
            let final_content = if combined_tokens > self.config.max_tokens {
                // Trim overlap to fit
                chunk.content.clone()
            } else {
                combined
            };

            result.push(BoundedChunk::new(
                final_content,
                &chunk.language,
                chunk.start_line.saturating_sub(overlap_token_count / 10), // Approximate line adjustment
                chunk.end_line,
                chunk.start_byte,
                chunk.end_byte,
            ));
        }

        result
    }

    // ========================================================================
    // HELPERS
    // ========================================================================

    fn generate_content_hash(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())[..16].to_string()
    }
}

// ============================================================================
// EMBEDDER TRAIT
// ============================================================================

/// Trait for embedding text (used by weighted average mode)
pub trait Embedder {
    /// Embed a single text, returning the embedding vector
    fn embed(&self, text: &str) -> Vec<f32>;

    /// Get the embedding dimension
    fn dimension(&self) -> usize;
}

// ============================================================================
// REGEX PATTERN SPLITTER
// ============================================================================

/// Splits regex patterns at logical boundaries
pub struct RegexPatternSplitter;

impl RegexPatternSplitter {
    /// Split a regex at logical boundaries
    ///
    /// Boundaries (in priority order):
    /// 1. `|` - Alternation
    /// 2. `)(` - Between groups
    /// 3. `(?:` - Non-capturing group start
    pub fn split(regex: &str, max_tokens: usize, chunker: &BoundedSemanticChunker) -> Vec<String> {
        let total_tokens = chunker.count_tokens(regex);
        if total_tokens <= max_tokens {
            return vec![regex.to_string()];
        }

        // Try splitting at | first
        let parts: Vec<&str> = regex.split('|').collect();
        if parts.len() > 1 {
            return Self::merge_parts_to_limit(&parts, "|", max_tokens, chunker);
        }

        // Try splitting at )(
        let parts: Vec<&str> = regex.split(")(").collect();
        if parts.len() > 1 {
            return Self::merge_parts_to_limit(&parts, ")(", max_tokens, chunker);
        }

        // Try splitting at (?:
        let parts: Vec<&str> = regex.split("(?:").collect();
        if parts.len() > 1 {
            return Self::merge_parts_to_limit(&parts, "(?:", max_tokens, chunker);
        }

        // Fall back to character-based splitting
        Self::split_by_chars(regex, max_tokens, chunker)
    }

    fn merge_parts_to_limit(
        parts: &[&str],
        delimiter: &str,
        max_tokens: usize,
        chunker: &BoundedSemanticChunker,
    ) -> Vec<String> {
        let target = (max_tokens as f32 * 0.8) as usize;
        let mut chunks = Vec::new();
        let mut current = String::new();
        let mut current_tokens = 0;

        for (i, part) in parts.iter().enumerate() {
            let part_with_delim = if i == 0 {
                part.to_string()
            } else {
                format!("{}{}", delimiter, part)
            };

            let part_tokens = chunker.count_tokens(&part_with_delim);

            if current_tokens + part_tokens > target && !current.is_empty() {
                chunks.push(current);
                current = part_with_delim;
                current_tokens = part_tokens;
            } else {
                current.push_str(&part_with_delim);
                current_tokens += part_tokens;
            }
        }

        if !current.is_empty() {
            chunks.push(current);
        }

        chunks
    }

    fn split_by_chars(text: &str, max_tokens: usize, chunker: &BoundedSemanticChunker) -> Vec<String> {
        let target_chars = (max_tokens as f32 * 3.5 * 0.8) as usize;
        let chars: Vec<char> = text.chars().collect();

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < chars.len() {
            let end = (start + target_chars).min(chars.len());
            let chunk: String = chars[start..end].iter().collect();

            // Verify it fits
            if chunker.count_tokens(&chunk) <= max_tokens {
                chunks.push(chunk);
            } else {
                // Reduce size
                let reduced_end = start + target_chars / 2;
                let chunk: String = chars[start..reduced_end.min(chars.len())].iter().collect();
                chunks.push(chunk);
            }

            start = end;
        }

        chunks
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_default_is_sensible() {
        let config = BoundedChunkerConfig::default();
        // Reduced defaults: 350 max, 280 target, 25% overlap (match line chunker)
        assert_eq!(config.max_tokens, 350);
        assert_eq!(config.target_tokens, 280);
        assert!((config.overlap_ratio - 0.25).abs() < 0.001);
        assert!(!config.inject_context);
    }

    #[test]
    fn config_for_model_calculates_target() {
        let config = BoundedChunkerConfig::for_model(1000);
        assert_eq!(config.max_tokens, 1000);
        assert_eq!(config.target_tokens, 800); // 80%
    }

    #[test]
    fn approx_tokens_is_conservative() {
        let chunker = BoundedSemanticChunker::new(BoundedChunkerConfig::default());

        // Short text
        let tokens = chunker.approx_tokens("hello");
        assert!(tokens >= 1, "Should count at least 1 token");

        // Longer text
        let tokens = chunker.approx_tokens("fn main() { println!(\"hello\"); }");
        assert!(tokens >= 5, "Should count multiple tokens");
    }

    #[test]
    fn empty_content_returns_empty() {
        let chunker = BoundedSemanticChunker::new(BoundedChunkerConfig::default());
        let chunks = chunker.chunk("", "rust");
        assert!(chunks.is_empty());
    }

    #[test]
    fn whitespace_only_returns_empty() {
        let chunker = BoundedSemanticChunker::new(BoundedChunkerConfig::default());
        let chunks = chunker.chunk("   \n\n\t  ", "rust");
        assert!(chunks.is_empty());
    }

    #[test]
    fn small_content_single_chunk() {
        let chunker = BoundedSemanticChunker::new(BoundedChunkerConfig::for_model(500));
        let chunks = chunker.chunk("fn add(a: i32, b: i32) -> i32 { a + b }", "rust");
        assert_eq!(chunks.len(), 1);
    }
}
