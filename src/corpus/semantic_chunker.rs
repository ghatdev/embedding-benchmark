//! Semantic chunking with context preservation
//!
//! Structure-aware chunking approach:
//! - For code: Inject class/function context into each chunk
//! - For docs: Inject header breadcrumb into each chunk
//!
//! This ensures every chunk is self-contained and semantically complete.

use std::path::Path;
use tree_sitter::{Language, Node, Parser};

use super::loader::{Chunk, ContentType};

/// Maximum characters per chunk (content + context)
pub const DEFAULT_MAX_CHARS: usize = 2000;

/// Maximum characters reserved for context injection
pub const DEFAULT_CONTEXT_BUDGET: usize = 400;

/// Minimum characters for a valid chunk
pub const MIN_CHUNK_CHARS: usize = 50;

// =============================================================================
// AST-BASED SEMANTIC CHUNKER (for code)
// =============================================================================

/// AST-based chunker with semantic context preservation
///
/// Key features:
/// - Parses code into AST using tree-sitter
/// - Extracts context (class definitions, function signatures)
/// - Injects context into each chunk
/// - Keeps semantic units together when possible
pub struct AstSemanticChunker {
    /// Maximum total characters per chunk (content + context)
    max_chars: usize,
    /// Maximum characters reserved for context
    context_budget: usize,
    /// Minimum characters for a valid chunk
    min_chars: usize,
}

impl AstSemanticChunker {
    pub fn new(max_chars: usize, context_budget: usize) -> Self {
        Self {
            max_chars,
            context_budget,
            min_chars: MIN_CHUNK_CHARS,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_MAX_CHARS, DEFAULT_CONTEXT_BUDGET)
    }

    /// Get tree-sitter language for a file extension
    fn get_language(ext: &str) -> Option<Language> {
        match ext {
            "rs" => Some(tree_sitter_rust::LANGUAGE.into()),
            "py" => Some(tree_sitter_python::LANGUAGE.into()),
            "js" | "jsx" => Some(tree_sitter_javascript::LANGUAGE.into()),
            "ts" => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
            "tsx" => Some(tree_sitter_typescript::LANGUAGE_TSX.into()),
            "go" => Some(tree_sitter_go::LANGUAGE.into()),
            _ => None,
        }
    }

    /// Check if a language is supported
    pub fn is_supported(ext: &str) -> bool {
        Self::get_language(ext).is_some()
    }

    /// Create chunks from code with context preservation
    pub fn create_chunks(
        &self,
        file_path: &Path,
        content: &str,
        language: &str,
        ext: &str,
    ) -> Vec<Chunk> {
        let file_path_str = file_path.to_string_lossy().to_string();

        // Try to parse with tree-sitter
        let language_ts = match Self::get_language(ext) {
            Some(lang) => lang,
            None => return self.fallback_line_chunks(file_path, content, language),
        };

        let mut parser = Parser::new();
        if parser.set_language(&language_ts).is_err() {
            return self.fallback_line_chunks(file_path, content, language);
        }

        let tree = match parser.parse(content, None) {
            Some(t) => t,
            None => return self.fallback_line_chunks(file_path, content, language),
        };

        let root = tree.root_node();

        // Extract file-level context
        let file_context = self.extract_file_context(content, language);

        // Find all top-level items (functions, classes, impl blocks)
        let items = self.find_top_level_items(root, content, language);

        if items.is_empty() {
            return self.fallback_line_chunks(file_path, content, language);
        }

        // Build chunks with context
        let mut chunks = Vec::new();
        let content_budget = self.max_chars - self.context_budget;

        // Collect covered ranges first
        let covered_ranges: Vec<(usize, usize)> = items.iter()
            .map(|i| (i.start_byte, i.end_byte))
            .collect();

        for item in &items {
            let item_chunks = self.chunk_item_with_context(
                &file_path_str,
                content,
                language,
                item,
                &file_context,
                content_budget,
            );
            chunks.extend(item_chunks);
        }

        // Handle any remaining content not covered by items
        // Split uncovered ranges to fit within max_chars
        let uncovered = self.find_uncovered_ranges(content.len(), &covered_ranges);
        for (start, end) in uncovered {
            let text = &content[start..end];
            if text.trim().len() >= self.min_chars {
                let context_prefix = self.build_context_prefix(&file_path_str, &file_context, None);
                let context_size = context_prefix.len();
                let effective_budget = self.max_chars.saturating_sub(context_size);

                // Split large uncovered ranges into smaller chunks
                let text_trimmed = text.trim();
                if text_trimmed.len() <= effective_budget {
                    // Fits in one chunk
                    let chunk_content = format!("{}{}", context_prefix, text_trimmed);
                    let start_line = content[..start].lines().count().max(1);
                    let end_line = content[..end].lines().count().max(start_line);

                    chunks.push(Chunk::new(
                        &file_path_str,
                        chunk_content,
                        language,
                        start_line,
                        end_line,
                        ContentType::Code,
                    ));
                } else {
                    // Need to split - use line-based splitting
                    let lines: Vec<&str> = text_trimmed.lines().collect();
                    let base_line = content[..start].lines().count().max(1);
                    let mut current_chunk = Vec::new();
                    let mut current_size = 0;

                    for (i, line) in lines.iter().enumerate() {
                        let line_size = line.len() + 1;
                        if current_size + line_size > effective_budget && !current_chunk.is_empty() {
                            let chunk_text = current_chunk.join("\n");
                            let chunk_content = format!("{}{}", context_prefix, chunk_text);
                            let chunk_start = base_line + i - current_chunk.len();
                            let chunk_end = base_line + i - 1;

                            chunks.push(Chunk::new(
                                &file_path_str,
                                chunk_content,
                                language,
                                chunk_start.max(1),
                                chunk_end.max(chunk_start),
                                ContentType::Code,
                            ));
                            current_chunk.clear();
                            current_size = 0;
                        }
                        current_chunk.push(*line);
                        current_size += line_size;
                    }

                    // Don't forget last chunk
                    if !current_chunk.is_empty() && current_size >= self.min_chars {
                        let chunk_text = current_chunk.join("\n");
                        let chunk_content = format!("{}{}", context_prefix, chunk_text);
                        let chunk_start = base_line + lines.len() - current_chunk.len();
                        let chunk_end = base_line + lines.len() - 1;

                        chunks.push(Chunk::new(
                            &file_path_str,
                            chunk_content,
                            language,
                            chunk_start.max(1),
                            chunk_end.max(chunk_start),
                            ContentType::Code,
                        ));
                    }
                }
            }
        }

        if chunks.is_empty() {
            return self.fallback_line_chunks(file_path, content, language);
        }

        chunks
    }

    /// Find top-level items (functions, classes, impl blocks, etc.)
    fn find_top_level_items<'a>(
        &self,
        root: Node<'a>,
        content: &'a str,
        language: &str,
    ) -> Vec<CodeItem<'a>> {
        let mut items = Vec::new();

        for child in root.children(&mut root.walk()) {
            if let Some(item) = self.node_to_code_item(child, content, language, None) {
                items.push(item);
            }
        }

        items
    }

    /// Convert a tree-sitter node to a CodeItem
    fn node_to_code_item<'a>(
        &self,
        node: Node<'a>,
        content: &'a str,
        language: &str,
        parent_context: Option<&str>,
    ) -> Option<CodeItem<'a>> {
        let kind = node.kind();

        match language {
            "rust" => self.rust_node_to_item(node, content, kind, parent_context),
            "python" => self.python_node_to_item(node, content, kind, parent_context),
            "typescript" | "javascript" => self.ts_node_to_item(node, content, kind, parent_context),
            "go" => self.go_node_to_item(node, content, kind, parent_context),
            _ => None,
        }
    }

    /// Convert Rust AST node to CodeItem
    fn rust_node_to_item<'a>(
        &self,
        node: Node<'a>,
        content: &'a str,
        kind: &str,
        parent_context: Option<&str>,
    ) -> Option<CodeItem<'a>> {
        match kind {
            "function_item" | "function_signature_item" => {
                let signature = self.extract_rust_function_signature(node, content);
                Some(CodeItem {
                    node,
                    kind: ItemKind::Function,
                    signature,
                    parent_context: parent_context.map(String::from),
                    start_byte: node.start_byte(),
                    end_byte: node.end_byte(),
                    children: Vec::new(),
                })
            }
            "impl_item" => {
                let signature = self.extract_rust_impl_signature(node, content);
                let mut item = CodeItem {
                    node,
                    kind: ItemKind::ImplBlock,
                    signature: signature.clone(),
                    parent_context: parent_context.map(String::from),
                    start_byte: node.start_byte(),
                    end_byte: node.end_byte(),
                    children: Vec::new(),
                };

                // Find methods inside impl block
                for child in node.children(&mut node.walk()) {
                    if child.kind() == "function_item" {
                        if let Some(method) = self.rust_node_to_item(child, content, "function_item", Some(&signature)) {
                            item.children.push(method);
                        }
                    }
                }

                Some(item)
            }
            "struct_item" | "enum_item" | "trait_item" => {
                let signature = self.extract_rust_type_signature(node, content);
                Some(CodeItem {
                    node,
                    kind: ItemKind::TypeDef,
                    signature,
                    parent_context: parent_context.map(String::from),
                    start_byte: node.start_byte(),
                    end_byte: node.end_byte(),
                    children: Vec::new(),
                })
            }
            "mod_item" => {
                let signature = self.extract_rust_mod_signature(node, content);
                Some(CodeItem {
                    node,
                    kind: ItemKind::Module,
                    signature,
                    parent_context: parent_context.map(String::from),
                    start_byte: node.start_byte(),
                    end_byte: node.end_byte(),
                    children: Vec::new(),
                })
            }
            _ => None,
        }
    }

    /// Extract Rust function signature (without body)
    fn extract_rust_function_signature(&self, node: Node, content: &str) -> String {
        // Find the block (function body) and exclude it
        let mut sig_end = node.end_byte();
        for child in node.children(&mut node.walk()) {
            if child.kind() == "block" {
                sig_end = child.start_byte();
                break;
            }
        }

        let sig = &content[node.start_byte()..sig_end];
        sig.trim().to_string()
    }

    /// Extract Rust impl block signature
    fn extract_rust_impl_signature(&self, node: Node, content: &str) -> String {
        // Get "impl Trait for Type" or "impl Type"
        let mut sig_end = node.end_byte();
        for child in node.children(&mut node.walk()) {
            if child.kind() == "declaration_list" {
                sig_end = child.start_byte();
                break;
            }
        }

        let sig = &content[node.start_byte()..sig_end];
        sig.trim().to_string()
    }

    /// Extract Rust type signature
    fn extract_rust_type_signature(&self, node: Node, content: &str) -> String {
        let text = &content[node.start_byte()..node.end_byte()];
        // Take first line or up to opening brace
        if let Some(brace_pos) = text.find('{') {
            text[..brace_pos].trim().to_string()
        } else {
            text.lines().next().unwrap_or("").trim().to_string()
        }
    }

    /// Extract Rust mod signature
    fn extract_rust_mod_signature(&self, node: Node, content: &str) -> String {
        let text = &content[node.start_byte()..node.end_byte()];
        text.lines().next().unwrap_or("").trim().to_string()
    }

    /// Convert Python AST node to CodeItem
    fn python_node_to_item<'a>(
        &self,
        node: Node<'a>,
        content: &'a str,
        kind: &str,
        parent_context: Option<&str>,
    ) -> Option<CodeItem<'a>> {
        match kind {
            "function_definition" => {
                let signature = self.extract_python_function_signature(node, content);
                Some(CodeItem {
                    node,
                    kind: ItemKind::Function,
                    signature,
                    parent_context: parent_context.map(String::from),
                    start_byte: node.start_byte(),
                    end_byte: node.end_byte(),
                    children: Vec::new(),
                })
            }
            "class_definition" => {
                let signature = self.extract_python_class_signature(node, content);
                let mut item = CodeItem {
                    node,
                    kind: ItemKind::Class,
                    signature: signature.clone(),
                    parent_context: parent_context.map(String::from),
                    start_byte: node.start_byte(),
                    end_byte: node.end_byte(),
                    children: Vec::new(),
                };

                // Find methods inside class
                if let Some(body) = node.child_by_field_name("body") {
                    for child in body.children(&mut body.walk()) {
                        if child.kind() == "function_definition" {
                            if let Some(method) = self.python_node_to_item(child, content, "function_definition", Some(&signature)) {
                                item.children.push(method);
                            }
                        }
                    }
                }

                Some(item)
            }
            _ => None,
        }
    }

    /// Extract Python function signature
    fn extract_python_function_signature(&self, node: Node, content: &str) -> String {
        // Find the ":" at end of signature
        let text = &content[node.start_byte()..node.end_byte()];
        if let Some(colon_pos) = text.find(':') {
            // Check if there's a return type annotation
            let sig = &text[..=colon_pos];
            sig.trim().to_string()
        } else {
            text.lines().next().unwrap_or("").trim().to_string()
        }
    }

    /// Extract Python class signature
    fn extract_python_class_signature(&self, node: Node, content: &str) -> String {
        let text = &content[node.start_byte()..node.end_byte()];
        // Get "class Name(Base):" line
        if let Some(colon_pos) = text.find(':') {
            text[..=colon_pos].trim().to_string()
        } else {
            text.lines().next().unwrap_or("").trim().to_string()
        }
    }

    /// Convert TypeScript/JavaScript AST node to CodeItem
    fn ts_node_to_item<'a>(
        &self,
        node: Node<'a>,
        content: &'a str,
        kind: &str,
        parent_context: Option<&str>,
    ) -> Option<CodeItem<'a>> {
        match kind {
            "function_declaration" | "arrow_function" | "method_definition" => {
                let signature = self.extract_ts_function_signature(node, content);
                Some(CodeItem {
                    node,
                    kind: ItemKind::Function,
                    signature,
                    parent_context: parent_context.map(String::from),
                    start_byte: node.start_byte(),
                    end_byte: node.end_byte(),
                    children: Vec::new(),
                })
            }
            "class_declaration" => {
                let signature = self.extract_ts_class_signature(node, content);
                let mut item = CodeItem {
                    node,
                    kind: ItemKind::Class,
                    signature: signature.clone(),
                    parent_context: parent_context.map(String::from),
                    start_byte: node.start_byte(),
                    end_byte: node.end_byte(),
                    children: Vec::new(),
                };

                // Find methods inside class
                if let Some(body) = node.child_by_field_name("body") {
                    for child in body.children(&mut body.walk()) {
                        if child.kind() == "method_definition" {
                            if let Some(method) = self.ts_node_to_item(child, content, "method_definition", Some(&signature)) {
                                item.children.push(method);
                            }
                        }
                    }
                }

                Some(item)
            }
            _ => None,
        }
    }

    /// Extract TypeScript/JavaScript function signature
    fn extract_ts_function_signature(&self, node: Node, content: &str) -> String {
        let text = &content[node.start_byte()..node.end_byte()];
        if let Some(brace_pos) = text.find('{') {
            text[..brace_pos].trim().to_string()
        } else {
            text.lines().next().unwrap_or("").trim().to_string()
        }
    }

    /// Extract TypeScript/JavaScript class signature
    fn extract_ts_class_signature(&self, node: Node, content: &str) -> String {
        let text = &content[node.start_byte()..node.end_byte()];
        if let Some(brace_pos) = text.find('{') {
            text[..brace_pos].trim().to_string()
        } else {
            text.lines().next().unwrap_or("").trim().to_string()
        }
    }

    /// Convert Go AST node to CodeItem
    fn go_node_to_item<'a>(
        &self,
        node: Node<'a>,
        content: &'a str,
        kind: &str,
        parent_context: Option<&str>,
    ) -> Option<CodeItem<'a>> {
        match kind {
            "function_declaration" | "method_declaration" => {
                let signature = self.extract_go_function_signature(node, content);
                Some(CodeItem {
                    node,
                    kind: ItemKind::Function,
                    signature,
                    parent_context: parent_context.map(String::from),
                    start_byte: node.start_byte(),
                    end_byte: node.end_byte(),
                    children: Vec::new(),
                })
            }
            "type_declaration" => {
                let signature = self.extract_go_type_signature(node, content);
                Some(CodeItem {
                    node,
                    kind: ItemKind::TypeDef,
                    signature,
                    parent_context: parent_context.map(String::from),
                    start_byte: node.start_byte(),
                    end_byte: node.end_byte(),
                    children: Vec::new(),
                })
            }
            _ => None,
        }
    }

    /// Extract Go function signature
    fn extract_go_function_signature(&self, node: Node, content: &str) -> String {
        let text = &content[node.start_byte()..node.end_byte()];
        if let Some(brace_pos) = text.find('{') {
            text[..brace_pos].trim().to_string()
        } else {
            text.lines().next().unwrap_or("").trim().to_string()
        }
    }

    /// Extract Go type signature
    fn extract_go_type_signature(&self, node: Node, content: &str) -> String {
        let text = &content[node.start_byte()..node.end_byte()];
        text.lines().next().unwrap_or("").trim().to_string()
    }

    /// Chunk an item with context injection
    fn chunk_item_with_context(
        &self,
        file_path: &str,
        content: &str,
        language: &str,
        item: &CodeItem,
        file_context: &Option<String>,
        content_budget: usize,
    ) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let item_text = &content[item.start_byte..item.end_byte];
        let item_size = item_text.len();

        // Build context for this item
        let item_context = self.build_item_context(item);
        let context_prefix = self.build_context_prefix(file_path, file_context, Some(&item_context));
        let context_size = context_prefix.len();

        // Calculate actual budget after accounting for context prefix
        let effective_budget = content_budget.saturating_sub(context_size);

        // If item fits in effective budget (item + context <= max), keep it whole
        if item_size <= effective_budget {
            let chunk_content = format!("{}{}", context_prefix, item_text);

            let start_line = content[..item.start_byte].lines().count().max(1);
            let end_line = content[..item.end_byte].lines().count().max(start_line);

            chunks.push(Chunk::new(
                file_path,
                chunk_content,
                language,
                start_line,
                end_line,
                ContentType::Code,
            ));
            return chunks;
        }

        // Item is too big - need to split
        // For items with children (classes, impl blocks), chunk children separately
        if !item.children.is_empty() {
            // First, add the item header as its own chunk
            let header_end = item.children.first()
                .map(|c| c.start_byte)
                .unwrap_or(item.end_byte);
            let header_text = &content[item.start_byte..header_end];

            if header_text.trim().len() >= self.min_chars {
                // Header chunks are usually small, just create directly
                let chunk_content = format!("{}{}", context_prefix, header_text.trim());

                let start_line = content[..item.start_byte].lines().count().max(1);
                let end_line = content[..header_end].lines().count().max(start_line);

                chunks.push(Chunk::new(
                    file_path,
                    chunk_content,
                    language,
                    start_line,
                    end_line,
                    ContentType::Code,
                ));
            }

            // Then chunk each child with parent context
            for child in &item.children {
                let child_chunks = self.chunk_item_with_context(
                    file_path,
                    content,
                    language,
                    child,
                    file_context,
                    content_budget,
                );
                chunks.extend(child_chunks);
            }
        } else {
            // No children - split by lines with context
            // Use effective_budget that accounts for context prefix
            let lines: Vec<&str> = item_text.lines().collect();
            let mut current_chunk = Vec::new();
            let mut current_size = 0;

            for (i, line) in lines.iter().enumerate() {
                let line_size = line.len() + 1; // +1 for newline

                if current_size + line_size > effective_budget && !current_chunk.is_empty() {
                    // Save current chunk
                    let chunk_text = current_chunk.join("\n");
                    let chunk_content = format!("{}{}", context_prefix, chunk_text);

                    let chunk_start_line = content[..item.start_byte].lines().count()
                        + (i - current_chunk.len())
                        + 1;
                    let chunk_end_line = content[..item.start_byte].lines().count() + i;

                    chunks.push(Chunk::new(
                        file_path,
                        chunk_content,
                        language,
                        chunk_start_line.max(1),
                        chunk_end_line.max(chunk_start_line),
                        ContentType::Code,
                    ));

                    current_chunk.clear();
                    current_size = 0;
                }

                current_chunk.push(*line);
                current_size += line_size;
            }

            // Don't forget last chunk
            if !current_chunk.is_empty() && current_size >= self.min_chars {
                let chunk_text = current_chunk.join("\n");
                let chunk_content = format!("{}{}", context_prefix, chunk_text);

                let chunk_start_line = content[..item.start_byte].lines().count()
                    + (lines.len() - current_chunk.len())
                    + 1;
                let chunk_end_line = content[..item.end_byte].lines().count();

                chunks.push(Chunk::new(
                    file_path,
                    chunk_content,
                    language,
                    chunk_start_line.max(1),
                    chunk_end_line.max(chunk_start_line),
                    ContentType::Code,
                ));
            }
        }

        chunks
    }

    /// Build context string for an item
    fn build_item_context(&self, item: &CodeItem) -> String {
        let mut context = String::new();

        // Add parent context if available
        if let Some(parent) = &item.parent_context {
            context.push_str(parent);
            context.push_str(" > ");
        }

        // Add this item's signature
        context.push_str(&item.signature);

        context
    }

    /// Build full context prefix for a chunk
    fn build_context_prefix(
        &self,
        file_path: &str,
        file_context: &Option<String>,
        item_context: Option<&str>,
    ) -> String {
        let mut prefix = format!("[file: {}]", file_path);

        if let Some(fc) = file_context {
            if !fc.is_empty() {
                prefix.push_str(&format!(" {}", fc));
            }
        }

        if let Some(ic) = item_context {
            if !ic.is_empty() {
                prefix.push_str(&format!("\n[context: {}]", ic));
            }
        }

        prefix.push_str("\n\n");
        prefix
    }

    /// Extract file-level context (module docs, etc.)
    fn extract_file_context(&self, content: &str, language: &str) -> Option<String> {
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
                    Some(doc_lines.into_iter().take(2).collect::<Vec<_>>().join(" "))
                }
            }
            "python" => {
                // Extract module docstring
                let mut in_docstring = false;
                let mut docstring_lines = Vec::new();

                for line in lines.iter().take(20) {
                    let trimmed = line.trim();

                    if !in_docstring {
                        if trimmed.is_empty() || trimmed.starts_with('#') {
                            continue;
                        }

                        if trimmed.starts_with("\"\"\"") || trimmed.starts_with("'''") {
                            in_docstring = true;
                            let quote = if trimmed.starts_with("\"\"\"") { "\"\"\"" } else { "'''" };
                            let after_start = trimmed.trim_start_matches(quote);

                            if after_start.ends_with(quote) {
                                return Some(after_start.trim_end_matches(quote).trim().to_string());
                            } else if !after_start.is_empty() {
                                docstring_lines.push(after_start);
                            }
                        } else {
                            break;
                        }
                    } else {
                        if trimmed.ends_with("\"\"\"") || trimmed.ends_with("'''") {
                            break;
                        } else {
                            docstring_lines.push(trimmed);
                        }
                    }
                }

                if docstring_lines.is_empty() {
                    None
                } else {
                    Some(docstring_lines.into_iter().take(2).collect::<Vec<_>>().join(" "))
                }
            }
            _ => None,
        }
    }

    /// Find ranges not covered by items
    fn find_uncovered_ranges(&self, total_len: usize, covered: &[(usize, usize)]) -> Vec<(usize, usize)> {
        let mut uncovered = Vec::new();
        let mut pos = 0;

        let mut sorted = covered.to_vec();
        sorted.sort_by_key(|r| r.0);

        for (start, end) in sorted {
            if pos < start {
                uncovered.push((pos, start));
            }
            pos = pos.max(end);
        }

        if pos < total_len {
            uncovered.push((pos, total_len));
        }

        uncovered
    }

    /// Fallback to simple line-based chunking
    fn fallback_line_chunks(&self, file_path: &Path, content: &str, language: &str) -> Vec<Chunk> {
        let file_path_str = file_path.to_string_lossy().to_string();
        let lines: Vec<&str> = content.lines().collect();
        let file_context = self.extract_file_context(content, language);
        let context_prefix = self.build_context_prefix(&file_path_str, &file_context, None);
        let context_size = context_prefix.len();

        let lines_per_chunk = 40;
        let overlap = 10;
        // Effective budget accounts for actual context prefix size
        let effective_budget = self.max_chars.saturating_sub(context_size);

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < lines.len() {
            let end = (start + lines_per_chunk).min(lines.len());
            let chunk_lines = &lines[start..end];
            let chunk_text = chunk_lines.join("\n");

            // Respect effective budget
            let truncated = if chunk_text.len() > effective_budget {
                // Find a good break point at line boundary
                let mut break_at = effective_budget;
                if let Some(last_newline) = chunk_text[..effective_budget].rfind('\n') {
                    break_at = last_newline;
                }
                &chunk_text[..break_at]
            } else {
                &chunk_text
            };

            if truncated.len() >= self.min_chars {
                let chunk_content = format!("{}{}", context_prefix, truncated);

                chunks.push(Chunk::new(
                    &file_path_str,
                    chunk_content,
                    language,
                    start + 1,
                    end,
                    ContentType::Code,
                ));
            }

            start += lines_per_chunk - overlap;
        }

        if chunks.is_empty() && content.len() >= self.min_chars {
            let truncated = if content.len() > effective_budget {
                &content[..effective_budget]
            } else {
                content
            };
            let chunk_content = format!("{}{}", context_prefix, truncated);

            chunks.push(Chunk::new(
                &file_path_str,
                chunk_content,
                language,
                1,
                lines.len(),
                ContentType::Code,
            ));
        }

        chunks
    }
}

/// Represents a code item (function, class, etc.) from the AST
#[derive(Debug)]
struct CodeItem<'a> {
    /// The AST node
    node: Node<'a>,
    /// Kind of item
    kind: ItemKind,
    /// Signature (without body)
    signature: String,
    /// Parent context (e.g., class name for methods)
    parent_context: Option<String>,
    /// Byte range in source
    start_byte: usize,
    end_byte: usize,
    /// Child items (methods inside class, etc.)
    children: Vec<CodeItem<'a>>,
}

#[derive(Debug, Clone, Copy)]
enum ItemKind {
    Function,
    Class,
    ImplBlock,
    TypeDef,
    Module,
}

// =============================================================================
// HEADER-BASED SEMANTIC CHUNKER (for docs)
// =============================================================================

/// Header-based chunker with semantic breadcrumb context
///
/// Key features:
/// - Splits at markdown headers (#, ##, ###)
/// - Injects header breadcrumb into each chunk
/// - Handles oversized sections by splitting at paragraphs
pub struct HeaderSemanticChunker {
    /// Maximum total characters per chunk (content + context)
    max_chars: usize,
    /// Maximum characters reserved for breadcrumb context
    context_budget: usize,
    /// Minimum characters for a valid chunk
    min_chars: usize,
}

impl HeaderSemanticChunker {
    pub fn new(max_chars: usize, context_budget: usize) -> Self {
        Self {
            max_chars,
            context_budget,
            min_chars: MIN_CHUNK_CHARS,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_MAX_CHARS, DEFAULT_CONTEXT_BUDGET)
    }

    /// Create chunks from markdown with breadcrumb context
    pub fn create_chunks(&self, file_path: &Path, content: &str) -> Vec<Chunk> {
        let file_path_str = file_path.to_string_lossy().to_string();
        let sections = self.parse_sections(content);

        if sections.is_empty() {
            return self.fallback_line_chunks(file_path, content);
        }

        let content_budget = self.max_chars - self.context_budget;
        let mut chunks = Vec::new();

        for section in sections {
            let section_chunks = self.chunk_section(
                &file_path_str,
                content,
                &section,
                content_budget,
            );
            chunks.extend(section_chunks);
        }

        if chunks.is_empty() {
            return self.fallback_line_chunks(file_path, content);
        }

        chunks
    }

    /// Parse markdown into sections based on headers
    fn parse_sections(&self, content: &str) -> Vec<Section> {
        let lines: Vec<&str> = content.lines().collect();
        let mut sections = Vec::new();
        let mut current_headers: Vec<(usize, String)> = Vec::new(); // (level, header_text)
        let mut current_content = Vec::new();
        let mut section_start_line = 0;
        let mut in_code_block = false;

        for (line_num, line) in lines.iter().enumerate() {
            // Track code blocks
            if line.starts_with("```") {
                in_code_block = !in_code_block;
            }

            let is_header = !in_code_block && line.starts_with('#');

            if is_header {
                let level = line.chars().take_while(|&c| c == '#').count();
                let header_text = line.trim_start_matches('#').trim().to_string();

                // Save current section if it has content
                if !current_content.is_empty() {
                    let breadcrumb = self.build_breadcrumb(&current_headers);
                    sections.push(Section {
                        breadcrumb,
                        content: current_content.join("\n"),
                        start_line: section_start_line,
                        end_line: line_num,
                    });
                }

                // Update header stack
                while current_headers.last().map(|(l, _)| *l >= level).unwrap_or(false) {
                    current_headers.pop();
                }
                current_headers.push((level, header_text.clone()));

                // Start new section
                current_content = vec![line.to_string()];
                section_start_line = line_num;
            } else {
                current_content.push(line.to_string());
            }
        }

        // Don't forget last section
        if !current_content.is_empty() {
            let breadcrumb = self.build_breadcrumb(&current_headers);
            sections.push(Section {
                breadcrumb,
                content: current_content.join("\n"),
                start_line: section_start_line,
                end_line: lines.len(),
            });
        }

        sections
    }

    /// Build breadcrumb from header stack
    fn build_breadcrumb(&self, headers: &[(usize, String)]) -> String {
        headers
            .iter()
            .map(|(level, text)| format!("{} {}", "#".repeat(*level), text))
            .collect::<Vec<_>>()
            .join(" > ")
    }

    /// Chunk a section with breadcrumb context
    fn chunk_section(
        &self,
        file_path: &str,
        _content: &str,
        section: &Section,
        content_budget: usize,
    ) -> Vec<Chunk> {
        let mut chunks = Vec::new();

        // If section fits, keep it whole
        if section.content.len() <= content_budget {
            let context_prefix = self.build_context_prefix(file_path, &section.breadcrumb, false);
            let chunk_content = format!("{}{}", context_prefix, section.content);

            chunks.push(Chunk::new(
                file_path,
                chunk_content,
                "markdown",
                section.start_line + 1,
                section.end_line,
                ContentType::Document,
            ));
            return chunks;
        }

        // Section too big - split at paragraphs
        let paragraphs = self.split_into_paragraphs(&section.content);
        let mut current_chunk = Vec::new();
        let mut current_size = 0;
        let mut chunk_start_line = section.start_line;
        let mut is_continuation = false;

        for para in &paragraphs {
            let para_size = para.len();

            // If single paragraph exceeds budget, split by sentences
            if para_size > content_budget {
                // Save current chunk first
                if !current_chunk.is_empty() {
                    let chunk_text = current_chunk.join("\n\n");
                    let context_prefix = self.build_context_prefix(file_path, &section.breadcrumb, is_continuation);
                    let chunk_content = format!("{}{}", context_prefix, chunk_text);

                    chunks.push(Chunk::new(
                        file_path,
                        chunk_content,
                        "markdown",
                        chunk_start_line + 1,
                        chunk_start_line + current_chunk.len(),
                        ContentType::Document,
                    ));

                    current_chunk.clear();
                    current_size = 0;
                    is_continuation = true;
                }

                // Split big paragraph by sentences
                let sentence_chunks = self.split_paragraph_by_sentences(
                    file_path,
                    para,
                    &section.breadcrumb,
                    content_budget,
                    chunk_start_line,
                );
                chunks.extend(sentence_chunks);
                is_continuation = true;
                continue;
            }

            if current_size + para_size > content_budget && !current_chunk.is_empty() {
                // Save current chunk
                let chunk_text = current_chunk.join("\n\n");
                let context_prefix = self.build_context_prefix(file_path, &section.breadcrumb, is_continuation);
                let chunk_content = format!("{}{}", context_prefix, chunk_text);

                chunks.push(Chunk::new(
                    file_path,
                    chunk_content,
                    "markdown",
                    chunk_start_line + 1,
                    chunk_start_line + current_chunk.len(),
                    ContentType::Document,
                ));

                current_chunk.clear();
                current_size = 0;
                chunk_start_line = section.start_line + chunks.len() * 10; // Approximate
                is_continuation = true;
            }

            current_chunk.push(para.clone());
            current_size += para_size + 2; // +2 for \n\n
        }

        // Don't forget last chunk
        if !current_chunk.is_empty() {
            let chunk_text = current_chunk.join("\n\n");
            if chunk_text.len() >= self.min_chars {
                let context_prefix = self.build_context_prefix(file_path, &section.breadcrumb, is_continuation);
                let chunk_content = format!("{}{}", context_prefix, chunk_text);

                chunks.push(Chunk::new(
                    file_path,
                    chunk_content,
                    "markdown",
                    chunk_start_line + 1,
                    section.end_line,
                    ContentType::Document,
                ));
            }
        }

        chunks
    }

    /// Split content into paragraphs
    fn split_into_paragraphs(&self, content: &str) -> Vec<String> {
        content
            .split("\n\n")
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Split a large paragraph by sentences
    fn split_paragraph_by_sentences(
        &self,
        file_path: &str,
        paragraph: &str,
        breadcrumb: &str,
        content_budget: usize,
        start_line: usize,
    ) -> Vec<Chunk> {
        let mut chunks = Vec::new();

        // Simple sentence splitting (could be improved with better heuristics)
        let sentences: Vec<&str> = paragraph
            .split(". ")
            .filter(|s| !s.is_empty())
            .collect();

        let mut current_chunk = Vec::new();
        let mut current_size = 0;

        for sentence in sentences {
            let sentence_size = sentence.len() + 2;

            if current_size + sentence_size > content_budget && !current_chunk.is_empty() {
                let chunk_text = current_chunk.join(". ");
                let context_prefix = self.build_context_prefix(file_path, breadcrumb, true);
                let chunk_content = format!("{}{}", context_prefix, chunk_text);

                if chunk_text.len() >= self.min_chars {
                    chunks.push(Chunk::new(
                        file_path,
                        chunk_content,
                        "markdown",
                        start_line + 1,
                        start_line + 1,
                        ContentType::Document,
                    ));
                }

                current_chunk.clear();
                current_size = 0;
            }

            current_chunk.push(sentence);
            current_size += sentence_size;
        }

        // Last chunk
        if !current_chunk.is_empty() {
            let chunk_text = current_chunk.join(". ");
            if chunk_text.len() >= self.min_chars {
                let context_prefix = self.build_context_prefix(file_path, breadcrumb, true);
                let chunk_content = format!("{}{}", context_prefix, chunk_text);

                chunks.push(Chunk::new(
                    file_path,
                    chunk_content,
                    "markdown",
                    start_line + 1,
                    start_line + 1,
                    ContentType::Document,
                ));
            }
        }

        chunks
    }

    /// Build context prefix with breadcrumb
    fn build_context_prefix(&self, file_path: &str, breadcrumb: &str, is_continuation: bool) -> String {
        let mut prefix = format!("[file: {}]", file_path);

        if !breadcrumb.is_empty() {
            let suffix = if is_continuation { " (continued)" } else { "" };
            prefix.push_str(&format!("\n[section: {}{}]", breadcrumb, suffix));
        }

        prefix.push_str("\n\n");
        prefix
    }

    /// Fallback to simple line-based chunking
    fn fallback_line_chunks(&self, file_path: &Path, content: &str) -> Vec<Chunk> {
        let file_path_str = file_path.to_string_lossy().to_string();
        let lines: Vec<&str> = content.lines().collect();

        let lines_per_chunk = 50;
        let overlap = 10;
        let content_budget = self.max_chars - self.context_budget;

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < lines.len() {
            let end = (start + lines_per_chunk).min(lines.len());
            let chunk_lines = &lines[start..end];
            let chunk_text = chunk_lines.join("\n");

            let truncated = if chunk_text.len() > content_budget {
                &chunk_text[..content_budget]
            } else {
                &chunk_text
            };

            if truncated.len() >= self.min_chars {
                let context_prefix = format!("[file: {}]\n\n", file_path_str);
                let chunk_content = format!("{}{}", context_prefix, truncated);

                chunks.push(Chunk::new(
                    &file_path_str,
                    chunk_content,
                    "markdown",
                    start + 1,
                    end,
                    ContentType::Document,
                ));
            }

            start += lines_per_chunk - overlap;
        }

        chunks
    }
}

/// A section of a markdown document
#[derive(Debug)]
struct Section {
    /// Header breadcrumb (e.g., "# API > ## Auth > ### OAuth")
    breadcrumb: String,
    /// Section content including header
    content: String,
    /// Start line in source file
    start_line: usize,
    /// End line in source file
    end_line: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_ast_semantic_rust_function() {
        let code = r#"
/// A simple function
fn hello(name: &str) -> String {
    format!("Hello, {}!", name)
}
"#;

        let chunker = AstSemanticChunker::with_defaults();
        let chunks = chunker.create_chunks(
            &PathBuf::from("test.rs"),
            code,
            "rust",
            "rs",
        );

        assert!(!chunks.is_empty());
        // Should have context prefix
        assert!(chunks[0].content.contains("[file:"));
    }

    #[test]
    fn test_header_semantic_markdown() {
        let markdown = r#"
# Main Title

Introduction.

## Section 1

Content for section 1.

### Subsection 1.1

Details here.

## Section 2

More content.
"#;

        let chunker = HeaderSemanticChunker::with_defaults();
        let chunks = chunker.create_chunks(&PathBuf::from("test.md"), markdown);

        assert!(!chunks.is_empty());
        // Should have breadcrumb context
        for chunk in &chunks {
            assert!(chunk.content.contains("[file:") || chunk.content.contains("[section:"));
        }
    }
}
