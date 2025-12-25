//! Benchmark queries with ground truth
//!
//! Provides queries across 4 difficulty levels for semantic search evaluation.
//!
//! ## Query Design Principles (Based on BEIR/MTEB methodology)
//!
//! 1. **Verified Ground Truth**: Each query has manually verified expected files
//! 2. **Diverse Difficulty**: Simple (keyword) → Medium (semantic) → Hard (cross-domain) → ExtraHard (architectural)
//! 3. **Multiple Relevant Files**: Most queries have 2+ relevant files for better nDCG calculation
//! 4. **Chunk-Realistic**: Queries match content that exists within file-level context
//!
//! ## Query File Format (JSON)
//!
//! Queries can be loaded from a JSON file with the following format:
//!
//! ```json
//! {
//!   "metadata": {
//!     "name": "my-project",
//!     "description": "Queries for my-project codebase",
//!     "version": "1.0"
//!   },
//!   "queries": [
//!     {
//!       "query": "authentication middleware",
//!       "difficulty": "medium",
//!       "query_type": "code",
//!       "expected_files": ["src/auth/middleware.ts", "src/auth/jwt.ts"],
//!       "description": "Find auth middleware implementation"
//!     }
//!   ]
//! }
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A valid answer region - defines criteria for matching a correct answer chunk
///
/// A chunk matches this region if:
/// - Keywords match (if specified): chunk body contains at least one keyword
/// - Lines match (if specified): chunk overlaps with the line range
/// - File matches (if specified): chunk is from a file matching the pattern
///
/// All specified criteria must be satisfied (AND logic within a region).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnswerRegion {
    /// Keywords that should appear in the answer (at least one must match)
    #[serde(default)]
    pub keywords: Vec<String>,
    /// Line range [start, end] where the answer is located
    #[serde(default)]
    pub lines: Option<(usize, usize)>,
    /// Optional file pattern to restrict this region to specific files
    /// (matches if file path contains this pattern)
    #[serde(default)]
    pub file_pattern: Option<String>,
}

impl AnswerRegion {
    /// Check if a chunk matches this answer region
    pub fn matches(&self, chunk_body: &str, chunk_file: &str, chunk_start: usize, chunk_end: usize) -> bool {
        // Check file pattern (if specified)
        if let Some(pattern) = &self.file_pattern {
            if !chunk_file.contains(pattern) {
                return false;
            }
        }

        // Check keywords (if specified) - at least one must match
        let keywords_match = if self.keywords.is_empty() {
            true // No keyword constraint
        } else {
            self.keywords.iter().any(|kw| chunk_body.contains(kw))
        };

        if !keywords_match {
            return false;
        }

        // Check line range (if specified)
        if let Some((start, end)) = self.lines {
            chunk_start <= end && chunk_end >= start
        } else {
            true // No line constraint
        }
    }

    /// Check if this region has any criteria defined
    pub fn has_criteria(&self) -> bool {
        !self.keywords.is_empty() || self.lines.is_some()
    }
}

/// A benchmark query with expected results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkQuery {
    /// Query text
    pub query: String,
    /// Difficulty level
    pub difficulty: QueryDifficulty,
    /// Query type (code or documentation)
    pub query_type: QueryType,
    /// Expected file matches (relative paths) - ordered by relevance
    pub expected_files: Vec<String>,
    /// Description of what the query is testing
    pub description: String,

    // === Answer-level ground truth (for fine-grained evaluation) ===
    // Two formats supported:
    // 1. Legacy: answer_keywords + answer_lines (single region)
    // 2. Multi-region: answer_regions (multiple valid answer locations)
    //
    // If answer_regions is non-empty, it takes precedence.
    // A chunk matches if it matches ANY region (OR logic across regions).

    /// [Legacy] Keywords that should appear in the answer chunk
    #[serde(default)]
    pub answer_keywords: Vec<String>,
    /// [Legacy] Expected line range [start, end] where the answer is located
    #[serde(default)]
    pub answer_lines: Option<(usize, usize)>,
    /// [Multi-region] Multiple valid answer regions (OR logic - match any)
    #[serde(default)]
    pub answer_regions: Vec<AnswerRegion>,
}

/// Query difficulty level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryDifficulty {
    Simple,
    Medium,
    Hard,
    ExtraHard,
}

impl QueryDifficulty {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Simple => "simple",
            Self::Medium => "medium",
            Self::Hard => "hard",
            Self::ExtraHard => "extra_hard",
        }
    }

    /// Parse from string (case-insensitive)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "simple" => Some(Self::Simple),
            "medium" => Some(Self::Medium),
            "hard" => Some(Self::Hard),
            "extra_hard" | "extrahard" | "extra-hard" => Some(Self::ExtraHard),
            _ => None,
        }
    }
}

/// Metadata about a query file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFileMetadata {
    /// Name of the corpus/project
    pub name: String,
    /// Optional description
    #[serde(default)]
    pub description: String,
    /// Optional version
    #[serde(default)]
    pub version: String,
}

/// A complete query file that can be loaded from JSON
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFile {
    /// Metadata about the query set
    pub metadata: QueryFileMetadata,
    /// The queries
    pub queries: Vec<BenchmarkQuery>,
}

impl QueryFile {
    /// Load queries from a JSON file
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read query file: {}", path.display()))?;

        let query_file: QueryFile = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse query file: {}", path.display()))?;

        // Validate queries
        for (i, q) in query_file.queries.iter().enumerate() {
            if q.query.is_empty() {
                anyhow::bail!("Query {} has empty query text", i);
            }
            if q.expected_files.is_empty() {
                anyhow::bail!("Query '{}' has no expected files", q.query);
            }
        }

        Ok(query_file)
    }

    /// Save queries to a JSON file
    pub fn save(&self, path: &Path) -> Result<()> {
        let content = serde_json::to_string_pretty(self)
            .context("Failed to serialize query file")?;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(path, content)
            .with_context(|| format!("Failed to write query file: {}", path.display()))?;

        Ok(())
    }

    /// Create a new query file
    pub fn new(name: impl Into<String>, queries: Vec<BenchmarkQuery>) -> Self {
        Self {
            metadata: QueryFileMetadata {
                name: name.into(),
                description: String::new(),
                version: "1.0".to_string(),
            },
            queries,
        }
    }
}

/// Query type - what kind of content we're searching for
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryType {
    /// Searching for code (functions, structs, implementations)
    Code,
    /// Searching for documentation (README, design docs, comments)
    Doc,
}

impl QueryType {
    /// Get the string name of this query type
    pub fn name(&self) -> &'static str {
        match self {
            Self::Code => "code",
            Self::Doc => "doc",
        }
    }

    /// Check if this query type matches a content type string
    pub fn matches_content_type(&self, content_type: &str) -> bool {
        match self {
            Self::Code => content_type == "code",
            Self::Doc => content_type == "doc" || content_type == "markdown",
        }
    }
}

impl BenchmarkQuery {
    /// Create a code search query
    pub fn code(
        query: impl Into<String>,
        difficulty: QueryDifficulty,
        expected_files: Vec<&str>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            query: query.into(),
            difficulty,
            query_type: QueryType::Code,
            expected_files: expected_files.into_iter().map(String::from).collect(),
            description: description.into(),
            answer_keywords: Vec::new(),
            answer_lines: None,
            answer_regions: Vec::new(),
        }
    }

    /// Create a documentation search query
    pub fn doc(
        query: impl Into<String>,
        difficulty: QueryDifficulty,
        expected_files: Vec<&str>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            query: query.into(),
            difficulty,
            query_type: QueryType::Doc,
            expected_files: expected_files.into_iter().map(String::from).collect(),
            description: description.into(),
            answer_keywords: Vec::new(),
            answer_lines: None,
            answer_regions: Vec::new(),
        }
    }

    /// Check if a chunk matches the expected answer criteria
    ///
    /// Supports two formats:
    /// 1. Multi-region (preferred): answer_regions - match if ANY region matches (OR logic)
    /// 2. Legacy: answer_keywords + answer_lines - single region
    ///
    /// Within each region, all specified criteria must match (AND logic).
    pub fn chunk_matches_answer(
        &self,
        chunk_content: &str,
        chunk_file: &str,
        chunk_start: usize,
        chunk_end: usize,
    ) -> bool {
        // Strip context prefix to get body for keyword matching
        let body = Self::strip_context_prefix(chunk_content);

        // If multi-region format is used, check against all regions (OR logic)
        if !self.answer_regions.is_empty() {
            return self.answer_regions.iter().any(|region| {
                region.has_criteria() && region.matches(body, chunk_file, chunk_start, chunk_end)
            });
        }

        // Fall back to legacy format (single region from answer_keywords + answer_lines)
        let has_keywords = !self.answer_keywords.is_empty();
        let has_lines = self.answer_lines.is_some();

        // If no answer criteria specified, can't match
        if !has_keywords && !has_lines {
            return false;
        }

        // Check keywords - at least one must match
        let keywords_match = if has_keywords {
            self.answer_keywords.iter().any(|kw| body.contains(kw))
        } else {
            true // No keywords constraint
        };

        if !keywords_match {
            return false;
        }

        // Check line range overlap
        if let Some((start, end)) = self.answer_lines {
            chunk_start <= end && chunk_end >= start
        } else {
            true // No line constraint
        }
    }

    /// Strip the context prefix from chunk content to get just the code body.
    /// Context prefix format: "[file: path] description\n\n" or "[file: path]\n\n"
    fn strip_context_prefix(content: &str) -> &str {
        // Find the first "\n\n" which marks end of context prefix
        if let Some(pos) = content.find("\n\n") {
            &content[pos + 2..]
        } else {
            // No prefix found, return whole content
            content
        }
    }

    /// Check if this query has answer-level ground truth
    pub fn has_answer_ground_truth(&self) -> bool {
        // Check multi-region format
        if self.answer_regions.iter().any(|r| r.has_criteria()) {
            return true;
        }
        // Check legacy format
        !self.answer_keywords.is_empty() || self.answer_lines.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // ANSWER MATCHING TESTS
    // =========================================================================

    #[test]
    fn test_strip_context_prefix() {
        // Standard prefix with description
        let content = "[file: src/lib.rs] Test library for benchmark\n\nfn main() {}";
        assert_eq!(BenchmarkQuery::strip_context_prefix(content), "fn main() {}");

        // Prefix without description
        let content = "[file: src/lib.rs]\n\nfn hello() {}";
        assert_eq!(BenchmarkQuery::strip_context_prefix(content), "fn hello() {}");

        // No prefix (raw content)
        let content = "fn raw() {}";
        assert_eq!(BenchmarkQuery::strip_context_prefix(content), "fn raw() {}");
    }

    #[test]
    fn test_chunk_matches_answer_keywords_only() {
        let query = BenchmarkQuery {
            query: "test".to_string(),
            difficulty: QueryDifficulty::Simple,
            query_type: QueryType::Code,
            expected_files: vec![],
            description: "test".to_string(),
            answer_keywords: vec!["ConfigBuilder".to_string(), "builder".to_string()],
            answer_lines: None,
            answer_regions: vec![],
        };

        // Match in body (after prefix)
        let content = "[file: src/lib.rs]\n\nstruct ConfigBuilder { }";
        assert!(query.chunk_matches_answer(content, "src/lib.rs", 1, 50));

        // Match another keyword
        let content = "[file: src/lib.rs]\n\nimpl builder {}";
        assert!(query.chunk_matches_answer(content, "src/lib.rs", 1, 50));

        // No match
        let content = "[file: src/lib.rs]\n\nstruct Config { }";
        assert!(!query.chunk_matches_answer(content, "src/lib.rs", 1, 50));
    }

    #[test]
    fn test_chunk_matches_answer_no_false_positive_from_prefix() {
        // Keyword matches filename in prefix but NOT in body - should NOT match
        let query = BenchmarkQuery {
            query: "test".to_string(),
            difficulty: QueryDifficulty::Simple,
            query_type: QueryType::Code,
            expected_files: vec![],
            description: "test".to_string(),
            answer_keywords: vec!["lib".to_string()],
            answer_lines: None,
            answer_regions: vec![],
        };

        // "lib" appears in prefix [file: src/lib.rs] but NOT in body
        let content = "[file: src/lib.rs] Test library\n\nfn main() {}";
        assert!(!query.chunk_matches_answer(content, "src/lib.rs", 1, 50));

        // "lib" also appears in body - should match
        let content = "[file: src/lib.rs]\n\nuse lib::foo;";
        assert!(query.chunk_matches_answer(content, "src/lib.rs", 1, 50));
    }

    #[test]
    fn test_chunk_matches_answer_lines_only() {
        let query = BenchmarkQuery {
            query: "test".to_string(),
            difficulty: QueryDifficulty::Simple,
            query_type: QueryType::Code,
            expected_files: vec![],
            description: "test".to_string(),
            answer_keywords: vec![],
            answer_lines: Some((50, 100)),
            answer_regions: vec![],
        };

        // Chunk overlaps with expected lines
        let content = "[file: src/lib.rs]\n\nsome code";
        assert!(query.chunk_matches_answer(content, "src/lib.rs", 40, 60)); // overlaps 50-60
        assert!(query.chunk_matches_answer(content, "src/lib.rs", 50, 100)); // exact match
        assert!(query.chunk_matches_answer(content, "src/lib.rs", 80, 120)); // overlaps 80-100

        // Chunk does NOT overlap
        assert!(!query.chunk_matches_answer(content, "src/lib.rs", 1, 49)); // before
        assert!(!query.chunk_matches_answer(content, "src/lib.rs", 101, 150)); // after
    }

    #[test]
    fn test_chunk_matches_answer_keywords_and_lines() {
        let query = BenchmarkQuery {
            query: "test".to_string(),
            difficulty: QueryDifficulty::Simple,
            query_type: QueryType::Code,
            expected_files: vec![],
            description: "test".to_string(),
            answer_keywords: vec!["retry".to_string()],
            answer_lines: Some((100, 150)),
            answer_regions: vec![],
        };

        // Both match - should pass
        let content = "[file: src/lib.rs]\n\nfn retry() {}";
        assert!(query.chunk_matches_answer(content, "src/lib.rs", 100, 150));

        // Keyword matches but lines don't - should FAIL
        assert!(!query.chunk_matches_answer(content, "src/lib.rs", 1, 50));

        // Lines match but keyword doesn't - should FAIL
        let content = "[file: src/lib.rs]\n\nfn backoff() {}";
        assert!(!query.chunk_matches_answer(content, "src/lib.rs", 100, 150));
    }

    #[test]
    fn test_chunk_matches_answer_no_ground_truth() {
        let query = BenchmarkQuery {
            query: "test".to_string(),
            difficulty: QueryDifficulty::Simple,
            query_type: QueryType::Code,
            expected_files: vec![],
            description: "test".to_string(),
            answer_keywords: vec![],
            answer_lines: None,
            answer_regions: vec![],
        };

        // No ground truth - always false
        let content = "[file: src/lib.rs]\n\nanything";
        assert!(!query.chunk_matches_answer(content, "src/lib.rs", 1, 100));
    }

    #[test]
    fn test_chunk_matches_answer_multi_region() {
        // Multi-region: two valid answer locations
        let query = BenchmarkQuery {
            query: "auth logic".to_string(),
            difficulty: QueryDifficulty::Medium,
            query_type: QueryType::Code,
            expected_files: vec![],
            description: "test".to_string(),
            answer_keywords: vec![],
            answer_lines: None,
            answer_regions: vec![
                AnswerRegion {
                    keywords: vec!["authenticate".to_string()],
                    lines: Some((50, 80)),
                    file_pattern: Some("login".to_string()),
                },
                AnswerRegion {
                    keywords: vec!["verify_token".to_string()],
                    lines: Some((20, 60)),
                    file_pattern: Some("middleware".to_string()),
                },
            ],
        };

        // Match first region
        let content = "[file: auth/login.rs]\n\nfn authenticate() {}";
        assert!(query.chunk_matches_answer(content, "auth/login.rs", 50, 80));

        // Match second region
        let content = "[file: auth/middleware.rs]\n\nfn verify_token() {}";
        assert!(query.chunk_matches_answer(content, "auth/middleware.rs", 30, 50));

        // Wrong file pattern - should NOT match even with keyword
        let content = "[file: auth/other.rs]\n\nfn authenticate() {}";
        assert!(!query.chunk_matches_answer(content, "auth/other.rs", 50, 80));

        // Wrong keyword - should NOT match
        let content = "[file: auth/login.rs]\n\nfn login() {}";
        assert!(!query.chunk_matches_answer(content, "auth/login.rs", 50, 80));
    }

    #[test]
    fn test_chunk_matches_answer_multi_region_no_file_pattern() {
        // Multi-region without file pattern restriction
        let query = BenchmarkQuery {
            query: "error handling".to_string(),
            difficulty: QueryDifficulty::Simple,
            query_type: QueryType::Code,
            expected_files: vec![],
            description: "test".to_string(),
            answer_keywords: vec![],
            answer_lines: None,
            answer_regions: vec![
                AnswerRegion {
                    keywords: vec!["Error".to_string()],
                    lines: Some((10, 30)),
                    file_pattern: None, // Any file
                },
                AnswerRegion {
                    keywords: vec!["Result".to_string()],
                    lines: None, // Any line
                    file_pattern: None,
                },
            ],
        };

        // Match first region (any file with Error at lines 10-30)
        let content = "[file: src/any.rs]\n\nenum Error {}";
        assert!(query.chunk_matches_answer(content, "src/any.rs", 15, 25));

        // Match second region (any file with Result at any line)
        let content = "[file: src/other.rs]\n\ntype Result<T> = ...";
        assert!(query.chunk_matches_answer(content, "src/other.rs", 100, 150));

        // No match - wrong keyword and wrong lines
        let content = "[file: src/any.rs]\n\nfn foo() {}";
        assert!(!query.chunk_matches_answer(content, "src/any.rs", 50, 80));
    }

    #[test]
    fn test_has_answer_ground_truth() {
        // Keywords only
        let mut query = BenchmarkQuery::code("test", QueryDifficulty::Simple, vec![], "desc");
        query.answer_keywords = vec!["keyword".to_string()];
        assert!(query.has_answer_ground_truth());

        // Lines only
        let mut query = BenchmarkQuery::code("test", QueryDifficulty::Simple, vec![], "desc");
        query.answer_lines = Some((10, 20));
        assert!(query.has_answer_ground_truth());

        // Both
        let mut query = BenchmarkQuery::code("test", QueryDifficulty::Simple, vec![], "desc");
        query.answer_keywords = vec!["keyword".to_string()];
        query.answer_lines = Some((10, 20));
        assert!(query.has_answer_ground_truth());

        // Neither
        let query = BenchmarkQuery::code("test", QueryDifficulty::Simple, vec![], "desc");
        assert!(!query.has_answer_ground_truth());
    }
}
