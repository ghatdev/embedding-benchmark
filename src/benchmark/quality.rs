//! Quality metrics for semantic search evaluation
//!
//! Implements Precision@K, Mean Reciprocal Rank (MRR), nDCG@K, and score distribution analysis.
//!
//! ## Metrics Overview
//!
//! - **Precision@K**: Fraction of relevant items in top K results
//! - **MRR**: 1/rank of first relevant result (averaged across queries)
//! - **nDCG@K**: Normalized Discounted Cumulative Gain - accounts for position of ALL
//!   relevant items, not just the first. This is the BEIR/MTEB standard metric.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// A search result with relevance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Document/chunk identifier
    pub id: String,
    /// File path
    pub file_path: String,
    /// Cosine similarity score (higher is more similar)
    pub score: f32,
    /// Rank in results (1-indexed)
    pub rank: usize,
    /// Whether this result is relevant (ground truth) - for binary relevance
    pub relevant: bool,
    /// Graded relevance score (0-3 for CodeSearchNet)
    /// 0 = Irrelevant, 1 = Somewhat, 2 = Relevant, 3 = Very Relevant
    /// Defaults to 0 if not using graded relevance
    #[serde(default)]
    pub relevance_grade: u8,
}

impl SearchResult {
    /// Create a result with binary relevance (for backward compatibility)
    pub fn binary(id: String, file_path: String, score: f32, rank: usize, relevant: bool) -> Self {
        Self {
            id,
            file_path,
            score,
            rank,
            relevant,
            relevance_grade: if relevant { 1 } else { 0 },
        }
    }

    /// Create a result with graded relevance
    pub fn graded(id: String, file_path: String, score: f32, rank: usize, grade: u8) -> Self {
        Self {
            id,
            file_path,
            score,
            rank,
            relevant: grade > 0,
            relevance_grade: grade,
        }
    }
}

// =============================================================================
// FILE-LEVEL EVALUATION (Recommended)
// =============================================================================

/// Normalize a file path for consistent matching
pub fn normalize_path(path: &str) -> String {
    path.replace('\\', "/")
}

/// A file-level search result (aggregated from chunks)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSearchResult {
    /// File path (normalized)
    pub file_path: String,
    /// Aggregated score (max of all chunks from this file)
    pub score: f32,
    /// Rank in file results (1-indexed)
    pub rank: usize,
    /// Whether this file is in expected_files
    pub relevant: bool,
    /// Number of chunks from this file in the candidate set
    pub chunk_count: usize,
    /// ID of the highest-scoring chunk
    pub top_chunk_id: String,
    /// Preview of the highest-scoring chunk (max 120 chars)
    pub top_chunk_preview: String,
}

impl FileSearchResult {
    /// Create a new file search result
    ///
    /// Note: `top_chunk_preview` should already be truncated/formatted.
    /// Use `create_preview()` to generate it before calling this.
    pub fn new(
        file_path: String,
        score: f32,
        rank: usize,
        relevant: bool,
        chunk_count: usize,
        top_chunk_id: String,
        top_chunk_preview: String, // Already formatted preview, not raw content
    ) -> Self {
        Self {
            file_path,
            score,
            rank,
            relevant,
            chunk_count,
            top_chunk_id,
            top_chunk_preview,
        }
    }
}

/// Index for efficient file path resolution
///
/// Supports O(1) lookups by exact path or basename.
#[derive(Debug, Clone)]
pub struct FileIndex {
    /// All normalized file paths
    all_files: HashSet<String>,
    /// Index by basename: "main.rs" → ["src/main.rs", "tests/main.rs"]
    by_basename: HashMap<String, Vec<String>>,
}

impl FileIndex {
    /// Build index from file paths
    pub fn new(files: impl Iterator<Item = String>) -> Self {
        let mut all_files = HashSet::new();
        let mut by_basename: HashMap<String, Vec<String>> = HashMap::new();

        for file in files {
            let normalized = normalize_path(&file);

            // Index by basename
            if let Some(basename) = normalized.rsplit('/').next() {
                by_basename
                    .entry(basename.to_string())
                    .or_default()
                    .push(normalized.clone());
            }

            all_files.insert(normalized);
        }

        Self { all_files, by_basename }
    }

    /// Check if a file exists in the index
    pub fn contains(&self, path: &str) -> bool {
        self.all_files.contains(&normalize_path(path))
    }

    /// Get all files in the index
    pub fn files(&self) -> &HashSet<String> {
        &self.all_files
    }

    /// Resolve an expected file to corpus paths
    ///
    /// Resolution order:
    /// 1. Exact match (normalized)
    /// 2. Basename match with suffix filter
    ///
    /// Returns:
    /// - Ok(path) if exactly one match
    /// - Err(FileResolution::NotFound) if no match
    /// - Err(FileResolution::Ambiguous(matches)) if multiple matches
    pub fn resolve(&self, expected: &str) -> Result<String, FileResolution> {
        let normalized_expected = normalize_path(expected);

        // 1. Try exact match
        if self.all_files.contains(&normalized_expected) {
            return Ok(normalized_expected);
        }

        // 2. Try basename match
        let basename = normalized_expected.rsplit('/').next().unwrap_or(&normalized_expected);

        if let Some(candidates) = self.by_basename.get(basename) {
            // Filter by suffix match - candidate must end with expected path
            // e.g., expected "http/routes.rs" matches "src/http/routes.rs"
            let matches: Vec<_> = candidates
                .iter()
                .filter(|c| c.ends_with(&normalized_expected))
                .cloned()
                .collect();

            match matches.len() {
                0 => {
                    // Basename matched but suffix didn't
                    // DO NOT fall back to looser match - this would cause false positives
                    // e.g., expected "auth/routes.rs" should NOT match "src/http/routes.rs"
                    // just because they share the same basename
                    if candidates.len() > 1 {
                        // Multiple candidates with same basename, none match suffix -> ambiguous
                        return Err(FileResolution::Ambiguous(candidates.clone()));
                    }
                    // Single candidate but suffix doesn't match -> not found
                    // (the basename match was a false positive)
                }
                1 => return Ok(matches[0].clone()),
                _ => return Err(FileResolution::Ambiguous(matches)),
            }
        }

        Err(FileResolution::NotFound)
    }

    /// Resolve multiple expected files, skipping ambiguous ones
    ///
    /// Returns (resolved_set, warnings)
    pub fn resolve_expected_files(
        &self,
        expected_files: &[String],
    ) -> (HashSet<String>, Vec<FileResolutionWarning>) {
        let mut resolved = HashSet::new();
        let mut warnings = Vec::new();

        for expected in expected_files {
            match self.resolve(expected) {
                Ok(path) => {
                    resolved.insert(path);
                }
                Err(FileResolution::NotFound) => {
                    warnings.push(FileResolutionWarning {
                        expected: expected.clone(),
                        issue: "Not found in corpus".to_string(),
                        matches: vec![],
                    });
                }
                Err(FileResolution::Ambiguous(matches)) => {
                    warnings.push(FileResolutionWarning {
                        expected: expected.clone(),
                        issue: format!("Ambiguous: {} matches", matches.len()),
                        matches,
                    });
                    // Skip ambiguous entries - don't add to resolved
                }
            }
        }

        (resolved, warnings)
    }
}

/// File resolution error
#[derive(Debug, Clone)]
pub enum FileResolution {
    NotFound,
    Ambiguous(Vec<String>),
}

/// Warning about file resolution issues
#[derive(Debug, Clone, Serialize)]
pub struct FileResolutionWarning {
    pub expected: String,
    pub issue: String,
    pub matches: Vec<String>,
}

// =============================================================================
// FILE-LEVEL METRICS (Correct approach for RAG evaluation)
// =============================================================================

/// Maximum length for chunk previews
const PREVIEW_MAX_LEN: usize = 120;

/// Create a preview string (max chars, single line, with ellipsis if truncated)
fn create_preview(content: &str, max_len: usize) -> String {
    // Take first line or first max_len chars, whichever is shorter
    let first_line = content.lines().next().unwrap_or("");
    let chars: Vec<char> = first_line.chars().take(max_len).collect();
    if first_line.chars().count() > max_len {
        format!("{}...", chars.into_iter().collect::<String>())
    } else {
        chars.into_iter().collect()
    }
}

/// Aggregate scored chunks into file-level results (zero-copy version)
///
/// This is the recommended function - it takes references to avoid cloning
/// full chunk content. Only creates preview strings when needed.
///
/// # Arguments
/// * `scored_chunks` - Iterator of (chunk_id, file_path, score, content) references
/// * `expected_files` - Set of resolved expected file paths
///
/// # Returns
/// File-level results sorted by score descending
pub fn aggregate_scored_chunks<'a, I>(
    scored_chunks: I,
    expected_files: &HashSet<String>,
) -> Vec<FileSearchResult>
where
    I: Iterator<Item = (&'a str, &'a str, f32, &'a str)>, // (id, file_path, score, content)
{
    // Group by file: (score, chunk_count, chunk_id, preview)
    let mut file_scores: HashMap<String, (f32, usize, String, String)> = HashMap::new();

    for (chunk_id, file_path, score, content) in scored_chunks {
        let normalized = normalize_path(file_path);

        match file_scores.get_mut(&normalized) {
            Some(entry) => {
                entry.1 += 1; // chunk count

                // Update to higher score if found
                if score > entry.0 {
                    entry.0 = score;
                    entry.2 = chunk_id.to_string();
                    // Only create preview when we actually update
                    entry.3 = create_preview(content, PREVIEW_MAX_LEN);
                }
            }
            None => {
                // First chunk for this file - create preview
                file_scores.insert(normalized, (
                    score,
                    1,
                    chunk_id.to_string(),
                    create_preview(content, PREVIEW_MAX_LEN),
                ));
            }
        }
    }

    finalize_file_results(file_scores, expected_files)
}

/// Aggregate chunk results into file-level results (owned version, for compatibility)
///
/// Groups chunks by file path, takes max score per file, and ranks files.
/// Uses the FileIndex to determine relevance.
///
/// # Arguments
/// * `chunks` - Chunk results with (id, file_path, score, content)
/// * `expected_files` - Set of resolved expected file paths
///
/// # Returns
/// File-level results sorted by score descending
pub fn aggregate_chunks_to_files(
    chunks: &[(String, String, f32, String)], // (id, file_path, score, content)
    expected_files: &HashSet<String>,
) -> Vec<FileSearchResult> {
    // Group by file, taking max score per file
    // Store only preview (not full content) to reduce memory for large corpora
    let mut file_scores: HashMap<String, (f32, usize, String, String)> = HashMap::new();

    for (chunk_id, file_path, score, content) in chunks {
        let normalized = normalize_path(file_path);

        match file_scores.get_mut(&normalized) {
            Some(entry) => {
                entry.1 += 1; // chunk count

                // Update to higher score if found
                if *score > entry.0 {
                    entry.0 = *score;
                    entry.2 = chunk_id.clone();
                    // Only create preview when we actually update
                    entry.3 = create_preview(content, PREVIEW_MAX_LEN);
                }
            }
            None => {
                // First chunk for this file - create preview
                file_scores.insert(normalized, (
                    *score,
                    1,
                    chunk_id.clone(),
                    create_preview(content, PREVIEW_MAX_LEN),
                ));
            }
        }
    }

    finalize_file_results(file_scores, expected_files)
}

/// Finalize aggregated file scores into sorted FileSearchResult list
fn finalize_file_results(
    file_scores: HashMap<String, (f32, usize, String, String)>,
    expected_files: &HashSet<String>,
) -> Vec<FileSearchResult> {

    // Sort by score descending, then by file path ascending for deterministic tie-break
    let mut files: Vec<_> = file_scores.into_iter().collect();
    files.sort_by(|a, b| {
        // Primary: score descending
        match b.1.0.partial_cmp(&a.1.0) {
            Some(std::cmp::Ordering::Equal) | None => {
                // Secondary: file path ascending (deterministic tie-break)
                a.0.cmp(&b.0)
            }
            Some(ord) => ord,
        }
    });

    // Build FileSearchResult list
    // Note: top_preview is already truncated with ellipsis by create_preview()
    files
        .into_iter()
        .enumerate()
        .map(|(i, (file_path, (score, chunk_count, top_chunk_id, top_preview)))| {
            let relevant = expected_files.contains(&file_path);
            FileSearchResult::new(
                file_path,
                score,
                i + 1, // 1-indexed rank
                relevant,
                chunk_count,
                top_chunk_id,
                top_preview, // Already formatted preview
            )
        })
        .collect()
}

/// Calculate file-level precision at K
///
/// Fraction of relevant files in top K results.
pub fn file_precision_at_k(results: &[FileSearchResult], k: usize) -> f32 {
    if k == 0 {
        return 0.0;
    }
    let relevant_count = results
        .iter()
        .take(k)
        .filter(|r| r.relevant)
        .count();
    relevant_count as f32 / k as f32
}

/// Calculate file-level reciprocal rank (MRR)
///
/// 1/rank of first relevant file. Returns 0 if no relevant file found.
pub fn file_mrr(results: &[FileSearchResult]) -> f32 {
    results
        .iter()
        .find(|r| r.relevant)
        .map(|r| 1.0 / r.rank as f32)
        .unwrap_or(0.0)
}

/// Calculate file-level recall at K
///
/// Fraction of ALL relevant files found in top K.
/// R = total_relevant (number of expected files that exist in searchable corpus)
pub fn file_recall_at_k(results: &[FileSearchResult], k: usize, total_relevant: usize) -> f32 {
    if total_relevant == 0 {
        return 0.0; // No relevant files exist
    }
    let found_relevant = results
        .iter()
        .take(k)
        .filter(|r| r.relevant)
        .count();
    found_relevant as f32 / total_relevant as f32
}

/// Calculate file-level nDCG at K
///
/// CRITICAL: IDCG uses `total_relevant` (R), NOT found count.
/// This is the correct formula per BEIR/MTEB methodology.
///
/// # Arguments
/// * `results` - File-level search results (ranked by score)
/// * `k` - Cutoff for DCG calculation
/// * `total_relevant` - R: total number of expected files in the searchable corpus
pub fn file_ndcg_at_k(results: &[FileSearchResult], k: usize, total_relevant: usize) -> f32 {
    if total_relevant == 0 {
        return 0.0; // No relevant files → can't compute nDCG
    }

    // DCG@k: sum of 1/log2(rank+1) for relevant results
    let dcg: f32 = results
        .iter()
        .take(k)
        .filter(|r| r.relevant)
        .map(|r| 1.0 / (r.rank as f32 + 1.0).log2())
        .sum();

    // IDCG@k: ideal DCG if all relevant files ranked at top
    let idcg: f32 = (0..total_relevant.min(k))
        .map(|i| 1.0 / (i as f32 + 2.0).log2())
        .sum();

    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

/// File-level query metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileQueryMetrics {
    /// Query text
    pub query: String,
    /// Query difficulty
    pub difficulty: String,
    /// Query type (code/doc)
    pub query_type: String,
    /// Total relevant files (R) after resolution
    pub total_relevant: usize,
    /// Warnings about file resolution
    pub resolution_warnings: Vec<String>,
    /// File Precision@1
    pub precision_at_1: f32,
    /// File Precision@5
    pub precision_at_5: f32,
    /// File Precision@10
    pub precision_at_10: f32,
    /// File MRR (reciprocal rank of first relevant file)
    pub mrr: f32,
    /// File Recall@10
    pub recall_at_10: f32,
    /// File nDCG@10 (PRIMARY METRIC)
    pub ndcg_at_10: f32,
    /// Top file score
    pub top_score: f32,
    /// Number of relevant files found in top 10
    pub relevant_in_top_10: usize,
    /// Top file results for inspection
    pub top_files: Vec<FileSearchResult>,
}

impl FileQueryMetrics {
    /// Evaluate a single query at file level
    ///
    /// # Arguments
    /// * `query` - Query text
    /// * `difficulty` - Query difficulty level
    /// * `query_type` - Query type (code/doc)
    /// * `file_results` - File-level search results (already aggregated from chunks)
    /// * `total_relevant` - R: number of expected files resolved successfully
    /// * `warnings` - Resolution warnings to record
    pub fn evaluate(
        query: &str,
        difficulty: &str,
        query_type: &str,
        file_results: &[FileSearchResult],
        total_relevant: usize,
        warnings: Vec<FileResolutionWarning>,
    ) -> Self {
        let warning_strings: Vec<String> = warnings
            .iter()
            .map(|w| format!("{}: {}", w.expected, w.issue))
            .collect();

        Self {
            query: query.to_string(),
            difficulty: difficulty.to_string(),
            query_type: query_type.to_string(),
            total_relevant,
            resolution_warnings: warning_strings,
            precision_at_1: file_precision_at_k(file_results, 1),
            precision_at_5: file_precision_at_k(file_results, 5),
            precision_at_10: file_precision_at_k(file_results, 10),
            mrr: file_mrr(file_results),
            recall_at_10: file_recall_at_k(file_results, 10, total_relevant),
            ndcg_at_10: file_ndcg_at_k(file_results, 10, total_relevant),
            top_score: file_results.first().map(|r| r.score).unwrap_or(0.0),
            relevant_in_top_10: file_results.iter().take(10).filter(|r| r.relevant).count(),
            top_files: file_results.iter().take(10).cloned().collect(), // Store 10 for consistent confidence stats
        }
    }
}

/// Aggregated file-level metrics across all queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileQualityMetrics {
    /// Model name
    pub model_name: String,
    /// Strategy used
    pub strategy: String,
    /// Number of queries evaluated
    pub query_count: usize,
    /// Number of queries skipped (R=0 after filtering)
    pub queries_skipped: usize,
    /// Mean File Precision@1
    pub mean_precision_at_1: f32,
    /// Mean File Precision@5
    pub mean_precision_at_5: f32,
    /// Mean File Precision@10
    pub mean_precision_at_10: f32,
    /// Mean File MRR
    pub mrr: f32,
    /// Mean File Recall@10
    pub mean_recall_at_10: f32,
    /// Mean File nDCG@10 - PRIMARY METRIC
    pub mean_ndcg_at_10: f32,
    /// Average top file score
    pub avg_top_score: f32,
    /// Total resolution warnings
    pub total_warnings: usize,
    /// Metrics by difficulty
    pub by_difficulty: HashMap<String, FileQualityMetricsSummary>,
    /// Metrics by query type
    pub by_query_type: HashMap<String, FileQualityMetricsSummary>,
    /// Individual query metrics
    pub query_metrics: Vec<FileQueryMetrics>,
}

/// Summary metrics for a subset of queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileQualityMetricsSummary {
    pub query_count: usize,
    pub mean_precision_at_10: f32,
    pub mrr: f32,
    pub mean_recall_at_10: f32,
    pub mean_ndcg_at_10: f32,
}

impl FileQualityMetrics {
    /// Aggregate file-level metrics from individual query results
    pub fn aggregate(
        model_name: &str,
        strategy: &str,
        query_metrics: Vec<FileQueryMetrics>,
        queries_skipped: usize,
    ) -> Self {
        let count = query_metrics.len();
        if count == 0 {
            return Self {
                model_name: model_name.to_string(),
                strategy: strategy.to_string(),
                query_count: 0,
                queries_skipped,
                mean_precision_at_1: 0.0,
                mean_precision_at_5: 0.0,
                mean_precision_at_10: 0.0,
                mrr: 0.0,
                mean_recall_at_10: 0.0,
                mean_ndcg_at_10: 0.0,
                avg_top_score: 0.0,
                total_warnings: 0,
                by_difficulty: HashMap::new(),
                by_query_type: HashMap::new(),
                query_metrics: vec![],
            };
        }

        let mean_p1: f32 = query_metrics.iter().map(|m| m.precision_at_1).sum::<f32>() / count as f32;
        let mean_p5: f32 = query_metrics.iter().map(|m| m.precision_at_5).sum::<f32>() / count as f32;
        let mean_p10: f32 = query_metrics.iter().map(|m| m.precision_at_10).sum::<f32>() / count as f32;
        let mrr: f32 = query_metrics.iter().map(|m| m.mrr).sum::<f32>() / count as f32;
        let mean_recall: f32 = query_metrics.iter().map(|m| m.recall_at_10).sum::<f32>() / count as f32;
        let mean_ndcg: f32 = query_metrics.iter().map(|m| m.ndcg_at_10).sum::<f32>() / count as f32;
        let avg_top_score: f32 = query_metrics.iter().map(|m| m.top_score).sum::<f32>() / count as f32;
        let total_warnings: usize = query_metrics.iter().map(|m| m.resolution_warnings.len()).sum();

        // Group by difficulty
        let mut by_difficulty: HashMap<String, Vec<&FileQueryMetrics>> = HashMap::new();
        for m in &query_metrics {
            by_difficulty.entry(m.difficulty.clone()).or_default().push(m);
        }
        let by_difficulty: HashMap<String, FileQualityMetricsSummary> = by_difficulty
            .into_iter()
            .map(|(k, v)| (k, Self::summarize(&v)))
            .collect();

        // Group by query type
        let mut by_query_type: HashMap<String, Vec<&FileQueryMetrics>> = HashMap::new();
        for m in &query_metrics {
            by_query_type.entry(m.query_type.clone()).or_default().push(m);
        }
        let by_query_type: HashMap<String, FileQualityMetricsSummary> = by_query_type
            .into_iter()
            .map(|(k, v)| (k, Self::summarize(&v)))
            .collect();

        Self {
            model_name: model_name.to_string(),
            strategy: strategy.to_string(),
            query_count: count,
            queries_skipped,
            mean_precision_at_1: mean_p1,
            mean_precision_at_5: mean_p5,
            mean_precision_at_10: mean_p10,
            mrr,
            mean_recall_at_10: mean_recall,
            mean_ndcg_at_10: mean_ndcg,
            avg_top_score,
            total_warnings,
            by_difficulty,
            by_query_type,
            query_metrics,
        }
    }

    fn summarize(metrics: &[&FileQueryMetrics]) -> FileQualityMetricsSummary {
        let n = metrics.len();
        if n == 0 {
            return FileQualityMetricsSummary {
                query_count: 0,
                mean_precision_at_10: 0.0,
                mrr: 0.0,
                mean_recall_at_10: 0.0,
                mean_ndcg_at_10: 0.0,
            };
        }
        FileQualityMetricsSummary {
            query_count: n,
            mean_precision_at_10: metrics.iter().map(|m| m.precision_at_10).sum::<f32>() / n as f32,
            mrr: metrics.iter().map(|m| m.mrr).sum::<f32>() / n as f32,
            mean_recall_at_10: metrics.iter().map(|m| m.recall_at_10).sum::<f32>() / n as f32,
            mean_ndcg_at_10: metrics.iter().map(|m| m.ndcg_at_10).sum::<f32>() / n as f32,
        }
    }

    /// Format as a summary string
    pub fn format_summary(&self) -> String {
        format!(
            "nDCG@10: {:.3} | MRR: {:.3} | R@10: {:.1}% | P@10: {:.1}% | Queries: {} (skipped: {})",
            self.mean_ndcg_at_10,
            self.mrr,
            self.mean_recall_at_10 * 100.0,
            self.mean_precision_at_10 * 100.0,
            self.query_count,
            self.queries_skipped
        )
    }
}

// =============================================================================
// ANSWER-LEVEL METRICS (Fine-grained: did we find the right chunk?)
// =============================================================================

/// Answer-level metrics for a single query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnswerQueryMetrics {
    /// Query text
    pub query: String,
    /// Query type (code/doc)
    pub query_type: String,
    /// Whether this query has answer ground truth
    pub has_ground_truth: bool,
    /// Did top-1 chunk contain the answer?
    pub hit_at_1: bool,
    /// Did any of top-3 chunks contain the answer?
    pub hit_at_3: bool,
    /// Did any of top-5 chunks contain the answer?
    pub hit_at_5: bool,
    /// Reciprocal rank of first chunk with answer @ 10 (0 if not found in top-10)
    pub mrr: f32,
    /// nDCG@10 for answer-level ranking
    /// Uses binary relevance: 1 if chunk contains answer, 0 otherwise
    pub ndcg_at_10: f32,
    /// Number of answer-containing chunks found in top-10
    pub relevant_in_top_10: usize,
    /// Rank of first chunk with answer (0 if not found in top-10)
    pub first_hit_rank: usize,
    /// Score of first chunk with answer
    pub first_hit_score: f32,
}

impl AnswerQueryMetrics {
    /// Create metrics when no answer ground truth is available
    pub fn no_ground_truth(query: &str, query_type: &str) -> Self {
        Self {
            query: query.to_string(),
            query_type: query_type.to_string(),
            has_ground_truth: false,
            hit_at_1: false,
            hit_at_3: false,
            hit_at_5: false,
            mrr: 0.0,
            ndcg_at_10: 0.0,
            relevant_in_top_10: 0,
            first_hit_rank: 0,
            first_hit_score: 0.0,
        }
    }

    /// Evaluate answer metrics from chunk results
    ///
    /// # Arguments
    /// * `query` - Query text
    /// * `query_type` - Query type (code/doc)
    /// * `chunk_matches` - For each chunk: (rank, score, matches_answer)
    pub fn evaluate(query: &str, query_type: &str, chunk_matches: &[(usize, f32, bool)]) -> Self {
        // Find first matching chunk
        let first_hit = chunk_matches.iter()
            .find(|(_, _, matches)| *matches);

        let (first_hit_rank, first_hit_score, mrr) = match first_hit {
            Some((rank, score, _)) => (*rank, *score, 1.0 / *rank as f32),
            None => (0, 0.0, 0.0),
        };

        // Count relevant chunks in top-10
        let relevant_in_top_10 = chunk_matches.iter()
            .take(10)
            .filter(|(_, _, m)| *m)
            .count();

        // Calculate nDCG@10 for answer-level
        // DCG@10: sum of 1/log2(rank+1) for relevant (answer-containing) chunks
        let dcg: f32 = chunk_matches.iter()
            .take(10)
            .filter(|(_, _, m)| *m)
            .map(|(rank, _, _)| 1.0 / (*rank as f32 + 1.0).log2())
            .sum();

        // IDCG@10: ideal DCG assuming all relevant chunks ranked at top
        // Use relevant_in_top_10 as total_relevant (we don't know true total without scanning all chunks)
        let total_relevant = relevant_in_top_10.max(1); // At least 1 for valid query
        let idcg: f32 = (0..total_relevant.min(10))
            .map(|i| 1.0 / (i as f32 + 2.0).log2())
            .sum();

        let ndcg_at_10 = if idcg > 0.0 { dcg / idcg } else { 0.0 };

        Self {
            query: query.to_string(),
            query_type: query_type.to_string(),
            has_ground_truth: true,
            hit_at_1: chunk_matches.iter().take(1).any(|(_, _, m)| *m),
            hit_at_3: chunk_matches.iter().take(3).any(|(_, _, m)| *m),
            hit_at_5: chunk_matches.iter().take(5).any(|(_, _, m)| *m),
            mrr,
            ndcg_at_10,
            relevant_in_top_10,
            first_hit_rank,
            first_hit_score,
        }
    }
}

/// Summary metrics for answer-level by query type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnswerMetricsSummary {
    pub query_count: usize,
    pub hit_at_1: f32,
    pub hit_at_5: f32,
    pub mrr: f32,
    pub ndcg_at_10: f32,
}

/// Aggregated answer-level metrics across all queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnswerMetrics {
    /// Number of queries with answer ground truth
    pub query_count: usize,
    /// Number of queries without answer ground truth (skipped)
    pub queries_without_ground_truth: usize,
    /// Mean Hit@1 (fraction of queries where top chunk has answer)
    pub hit_at_1: f32,
    /// Mean Hit@3
    pub hit_at_3: f32,
    /// Mean Hit@5
    pub hit_at_5: f32,
    /// Mean Reciprocal Rank @ 10 (computed from top-10 chunks only)
    pub mrr: f32,
    /// Mean nDCG@10 for answer-level ranking
    pub ndcg_at_10: f32,
    /// Metrics by query type (code/doc)
    pub by_query_type: HashMap<String, AnswerMetricsSummary>,
    /// Per-query metrics
    pub query_metrics: Vec<AnswerQueryMetrics>,
}

impl AnswerMetrics {
    /// Aggregate answer metrics from individual query results
    pub fn aggregate(query_metrics: Vec<AnswerQueryMetrics>) -> Self {
        let with_gt: Vec<_> = query_metrics.iter()
            .filter(|m| m.has_ground_truth)
            .collect();

        let count = with_gt.len();
        let without_gt = query_metrics.len() - count;

        if count == 0 {
            return Self {
                query_count: 0,
                queries_without_ground_truth: without_gt,
                hit_at_1: 0.0,
                hit_at_3: 0.0,
                hit_at_5: 0.0,
                mrr: 0.0,
                ndcg_at_10: 0.0,
                by_query_type: HashMap::new(),
                query_metrics,
            };
        }

        // Group by query type
        let mut by_query_type: HashMap<String, Vec<&AnswerQueryMetrics>> = HashMap::new();
        for m in &with_gt {
            by_query_type.entry(m.query_type.clone()).or_default().push(m);
        }
        let by_query_type: HashMap<String, AnswerMetricsSummary> = by_query_type
            .into_iter()
            .map(|(k, v)| (k, Self::summarize(&v)))
            .collect();

        Self {
            query_count: count,
            queries_without_ground_truth: without_gt,
            hit_at_1: with_gt.iter().filter(|m| m.hit_at_1).count() as f32 / count as f32,
            hit_at_3: with_gt.iter().filter(|m| m.hit_at_3).count() as f32 / count as f32,
            hit_at_5: with_gt.iter().filter(|m| m.hit_at_5).count() as f32 / count as f32,
            mrr: with_gt.iter().map(|m| m.mrr).sum::<f32>() / count as f32,
            ndcg_at_10: with_gt.iter().map(|m| m.ndcg_at_10).sum::<f32>() / count as f32,
            by_query_type,
            query_metrics,
        }
    }

    fn summarize(metrics: &[&AnswerQueryMetrics]) -> AnswerMetricsSummary {
        let n = metrics.len();
        if n == 0 {
            return AnswerMetricsSummary {
                query_count: 0,
                hit_at_1: 0.0,
                hit_at_5: 0.0,
                mrr: 0.0,
                ndcg_at_10: 0.0,
            };
        }
        AnswerMetricsSummary {
            query_count: n,
            hit_at_1: metrics.iter().filter(|m| m.hit_at_1).count() as f32 / n as f32,
            hit_at_5: metrics.iter().filter(|m| m.hit_at_5).count() as f32 / n as f32,
            mrr: metrics.iter().map(|m| m.mrr).sum::<f32>() / n as f32,
            ndcg_at_10: metrics.iter().map(|m| m.ndcg_at_10).sum::<f32>() / n as f32,
        }
    }

    /// Format as a summary string
    pub fn format_summary(&self) -> String {
        if self.query_count == 0 {
            return "No answer ground truth available".to_string();
        }
        format!(
            "Hit@1: {:.1}% | Hit@3: {:.1}% | Hit@5: {:.1}% | MRR@10: {:.3} | nDCG@10: {:.3} | Queries: {}",
            self.hit_at_1 * 100.0,
            self.hit_at_3 * 100.0,
            self.hit_at_5 * 100.0,
            self.mrr,
            self.ndcg_at_10,
            self.query_count
        )
    }
}

/// Quality metrics for a single query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMetrics {
    /// Query text
    pub query: String,
    /// Query difficulty level
    pub difficulty: String,
    /// Precision@1
    pub precision_at_1: f32,
    /// Precision@3
    pub precision_at_3: f32,
    /// Precision@5
    pub precision_at_5: f32,
    /// Precision@10
    pub precision_at_10: f32,
    /// Reciprocal rank (1/rank of first relevant result)
    pub reciprocal_rank: f32,
    /// nDCG@10 (Normalized Discounted Cumulative Gain) - BEIR/MTEB standard metric
    pub ndcg_at_10: f32,
    /// Top result score
    pub top_score: f32,
    /// Number of relevant results in top 10
    pub relevant_in_top_10: usize,
}

/// Aggregated quality metrics across all queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Model name
    pub model_name: String,
    /// Number of queries evaluated
    pub query_count: usize,
    /// Mean Precision@1
    pub mean_precision_at_1: f32,
    /// Mean Precision@3
    pub mean_precision_at_3: f32,
    /// Mean Precision@5
    pub mean_precision_at_5: f32,
    /// Mean Precision@10
    pub mean_precision_at_10: f32,
    /// Mean Reciprocal Rank
    pub mrr: f32,
    /// Mean nDCG@10 - PRIMARY METRIC (BEIR/MTEB standard)
    pub mean_ndcg_at_10: f32,
    /// Average top score
    pub avg_top_score: f32,
    /// Score standard deviation
    pub score_std_dev: f32,
    /// Metrics by difficulty level
    pub by_difficulty: Vec<(String, QualityMetrics)>,
    /// Individual query metrics
    pub query_metrics: Vec<QueryMetrics>,
}

impl QualityMetrics {
    /// Calculate precision at K
    pub fn precision_at_k(results: &[SearchResult], k: usize) -> f32 {
        let relevant_count = results
            .iter()
            .take(k)
            .filter(|r| r.relevant)
            .count();
        relevant_count as f32 / k as f32
    }

    /// Calculate reciprocal rank (1/rank of first relevant result)
    pub fn reciprocal_rank(results: &[SearchResult]) -> f32 {
        results
            .iter()
            .find(|r| r.relevant)
            .map(|r| 1.0 / r.rank as f32)
            .unwrap_or(0.0)
    }

    /// Calculate Discounted Cumulative Gain at K with binary relevance
    ///
    /// DCG@k = Σ (rel_i / log2(i+1)) for i = 1 to k
    /// For binary relevance: rel_i = 1 if relevant, 0 otherwise
    pub fn dcg_at_k(results: &[SearchResult], k: usize) -> f32 {
        results
            .iter()
            .take(k)
            .enumerate()
            .filter(|(_, r)| r.relevant)
            .map(|(i, _)| {
                // i is 0-indexed, rank is i+1
                // DCG uses log2(rank + 1) = log2(i + 2)
                1.0 / (i as f32 + 2.0).log2()
            })
            .sum()
    }

    /// Calculate Discounted Cumulative Gain at K with graded relevance
    ///
    /// DCG@k = Σ ((2^rel_i - 1) / log2(i+2)) for i = 0 to k-1
    /// This is the standard formula used by BEIR, MTEB, and CodeSearchNet.
    pub fn dcg_at_k_graded(results: &[SearchResult], k: usize) -> f32 {
        results
            .iter()
            .take(k)
            .enumerate()
            .map(|(i, r)| {
                let gain = (2.0_f32).powi(r.relevance_grade as i32) - 1.0;
                let discount = (i as f32 + 2.0).log2();
                gain / discount
            })
            .sum()
    }

    /// Calculate Ideal DCG at K (best possible ranking) with binary relevance
    ///
    /// IDCG@k = Σ (1 / log2(i+1)) for i = 1 to min(k, num_relevant)
    pub fn idcg_at_k(num_relevant: usize, k: usize) -> f32 {
        (0..num_relevant.min(k))
            .map(|i| 1.0 / (i as f32 + 2.0).log2())
            .sum()
    }

    /// Calculate Ideal DCG at K with graded relevance
    ///
    /// Takes a slice of relevance grades and computes IDCG assuming best ranking.
    pub fn idcg_at_k_graded(relevance_grades: &[u8], k: usize) -> f32 {
        let mut sorted_grades: Vec<u8> = relevance_grades.to_vec();
        sorted_grades.sort_by(|a, b| b.cmp(a)); // Sort descending

        sorted_grades
            .iter()
            .take(k)
            .enumerate()
            .map(|(i, &grade)| {
                let gain = (2.0_f32).powi(grade as i32) - 1.0;
                let discount = (i as f32 + 2.0).log2();
                gain / discount
            })
            .sum()
    }

    /// Calculate Normalized Discounted Cumulative Gain at K with binary relevance
    ///
    /// nDCG@k = DCG@k / IDCG@k
    ///
    /// This is the primary metric used by BEIR and MTEB benchmarks.
    /// It accounts for the position of ALL relevant items, not just the first.
    pub fn ndcg_at_k(results: &[SearchResult], k: usize, total_relevant: usize) -> f32 {
        let dcg = Self::dcg_at_k(results, k);
        let idcg = Self::idcg_at_k(total_relevant, k);

        if idcg == 0.0 {
            0.0 // No relevant documents exist
        } else {
            dcg / idcg
        }
    }

    /// Calculate Normalized Discounted Cumulative Gain at K with graded relevance
    ///
    /// nDCG@k = DCG@k / IDCG@k
    ///
    /// Uses exponential gain formula: (2^rel - 1) / log2(rank + 1)
    /// This is the standard for CodeSearchNet and BEIR with graded judgments.
    ///
    /// `all_relevance_grades` should contain grades for all judged documents for this query.
    pub fn ndcg_at_k_graded(results: &[SearchResult], k: usize, all_relevance_grades: &[u8]) -> f32 {
        let dcg = Self::dcg_at_k_graded(results, k);
        let idcg = Self::idcg_at_k_graded(all_relevance_grades, k);

        if idcg == 0.0 {
            0.0 // No relevant documents exist
        } else {
            dcg / idcg
        }
    }

    /// Calculate metrics for a single query (binary relevance)
    ///
    /// `total_relevant_in_corpus` is the total number of relevant documents
    /// for this query in the entire corpus (used for IDCG calculation).
    /// If unknown, pass the count of relevant in results.
    pub fn evaluate_query(
        query: &str,
        difficulty: &str,
        results: &[SearchResult],
    ) -> QueryMetrics {
        // Count relevant in results as a proxy for total relevant
        // (In a proper benchmark, we'd know the true total from ground truth)
        let relevant_count = results.iter().filter(|r| r.relevant).count();
        // Use at least 1 for nDCG calculation if there are expected relevant docs
        let total_relevant = relevant_count.max(1);

        QueryMetrics {
            query: query.to_string(),
            difficulty: difficulty.to_string(),
            precision_at_1: Self::precision_at_k(results, 1),
            precision_at_3: Self::precision_at_k(results, 3),
            precision_at_5: Self::precision_at_k(results, 5),
            precision_at_10: Self::precision_at_k(results, 10),
            reciprocal_rank: Self::reciprocal_rank(results),
            ndcg_at_10: Self::ndcg_at_k(results, 10, total_relevant),
            top_score: results.first().map(|r| r.score).unwrap_or(0.0),
            relevant_in_top_10: results.iter().take(10).filter(|r| r.relevant).count(),
        }
    }

    /// Calculate metrics for a single query with graded relevance (CodeSearchNet style)
    ///
    /// Uses graded relevance (0-3 scale) for nDCG calculation.
    /// `all_relevance_grades` should contain grades for ALL judged documents for this query.
    pub fn evaluate_query_graded(
        query: &str,
        difficulty: &str,
        results: &[SearchResult],
        all_relevance_grades: &[u8],
    ) -> QueryMetrics {
        QueryMetrics {
            query: query.to_string(),
            difficulty: difficulty.to_string(),
            precision_at_1: Self::precision_at_k(results, 1),
            precision_at_3: Self::precision_at_k(results, 3),
            precision_at_5: Self::precision_at_k(results, 5),
            precision_at_10: Self::precision_at_k(results, 10),
            reciprocal_rank: Self::reciprocal_rank(results),
            ndcg_at_10: Self::ndcg_at_k_graded(results, 10, all_relevance_grades),
            top_score: results.first().map(|r| r.score).unwrap_or(0.0),
            relevant_in_top_10: results.iter().take(10).filter(|r| r.relevant).count(),
        }
    }

    /// Aggregate metrics from multiple query results (non-recursive to avoid stack overflow)
    pub fn aggregate(model_name: &str, query_metrics: Vec<QueryMetrics>) -> Self {
        Self::aggregate_impl(model_name, query_metrics, true)
    }

    /// Internal implementation with recursion control
    fn aggregate_impl(model_name: &str, query_metrics: Vec<QueryMetrics>, compute_by_difficulty: bool) -> Self {
        let count = query_metrics.len();
        if count == 0 {
            return Self {
                model_name: model_name.to_string(),
                query_count: 0,
                mean_precision_at_1: 0.0,
                mean_precision_at_3: 0.0,
                mean_precision_at_5: 0.0,
                mean_precision_at_10: 0.0,
                mrr: 0.0,
                mean_ndcg_at_10: 0.0,
                avg_top_score: 0.0,
                score_std_dev: 0.0,
                by_difficulty: vec![],
                query_metrics: vec![],
            };
        }

        let mean_p1: f32 = query_metrics.iter().map(|m| m.precision_at_1).sum::<f32>() / count as f32;
        let mean_p3: f32 = query_metrics.iter().map(|m| m.precision_at_3).sum::<f32>() / count as f32;
        let mean_p5: f32 = query_metrics.iter().map(|m| m.precision_at_5).sum::<f32>() / count as f32;
        let mean_p10: f32 = query_metrics.iter().map(|m| m.precision_at_10).sum::<f32>() / count as f32;
        let mrr: f32 = query_metrics.iter().map(|m| m.reciprocal_rank).sum::<f32>() / count as f32;
        let mean_ndcg: f32 = query_metrics.iter().map(|m| m.ndcg_at_10).sum::<f32>() / count as f32;

        let scores: Vec<f32> = query_metrics.iter().map(|m| m.top_score).collect();
        let avg_top_score = scores.iter().sum::<f32>() / count as f32;

        // Standard deviation of scores
        let variance = scores
            .iter()
            .map(|s| (s - avg_top_score).powi(2))
            .sum::<f32>()
            / count as f32;
        let score_std_dev = variance.sqrt();

        // Group by difficulty (only at top level to prevent recursion)
        let by_difficulty: Vec<(String, QualityMetrics)> = if compute_by_difficulty {
            let difficulties: HashSet<&str> = query_metrics.iter().map(|m| m.difficulty.as_str()).collect();
            difficulties
                .into_iter()
                .map(|d| {
                    let filtered: Vec<QueryMetrics> = query_metrics
                        .iter()
                        .filter(|m| m.difficulty == d)
                        .cloned()
                        .collect();
                    // Pass false to prevent recursive difficulty grouping
                    (d.to_string(), Self::aggregate_impl(&format!("{}-{}", model_name, d), filtered, false))
                })
                .collect()
        } else {
            vec![]
        };

        Self {
            model_name: model_name.to_string(),
            query_count: count,
            mean_precision_at_1: mean_p1,
            mean_precision_at_3: mean_p3,
            mean_precision_at_5: mean_p5,
            mean_precision_at_10: mean_p10,
            mrr,
            mean_ndcg_at_10: mean_ndcg,
            avg_top_score,
            score_std_dev,
            by_difficulty,
            query_metrics,
        }
    }

    /// Format as a summary string (nDCG@10 is PRIMARY metric per BEIR/MTEB)
    pub fn format_summary(&self) -> String {
        format!(
            "nDCG@10: {:.3} | MRR: {:.3} | P@1: {:.1}% | P@10: {:.1}% | Avg Score: {:.3}",
            self.mean_ndcg_at_10,
            self.mrr,
            self.mean_precision_at_1 * 100.0,
            self.mean_precision_at_10 * 100.0,
            self.avg_top_score
        )
    }
}

/// Calculate cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

// =============================================================================
// CORPUS STATISTICS
// =============================================================================

/// Basic statistics about the corpus being evaluated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusStats {
    pub name: String,
    pub file_count: usize,
    pub chunk_count: usize,
    pub code_chunk_count: usize,
    pub doc_chunk_count: usize,
    pub total_lines: usize,
}

// =============================================================================
// CONFIDENCE METRICS
// =============================================================================

/// Bootstrap confidence interval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub mean: f32,
    pub lower: f32,
    pub upper: f32,
    pub confidence_level: f32,
}

/// Per-query confidence metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryConfidenceMetrics {
    pub query: String,
    pub top_1_score: f32,
    pub top_2_score: f32,
    pub score_gap: f32,
    pub top_10_mean: f32,
    pub top_10_std: f32,
    pub top_10_range: f32,
    pub cv: f32,
    pub top_1_relevant: bool,
}

/// Model-level confidence metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfidenceMetrics {
    pub model_name: String,
    pub query_count: usize,
    pub mean_score_gap: f32,
    pub std_score_gap: f32,
    pub mean_within_query_std: f32,
    pub mean_top_1_score: f32,
    pub std_top_1_score: f32,
    pub score_relevance_correlation: f32,
    pub ndcg_ci: ConfidenceInterval,
    pub ci_width_percent: f32,
    pub difficulty_consistency: f32,
    pub query_confidence: Vec<QueryConfidenceMetrics>,
}

impl ModelConfidenceMetrics {
    /// Format as a summary string
    pub fn format_summary(&self) -> String {
        format!(
            "Score Gap: {:.3}±{:.3} | Score-Rel Corr: {:.3} | CI Width: {:.1}% | Difficulty Consistency: {:.2}",
            self.mean_score_gap,
            self.std_score_gap,
            self.score_relevance_correlation,
            self.ci_width_percent,
            self.difficulty_consistency
        )
    }

    /// Interpret overall reliability
    pub fn reliability_interpretation(&self) -> &'static str {
        // High reliability: good score gaps, narrow CI, consistent difficulty ordering
        let good_gap = self.mean_score_gap > 0.05;
        let narrow_ci = self.ci_width_percent < 20.0;
        let consistent = self.difficulty_consistency > 0.6;

        match (good_gap, narrow_ci, consistent) {
            (true, true, true) => "High Reliability",
            (true, true, false) | (true, false, true) | (false, true, true) => "Medium Reliability",
            _ => "Low Reliability",
        }
    }
}

/// Bootstrap confidence interval calculation
pub fn bootstrap_confidence_interval(values: &[f32], confidence_level: f32, n_bootstrap: usize) -> ConfidenceInterval {
    if values.is_empty() {
        return ConfidenceInterval {
            mean: 0.0,
            lower: 0.0,
            upper: 0.0,
            confidence_level,
        };
    }

    let n = values.len();
    let mean = values.iter().sum::<f32>() / n as f32;

    // Simple bootstrap using deterministic sampling for reproducibility
    let mut bootstrap_means: Vec<f32> = Vec::with_capacity(n_bootstrap);
    for i in 0..n_bootstrap {
        let mut sum = 0.0;
        for j in 0..n {
            // Use simple hash for reproducible "random" sampling
            let idx = ((i * 31 + j * 17) % n) as usize;
            sum += values[idx];
        }
        bootstrap_means.push(sum / n as f32);
    }

    bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - confidence_level;
    let lower_idx = ((alpha / 2.0) * n_bootstrap as f32) as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f32) as usize;

    ConfidenceInterval {
        mean,
        lower: bootstrap_means.get(lower_idx).copied().unwrap_or(mean),
        upper: bootstrap_means.get(upper_idx.min(n_bootstrap - 1)).copied().unwrap_or(mean),
        confidence_level,
    }
}

/// Calculate standard deviation
pub fn std_dev(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    variance.sqrt()
}

/// Calculate Pearson correlation coefficient
pub fn pearson_correlation(x: &[f32], y: &[f32]) -> f32 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len() as f32;
    let mean_x = x.iter().sum::<f32>() / n;
    let mean_y = y.iter().sum::<f32>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x == 0.0 || var_y == 0.0 {
        return 0.0;
    }

    cov / (var_x.sqrt() * var_y.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create binary relevance results
    fn binary_result(id: &str, score: f32, rank: usize, relevant: bool) -> SearchResult {
        SearchResult::binary(id.to_string(), format!("{}.rs", id), score, rank, relevant)
    }

    // Helper to create graded relevance results
    fn graded_result(id: &str, score: f32, rank: usize, grade: u8) -> SearchResult {
        SearchResult::graded(id.to_string(), format!("{}.rs", id), score, rank, grade)
    }

    #[test]
    fn test_precision_at_k() {
        let results = vec![
            binary_result("1", 0.9, 1, true),
            binary_result("2", 0.8, 2, false),
            binary_result("3", 0.7, 3, true),
            binary_result("4", 0.6, 4, false),
            binary_result("5", 0.5, 5, false),
        ];

        assert_eq!(QualityMetrics::precision_at_k(&results, 1), 1.0);
        assert_eq!(QualityMetrics::precision_at_k(&results, 3), 2.0 / 3.0);
        assert_eq!(QualityMetrics::precision_at_k(&results, 5), 2.0 / 5.0);
    }

    #[test]
    fn test_reciprocal_rank() {
        let results = vec![
            binary_result("1", 0.9, 1, false),
            binary_result("2", 0.8, 2, false),
            binary_result("3", 0.7, 3, true),
        ];

        assert_eq!(QualityMetrics::reciprocal_rank(&results), 1.0 / 3.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.0001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c)).abs() < 0.0001);
    }

    #[test]
    fn test_dcg_at_k() {
        // Perfect ranking: relevant at rank 1
        let results = vec![
            binary_result("1", 0.9, 1, true),
            binary_result("2", 0.8, 2, false),
        ];
        // DCG@2 = 1/log2(2) + 0 = 1/1 = 1.0
        let dcg = QualityMetrics::dcg_at_k(&results, 2);
        assert!((dcg - 1.0).abs() < 0.0001);

        // Relevant at rank 2
        let results2 = vec![
            binary_result("1", 0.9, 1, false),
            binary_result("2", 0.8, 2, true),
        ];
        // DCG@2 = 0 + 1/log2(3) ≈ 0.631
        let dcg2 = QualityMetrics::dcg_at_k(&results2, 2);
        assert!((dcg2 - 0.631).abs() < 0.01);
    }

    #[test]
    fn test_dcg_at_k_graded() {
        // Grade 3 at rank 1, grade 1 at rank 2
        let results = vec![
            graded_result("1", 0.9, 1, 3),  // gain = 2^3 - 1 = 7
            graded_result("2", 0.8, 2, 1),  // gain = 2^1 - 1 = 1
        ];
        // DCG@2 = 7/log2(2) + 1/log2(3) = 7/1 + 0.631 ≈ 7.631
        let dcg = QualityMetrics::dcg_at_k_graded(&results, 2);
        assert!((dcg - 7.631).abs() < 0.01);
    }

    #[test]
    fn test_idcg_at_k_graded() {
        // Two judgments: grade 3 and grade 1
        // Best order: [3, 1]
        let grades = vec![1, 3];  // Will be sorted to [3, 1]
        // IDCG@2 = (2^3-1)/log2(2) + (2^1-1)/log2(3) = 7/1 + 1/1.585 ≈ 7.631
        let idcg = QualityMetrics::idcg_at_k_graded(&grades, 2);
        assert!((idcg - 7.631).abs() < 0.01);
    }

    #[test]
    fn test_ndcg_at_k_graded() {
        // Perfect ranking: grade 3 at rank 1, grade 1 at rank 2
        let results = vec![
            graded_result("1", 0.9, 1, 3),
            graded_result("2", 0.8, 2, 1),
        ];
        let all_grades = vec![3, 1];
        // Perfect ranking -> nDCG = 1.0
        let ndcg = QualityMetrics::ndcg_at_k_graded(&results, 10, &all_grades);
        assert!((ndcg - 1.0).abs() < 0.0001);

        // Imperfect ranking: grade 1 at rank 1, grade 3 at rank 2
        let results2 = vec![
            graded_result("1", 0.9, 1, 1),
            graded_result("2", 0.8, 2, 3),
        ];
        // DCG@2 = 1/log2(2) + 7/log2(3) = 1 + 4.416 ≈ 5.416
        // IDCG@2 = 7/log2(2) + 1/log2(3) = 7 + 0.631 ≈ 7.631
        // nDCG ≈ 5.416/7.631 ≈ 0.710
        let ndcg2 = QualityMetrics::ndcg_at_k_graded(&results2, 10, &all_grades);
        assert!((ndcg2 - 0.710).abs() < 0.01);
    }

    #[test]
    fn test_ndcg_at_k() {
        // Perfect ranking: only relevant at rank 1, 1 total relevant
        let results = vec![
            binary_result("1", 0.9, 1, true),
            binary_result("2", 0.8, 2, false),
        ];
        // nDCG = DCG/IDCG = 1.0/1.0 = 1.0
        let ndcg = QualityMetrics::ndcg_at_k(&results, 10, 1);
        assert!((ndcg - 1.0).abs() < 0.0001);

        // Imperfect ranking: relevant at rank 2, 1 total relevant
        let results2 = vec![
            binary_result("1", 0.9, 1, false),
            binary_result("2", 0.8, 2, true),
        ];
        // IDCG = 1/log2(2) = 1.0
        // DCG = 1/log2(3) ≈ 0.631
        // nDCG ≈ 0.631
        let ndcg2 = QualityMetrics::ndcg_at_k(&results2, 10, 1);
        assert!((ndcg2 - 0.631).abs() < 0.01);

        // Two relevant items, both in top 2
        let results3 = vec![
            binary_result("1", 0.9, 1, true),
            binary_result("2", 0.8, 2, true),
        ];
        // Perfect ranking with 2 relevant -> nDCG = 1.0
        let ndcg3 = QualityMetrics::ndcg_at_k(&results3, 10, 2);
        assert!((ndcg3 - 1.0).abs() < 0.0001);

        // No relevant results
        let results4 = vec![
            binary_result("1", 0.9, 1, false),
        ];
        let ndcg4 = QualityMetrics::ndcg_at_k(&results4, 10, 1);
        assert_eq!(ndcg4, 0.0);
    }

    // =========================================================================
    // FILE-LEVEL METRICS TESTS
    // =========================================================================

    fn file_result(path: &str, score: f32, rank: usize, relevant: bool) -> FileSearchResult {
        FileSearchResult::new(
            path.to_string(),
            score,
            rank,
            relevant,
            1,
            format!("{}-chunk", path),
            "preview content".to_string(),
        )
    }

    #[test]
    fn test_file_precision_at_k() {
        let results = vec![
            file_result("src/main.rs", 0.9, 1, true),
            file_result("src/lib.rs", 0.8, 2, false),
            file_result("src/utils.rs", 0.7, 3, true),
            file_result("src/config.rs", 0.6, 4, false),
        ];

        assert_eq!(file_precision_at_k(&results, 1), 1.0);
        assert!((file_precision_at_k(&results, 3) - 2.0 / 3.0).abs() < 0.001);
        assert_eq!(file_precision_at_k(&results, 4), 0.5);
    }

    #[test]
    fn test_file_mrr() {
        // First result is relevant
        let results1 = vec![
            file_result("src/main.rs", 0.9, 1, true),
            file_result("src/lib.rs", 0.8, 2, false),
        ];
        assert_eq!(file_mrr(&results1), 1.0);

        // Second result is relevant
        let results2 = vec![
            file_result("src/main.rs", 0.9, 1, false),
            file_result("src/lib.rs", 0.8, 2, true),
        ];
        assert_eq!(file_mrr(&results2), 0.5);

        // No relevant results
        let results3 = vec![
            file_result("src/main.rs", 0.9, 1, false),
        ];
        assert_eq!(file_mrr(&results3), 0.0);
    }

    #[test]
    fn test_file_recall_at_k() {
        let results = vec![
            file_result("src/main.rs", 0.9, 1, true),
            file_result("src/lib.rs", 0.8, 2, false),
            file_result("src/utils.rs", 0.7, 3, true),
        ];

        // 2 relevant found out of 3 total relevant
        assert!((file_recall_at_k(&results, 3, 3) - 2.0 / 3.0).abs() < 0.001);
        // 1 relevant found in top 1 out of 3 total
        assert!((file_recall_at_k(&results, 1, 3) - 1.0 / 3.0).abs() < 0.001);
        // 0 total relevant - edge case
        assert_eq!(file_recall_at_k(&results, 3, 0), 0.0);
    }

    #[test]
    fn test_file_ndcg_at_k() {
        // Perfect ranking: relevant at rank 1
        let results = vec![
            file_result("src/main.rs", 0.9, 1, true),
            file_result("src/lib.rs", 0.8, 2, false),
        ];
        assert!((file_ndcg_at_k(&results, 10, 1) - 1.0).abs() < 0.001);

        // Imperfect ranking: relevant at rank 2
        let results2 = vec![
            file_result("src/main.rs", 0.9, 1, false),
            file_result("src/lib.rs", 0.8, 2, true),
        ];
        // IDCG = 1/log2(2) = 1.0
        // DCG = 1/log2(3) ≈ 0.631
        assert!((file_ndcg_at_k(&results2, 10, 1) - 0.631).abs() < 0.01);

        // Two relevant, both found at positions 1 and 2
        let results3 = vec![
            file_result("src/main.rs", 0.9, 1, true),
            file_result("src/lib.rs", 0.8, 2, true),
        ];
        assert!((file_ndcg_at_k(&results3, 10, 2) - 1.0).abs() < 0.001);

        // CRITICAL: Test IDCG uses total_relevant, not found count
        // Two relevant expected, but only 1 found at rank 2
        let results4 = vec![
            file_result("src/main.rs", 0.9, 1, false),
            file_result("src/lib.rs", 0.8, 2, true),
            file_result("src/utils.rs", 0.7, 3, false),
        ];
        // IDCG@10 for 2 relevant = 1/log2(2) + 1/log2(3) = 1.0 + 0.631 = 1.631
        // DCG@10 = 1/log2(3) = 0.631 (only one found at rank 2)
        // nDCG = 0.631 / 1.631 ≈ 0.387
        let ndcg = file_ndcg_at_k(&results4, 10, 2);
        assert!((ndcg - 0.387).abs() < 0.01, "nDCG was {}, expected ~0.387", ndcg);

        // R=0 edge case
        assert_eq!(file_ndcg_at_k(&results, 10, 0), 0.0);
    }

    #[test]
    fn test_aggregate_chunks_to_files() {
        let chunks = vec![
            ("chunk1".to_string(), "src/main.rs".to_string(), 0.9, "content1".to_string()),
            ("chunk2".to_string(), "src/main.rs".to_string(), 0.8, "content2".to_string()),
            ("chunk3".to_string(), "src/lib.rs".to_string(), 0.7, "content3".to_string()),
            ("chunk4".to_string(), "src/utils.rs".to_string(), 0.85, "content4".to_string()),
        ];

        let expected: HashSet<String> = ["src/main.rs".to_string()].into_iter().collect();
        let results = aggregate_chunks_to_files(&chunks, &expected);

        // Should have 3 files
        assert_eq!(results.len(), 3);

        // First should be main.rs with max score 0.9
        assert_eq!(results[0].file_path, "src/main.rs");
        assert_eq!(results[0].score, 0.9);
        assert_eq!(results[0].rank, 1);
        assert!(results[0].relevant);
        assert_eq!(results[0].chunk_count, 2); // Two chunks from main.rs

        // Second should be utils.rs with score 0.85
        assert_eq!(results[1].file_path, "src/utils.rs");
        assert_eq!(results[1].score, 0.85);
        assert_eq!(results[1].rank, 2);
        assert!(!results[1].relevant);

        // Third should be lib.rs
        assert_eq!(results[2].file_path, "src/lib.rs");
        assert_eq!(results[2].rank, 3);
    }

    #[test]
    fn test_file_index_resolution() {
        let files = vec![
            "src/main.rs".to_string(),
            "src/http/routes.rs".to_string(),
            "tests/main.rs".to_string(),
        ];
        let index = FileIndex::new(files.into_iter());

        // Exact match
        assert_eq!(index.resolve("src/main.rs").unwrap(), "src/main.rs");

        // Suffix match (unambiguous)
        assert_eq!(index.resolve("http/routes.rs").unwrap(), "src/http/routes.rs");

        // Ambiguous basename
        let result = index.resolve("main.rs");
        assert!(matches!(result, Err(FileResolution::Ambiguous(_))));

        // Not found
        let result = index.resolve("nonexistent.rs");
        assert!(matches!(result, Err(FileResolution::NotFound)));

        // CRITICAL: Basename matches but suffix doesn't - should NOT match
        // This prevents false positives like "auth/routes.rs" matching "src/http/routes.rs"
        let result = index.resolve("auth/routes.rs");
        assert!(matches!(result, Err(FileResolution::NotFound)),
            "Should NOT match 'src/http/routes.rs' when expected is 'auth/routes.rs'");
    }

    #[test]
    fn test_file_index_resolve_expected() {
        let files = vec![
            "src/main.rs".to_string(),
            "src/http/routes.rs".to_string(),
            "tests/main.rs".to_string(),
        ];
        let index = FileIndex::new(files.into_iter());

        let expected = vec![
            "src/main.rs".to_string(),     // Exact match
            "http/routes.rs".to_string(),  // Suffix match
            "main.rs".to_string(),         // Ambiguous - should be skipped
            "nonexistent.rs".to_string(),  // Not found - should warn
        ];

        let (resolved, warnings) = index.resolve_expected_files(&expected);

        // Should resolve 2 files
        assert_eq!(resolved.len(), 2);
        assert!(resolved.contains("src/main.rs"));
        assert!(resolved.contains("src/http/routes.rs"));

        // Should have 2 warnings
        assert_eq!(warnings.len(), 2);
    }

    #[test]
    fn test_normalize_path() {
        assert_eq!(normalize_path("src\\main.rs"), "src/main.rs");
        assert_eq!(normalize_path("src/main.rs"), "src/main.rs");
    }

    // =========================================================================
    // ANSWER-LEVEL METRICS TESTS
    // =========================================================================

    #[test]
    fn test_answer_query_metrics_no_ground_truth() {
        let metrics = AnswerQueryMetrics::no_ground_truth("test query", "code");
        assert_eq!(metrics.query, "test query");
        assert_eq!(metrics.query_type, "code");
        assert!(!metrics.has_ground_truth);
        assert!(!metrics.hit_at_1);
        assert!(!metrics.hit_at_3);
        assert!(!metrics.hit_at_5);
        assert_eq!(metrics.mrr, 0.0);
        assert_eq!(metrics.ndcg_at_10, 0.0);
        assert_eq!(metrics.relevant_in_top_10, 0);
        assert_eq!(metrics.first_hit_rank, 0);
    }

    #[test]
    fn test_answer_query_metrics_hit_at_1() {
        // First chunk matches
        let chunk_matches = vec![
            (1, 0.9, true),  // rank 1, score 0.9, matches
            (2, 0.8, false),
            (3, 0.7, false),
        ];
        let metrics = AnswerQueryMetrics::evaluate("test", "code", &chunk_matches);
        assert!(metrics.has_ground_truth);
        assert_eq!(metrics.query_type, "code");
        assert!(metrics.hit_at_1);
        assert!(metrics.hit_at_3);
        assert!(metrics.hit_at_5);
        assert_eq!(metrics.mrr, 1.0);
        assert!((metrics.ndcg_at_10 - 1.0).abs() < 0.001); // Perfect ranking
        assert_eq!(metrics.relevant_in_top_10, 1);
        assert_eq!(metrics.first_hit_rank, 1);
        assert_eq!(metrics.first_hit_score, 0.9);
    }

    #[test]
    fn test_answer_query_metrics_hit_at_3() {
        // Third chunk matches
        let chunk_matches = vec![
            (1, 0.9, false),
            (2, 0.8, false),
            (3, 0.7, true),  // rank 3 matches
            (4, 0.6, false),
        ];
        let metrics = AnswerQueryMetrics::evaluate("test", "code", &chunk_matches);
        assert!(!metrics.hit_at_1);
        assert!(metrics.hit_at_3);
        assert!(metrics.hit_at_5);
        assert!((metrics.mrr - 1.0 / 3.0).abs() < 0.001);
        // nDCG@10: DCG = 1/log2(4) ≈ 0.5, IDCG = 1/log2(2) = 1.0
        assert!((metrics.ndcg_at_10 - 0.5).abs() < 0.01);
        assert_eq!(metrics.relevant_in_top_10, 1);
        assert_eq!(metrics.first_hit_rank, 3);
    }

    #[test]
    fn test_answer_query_metrics_hit_at_5() {
        // Fifth chunk matches
        let chunk_matches = vec![
            (1, 0.9, false),
            (2, 0.8, false),
            (3, 0.7, false),
            (4, 0.6, false),
            (5, 0.5, true),  // rank 5 matches
        ];
        let metrics = AnswerQueryMetrics::evaluate("test", "code", &chunk_matches);
        assert!(!metrics.hit_at_1);
        assert!(!metrics.hit_at_3);
        assert!(metrics.hit_at_5);
        assert!((metrics.mrr - 0.2).abs() < 0.001);
        // nDCG@10: DCG = 1/log2(6) ≈ 0.387, IDCG = 1/log2(2) = 1.0
        assert!((metrics.ndcg_at_10 - 0.387).abs() < 0.01);
        assert_eq!(metrics.relevant_in_top_10, 1);
        assert_eq!(metrics.first_hit_rank, 5);
    }

    #[test]
    fn test_answer_query_metrics_no_hit_in_top_10() {
        // No matches in results (simulating no hit in top-10)
        let chunk_matches = vec![
            (1, 0.9, false),
            (2, 0.8, false),
            (3, 0.7, false),
        ];
        let metrics = AnswerQueryMetrics::evaluate("test", "code", &chunk_matches);
        assert!(metrics.has_ground_truth);
        assert!(!metrics.hit_at_1);
        assert!(!metrics.hit_at_3);
        assert!(!metrics.hit_at_5);
        assert_eq!(metrics.mrr, 0.0);
        assert_eq!(metrics.ndcg_at_10, 0.0);
        assert_eq!(metrics.relevant_in_top_10, 0);
        assert_eq!(metrics.first_hit_rank, 0);
    }

    #[test]
    fn test_answer_query_metrics_multiple_hits() {
        // Multiple answer chunks found
        let chunk_matches = vec![
            (1, 0.9, true),   // rank 1 matches
            (2, 0.8, false),
            (3, 0.7, true),   // rank 3 also matches
            (4, 0.6, false),
        ];
        let metrics = AnswerQueryMetrics::evaluate("test", "code", &chunk_matches);
        assert!(metrics.hit_at_1);
        assert_eq!(metrics.mrr, 1.0);
        assert_eq!(metrics.relevant_in_top_10, 2);
        // nDCG@10 with 2 relevant: DCG = 1/log2(2) + 1/log2(4), IDCG = 1/log2(2) + 1/log2(3)
        // Perfect ranking would have both at top positions
        assert!((metrics.ndcg_at_10 - 0.924).abs() < 0.01);
        assert_eq!(metrics.first_hit_rank, 1);
    }

    #[test]
    fn test_answer_metrics_aggregate() {
        let query_metrics = vec![
            AnswerQueryMetrics {
                query: "q1".to_string(),
                query_type: "code".to_string(),
                has_ground_truth: true,
                hit_at_1: true,
                hit_at_3: true,
                hit_at_5: true,
                mrr: 1.0,
                ndcg_at_10: 1.0,
                relevant_in_top_10: 1,
                first_hit_rank: 1,
                first_hit_score: 0.9,
            },
            AnswerQueryMetrics {
                query: "q2".to_string(),
                query_type: "code".to_string(),
                has_ground_truth: true,
                hit_at_1: false,
                hit_at_3: true,
                hit_at_5: true,
                mrr: 0.5, // 1/2
                ndcg_at_10: 0.631, // approx for rank 2
                relevant_in_top_10: 1,
                first_hit_rank: 2,
                first_hit_score: 0.8,
            },
            AnswerQueryMetrics::no_ground_truth("q3", "code"), // Should be excluded
        ];

        let metrics = AnswerMetrics::aggregate(query_metrics);
        assert_eq!(metrics.query_count, 2);
        assert_eq!(metrics.queries_without_ground_truth, 1);
        assert!((metrics.hit_at_1 - 0.5).abs() < 0.001); // 1/2
        assert!((metrics.hit_at_3 - 1.0).abs() < 0.001); // 2/2
        assert!((metrics.hit_at_5 - 1.0).abs() < 0.001); // 2/2
        assert!((metrics.mrr - 0.75).abs() < 0.001);     // (1.0 + 0.5) / 2 = 0.75
        assert!((metrics.ndcg_at_10 - 0.8155).abs() < 0.01); // (1.0 + 0.631) / 2
        // Check by_query_type
        assert!(metrics.by_query_type.contains_key("code"));
        assert_eq!(metrics.by_query_type["code"].query_count, 2);
    }

    #[test]
    fn test_answer_metrics_aggregate_empty() {
        let query_metrics = vec![
            AnswerQueryMetrics::no_ground_truth("q1", "code"),
            AnswerQueryMetrics::no_ground_truth("q2", "doc"),
        ];

        let metrics = AnswerMetrics::aggregate(query_metrics);
        assert_eq!(metrics.query_count, 0);
        assert_eq!(metrics.queries_without_ground_truth, 2);
        assert_eq!(metrics.hit_at_1, 0.0);
        assert_eq!(metrics.mrr, 0.0);
        assert_eq!(metrics.ndcg_at_10, 0.0);
        assert!(metrics.by_query_type.is_empty());
    }

    #[test]
    fn test_answer_metrics_by_query_type() {
        let query_metrics = vec![
            AnswerQueryMetrics {
                query: "code_q1".to_string(),
                query_type: "code".to_string(),
                has_ground_truth: true,
                hit_at_1: true,
                hit_at_3: true,
                hit_at_5: true,
                mrr: 1.0,
                ndcg_at_10: 1.0,
                relevant_in_top_10: 1,
                first_hit_rank: 1,
                first_hit_score: 0.9,
            },
            AnswerQueryMetrics {
                query: "doc_q1".to_string(),
                query_type: "doc".to_string(),
                has_ground_truth: true,
                hit_at_1: false,
                hit_at_3: true,
                hit_at_5: true,
                mrr: 0.5,
                ndcg_at_10: 0.631,
                relevant_in_top_10: 1,
                first_hit_rank: 2,
                first_hit_score: 0.8,
            },
        ];

        let metrics = AnswerMetrics::aggregate(query_metrics);
        assert_eq!(metrics.query_count, 2);

        // Check code metrics
        assert!(metrics.by_query_type.contains_key("code"));
        let code_metrics = &metrics.by_query_type["code"];
        assert_eq!(code_metrics.query_count, 1);
        assert!((code_metrics.hit_at_1 - 1.0).abs() < 0.001);
        assert!((code_metrics.ndcg_at_10 - 1.0).abs() < 0.001);

        // Check doc metrics
        assert!(metrics.by_query_type.contains_key("doc"));
        let doc_metrics = &metrics.by_query_type["doc"];
        assert_eq!(doc_metrics.query_count, 1);
        assert!((doc_metrics.hit_at_1 - 0.0).abs() < 0.001);
        assert!((doc_metrics.ndcg_at_10 - 0.631).abs() < 0.01);
    }
}
