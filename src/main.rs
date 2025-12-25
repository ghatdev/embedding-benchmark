//! Embedding Benchmark CLI
//!
//! A generic, configurable benchmark for embedding models on any codebase.
//!
//! ## Quick Start
//!
//! ```bash
//! # Run benchmark with line-based chunking (baseline)
//! ./embedding-benchmark run \
//!     --corpus ./my-codebase \
//!     --queries ./queries.json \
//!     --strategy line
//!
//! # Run benchmark with semantic chunking
//! ./embedding-benchmark run \
//!     --corpus ./my-codebase \
//!     --queries ./queries.json \
//!     --strategy semantic
//!
//! # Use custom models config
//! ./embedding-benchmark run \
//!     --corpus ./my-codebase \
//!     --queries ./queries.json \
//!     --strategy semantic \
//!     --models-config ./my-models.toml
//! ```
//!
//! ## Models Configuration
//!
//! Models are configured in `models.toml`. All models in the config are
//! tested in a single run with the same chunks.
//!
//! See `models.toml` for example configuration.

mod benchmark;
mod config;
mod corpus;
mod embedders;
mod queries;
mod resource_monitor;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

use benchmark::{
    CorpusStats, ModelConfidenceMetrics, QueryConfidenceMetrics,
    ConfidenceInterval, bootstrap_confidence_interval, std_dev, pearson_correlation,
    cosine_similarity, aggregate_scored_chunks, normalize_path,
    FileIndex, FileQueryMetrics, FileQualityMetrics,
    AnswerQueryMetrics, AnswerMetrics,
};
use benchmark::quality::FileResolutionWarning;
use config::{ModelsConfig, Strategy};
use corpus::{load_corpus_generic, CorpusConfig, ContentType};
use embedders::{EmbedderBackend, FastEmbedBackend, FastEmbedModel, MistralRsBackend, MistralRsDevice, MistralRsModel, MistralRsQuantization, ResourceMetrics};
use queries::{QueryFile, BenchmarkQuery, QueryType};
use resource_monitor::ResourceMonitor;

/// Chunking strategy for CLI
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum StrategyArg {
    /// Line-based chunking (baseline)
    #[default]
    Line,
    /// Semantic structure-aware chunking
    Semantic,
}

impl From<StrategyArg> for Strategy {
    fn from(arg: StrategyArg) -> Self {
        match arg {
            StrategyArg::Line => Strategy::Line,
            StrategyArg::Semantic => Strategy::Semantic,
        }
    }
}

#[derive(Parser)]
#[command(name = "embedding-benchmark")]
#[command(about = "Generic embedding model benchmark for any codebase")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run benchmark on a corpus with a query file
    ///
    /// All models in the config file are tested with the same chunks.
    /// Results are saved to a JSON file.
    Run {
        /// Path to the corpus (codebase directory)
        #[arg(short, long)]
        corpus: PathBuf,

        /// Path to the query JSON file
        #[arg(short, long)]
        queries: PathBuf,

        /// Output file for results (JSON)
        #[arg(short, long, default_value = "results/benchmark_results.json")]
        output: PathBuf,

        /// Chunking strategy: line (baseline) or semantic (structure-aware)
        #[arg(short, long, value_enum, default_value = "line")]
        strategy: StrategyArg,

        /// Path to models config file (TOML)
        #[arg(short, long, default_value = "models.toml")]
        models_config: PathBuf,

        /// Name for this corpus (used in reports)
        #[arg(short, long)]
        name: Option<String>,

        /// File extensions to include (comma-separated, e.g., "rs,py,md")
        /// If not specified, auto-detects based on common code extensions
        #[arg(short, long, value_delimiter = ',')]
        extensions: Option<Vec<String>>,

        /// Maximum chunks to embed (0 = unlimited)
        #[arg(long, default_value = "0")]
        max_chunks: usize,

        /// Maximum characters per chunk
        #[arg(long, default_value = "2000")]
        max_chars: usize,

        /// Context budget for semantic chunking (chars reserved for context)
        #[arg(long, default_value = "400")]
        context_budget: usize,

        /// Lines per code chunk (for line-based strategy)
        #[arg(long, default_value = "40")]
        code_lines: usize,

        /// Lines per doc chunk (for line-based strategy)
        #[arg(long, default_value = "60")]
        doc_lines: usize,

        /// Chunk overlap ratio (0.0 to 0.5, for line-based strategy)
        #[arg(long, default_value = "0.25")]
        overlap: f32,

        /// Additional skip patterns (comma-separated path substrings)
        #[arg(long, value_delimiter = ',')]
        skip: Option<Vec<String>>,
    },

    /// Validate a query file
    ValidateQueries {
        /// Path to the query JSON file
        #[arg(short, long)]
        queries: PathBuf,
    },

    /// List available embedding models
    List,

    /// Test a single model with sample texts
    Test {
        /// Model to test (e.g., "bge-small", "embedding-gemma")
        model: String,

        /// Backend: fastembed or mistralrs
        #[arg(short, long, default_value = "fastembed")]
        backend: String,

        /// Quantization (for mistralrs): none, q8_0, q4k
        #[arg(short, long, default_value = "none")]
        quantization: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            corpus,
            queries,
            output,
            strategy,
            models_config,
            name,
            extensions,
            max_chunks,
            max_chars,
            context_budget,
            code_lines,
            doc_lines,
            overlap,
            skip,
        } => {
            run_benchmark(
                &corpus,
                &queries,
                &output,
                strategy.into(),
                &models_config,
                name,
                extensions,
                max_chunks,
                max_chars,
                context_budget,
                code_lines,
                doc_lines,
                overlap,
                skip,
            )
            .await?;
        }

        Commands::ValidateQueries { queries } => {
            validate_queries(&queries)?;
        }

        Commands::List => {
            list_models();
        }

        Commands::Test {
            model,
            backend,
            quantization,
        } => {
            test_single_model(&model, &backend, &quantization).await?;
        }
    }

    Ok(())
}

/// Run the benchmark with all models from config
async fn run_benchmark(
    corpus_path: &PathBuf,
    queries_path: &PathBuf,
    output: &PathBuf,
    strategy: Strategy,
    models_config_path: &PathBuf,
    name: Option<String>,
    extensions: Option<Vec<String>>,
    max_chunks: usize,
    max_chars: usize,
    context_budget: usize,
    code_lines: usize,
    doc_lines: usize,
    overlap: f32,
    skip_patterns: Option<Vec<String>>,
) -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              EMBEDDING BENCHMARK                             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load models config
    let models_config = if models_config_path.exists() {
        println!("Loading models from {:?}...", models_config_path);
        ModelsConfig::load(models_config_path)?
    } else {
        println!("Using default models config...");
        ModelsConfig::default()
    };

    if models_config.is_empty() {
        anyhow::bail!("No models configured. Add models to models.toml");
    }

    println!(
        "  {} FastEmbed models, {} MistralRS models",
        models_config.fastembed.len(),
        models_config.mistralrs.len()
    );

    // Load and validate query file
    println!("\nLoading queries from {:?}...", queries_path);
    let query_file = QueryFile::load(queries_path)?;
    println!(
        "  Loaded {} queries from '{}'",
        query_file.queries.len(),
        query_file.metadata.name
    );

    // Build corpus config
    let mut config = CorpusConfig::new()
        .with_strategy(strategy)
        .with_max_chunk_chars(max_chars)
        .with_context_budget(context_budget)
        .with_code_chunk_lines(code_lines)
        .with_doc_chunk_lines(doc_lines)
        .with_overlap(overlap);

    if let Some(exts) = extensions {
        let ext_refs: Vec<&str> = exts.iter().map(|s| s.as_str()).collect();
        config = config.with_extensions(ext_refs);
    }

    if let Some(skip) = skip_patterns {
        let skip_refs: Vec<&str> = skip.iter().map(|s| s.as_str()).collect();
        config = config.with_skip_patterns(skip_refs);
    }

    // Determine corpus name
    let corpus_name = name.unwrap_or_else(|| {
        corpus_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("corpus")
            .to_string()
    });

    // Load corpus - use eprintln for progress (line-buffered even when piped)
    eprintln!(
        "\nLoading corpus '{}' from {:?}...",
        corpus_name, corpus_path
    );
    eprintln!("  Strategy: {} - {}", strategy.name(), strategy.description());

    let corpus = load_corpus_generic(corpus_path, &corpus_name, &config)?;

    // Handle 0 as unlimited
    let effective_max = if max_chunks == 0 {
        usize::MAX
    } else {
        max_chunks
    };
    let chunk_count = corpus.chunks.len().min(effective_max);

    let code_chunk_count = corpus
        .chunks
        .iter()
        .filter(|c| c.content_type == ContentType::Code)
        .count();
    let doc_chunk_count = corpus
        .chunks
        .iter()
        .filter(|c| c.content_type == ContentType::Document)
        .count();

    eprintln!(
        "  Loaded: {} files, {} chunks [{} code + {} doc], {} lines",
        corpus.file_count,
        corpus.chunks.len(),
        code_chunk_count,
        doc_chunk_count,
        corpus.total_lines
    );
    eprintln!("  Using: {} chunks", chunk_count);

    // Build corpus stats
    let corpus_stats = CorpusStats {
        name: corpus_name.clone(),
        file_count: corpus.file_count,
        chunk_count,
        code_chunk_count,
        doc_chunk_count,
        total_lines: corpus.total_lines,
    };

    // Get chunks for embedding
    let chunks: Vec<_> = corpus.chunks.iter().take(chunk_count).collect();
    let chunk_texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();

    // Debug: show sample chunks
    eprintln!("\n  Sample chunks:");
    for chunk in chunks.iter().take(3) {
        let preview: String = chunk.content.chars().take(80).collect();
        eprintln!("    - {}... ({}:{})", preview.replace('\n', " "), chunk.file_path, chunk.start_line);
    }

    let mut model_results = Vec::new();

    // Run FastEmbed models
    if !models_config.fastembed.is_empty() {
        eprintln!("\n┌─────────────────────────────────────────────────────────────┐");
        eprintln!("│                FASTEMBED MODELS (ONNX/CPU)                  │");
        eprintln!("└─────────────────────────────────────────────────────────────┘\n");

        for model_config in &models_config.fastembed {
            let model_name = format!("fastembed-{}", model_config.model.replace('/', "-"));
            eprintln!("▶ Testing: {} (max_tokens: {:?})", model_config.model, model_config.max_tokens);

            // Map model ID to FastEmbedModel enum
            let model_type = match model_config.model.as_str() {
                "BAAI/bge-small-en-v1.5" => FastEmbedModel::BgeSmallEnV15,
                "BAAI/bge-base-en-v1.5" => FastEmbedModel::BgeBaseEnV15,
                "BAAI/bge-large-en-v1.5" => FastEmbedModel::BgeLargeEnV15,
                "BAAI/bge-small-en-v1.5-q" => FastEmbedModel::BgeSmallEnV15Q,
                "thenlper/gte-small" | "thenlper/gte-base" | "thenlper/gte-large" => {
                    // GTE models - only large is available in fastembed
                    FastEmbedModel::GteLargeEnV15
                }
                "google/embedding-gemma-300m" => FastEmbedModel::EmbeddingGemma300M,
                "jinaai/jina-embeddings-v2-base-code" => FastEmbedModel::JinaEmbeddingsV2BaseCode,
                "nomic-ai/nomic-embed-text-v1.5" => FastEmbedModel::NomicEmbedTextV15,
                "Snowflake/snowflake-arctic-embed-m" => FastEmbedModel::SnowflakeArcticEmbedM,
                "Snowflake/snowflake-arctic-embed-l" => FastEmbedModel::SnowflakeArcticEmbedL,
                "intfloat/multilingual-e5-base" => FastEmbedModel::MultilingualE5Base,
                "intfloat/multilingual-e5-large" => FastEmbedModel::MultilingualE5Large,
                other => {
                    eprintln!("  Unknown FastEmbed model: {}, skipping", other);
                    continue;
                }
            };

            // Create resource monitor before model load
            let mut resource_monitor = ResourceMonitor::new();
            resource_monitor.snapshot_baseline();

            match FastEmbedBackend::new(model_type) {
                Ok(embedder) => {
                    match run_model_benchmark(
                        &model_name,
                        strategy.name(),
                        &embedder,
                        &chunks,
                        &chunk_texts,
                        &query_file.queries,
                        resource_monitor,
                    )
                    .await
                    {
                        Ok(metrics) => model_results.push(metrics),
                        Err(e) => eprintln!("  Benchmark failed: {}", e),
                    }
                }
                Err(e) => eprintln!("  Failed to initialize: {}", e),
            }
        }
    }

    // Run MistralRS models
    if !models_config.mistralrs.is_empty() {
        eprintln!("\n┌─────────────────────────────────────────────────────────────┐");
        eprintln!("│              MISTRALRS MODELS (Candle/Metal)                │");
        eprintln!("└─────────────────────────────────────────────────────────────┘\n");

        for model_config in &models_config.mistralrs {
            let model_name = format!(
                "mistralrs-{}-{:?}",
                model_config.model.replace('/', "-"),
                model_config.quantization
            );
            eprintln!(
                "▶ Testing: {} (quant: {:?}, device: {:?})",
                model_config.model, model_config.quantization, model_config.device
            );

            // Map model ID to MistralRsModel enum
            let model_type = match model_config.model.as_str() {
                "embedding-gemma" | "embedding-gemma-300m" => MistralRsModel::EmbeddingGemma300M,
                "qwen3-0.6b" => MistralRsModel::Qwen3Embedding06B,
                "qwen3-4b" => MistralRsModel::Qwen3Embedding4B,
                "qwen3-8b" => MistralRsModel::Qwen3Embedding8B,
                other => {
                    eprintln!("  Unknown MistralRS model: {}, skipping", other);
                    continue;
                }
            };

            let quant = match model_config.quantization {
                config::MistralRsQuantization::None => MistralRsQuantization::None,
                config::MistralRsQuantization::Q8 => MistralRsQuantization::Q8_0,
                config::MistralRsQuantization::Q4k => MistralRsQuantization::Q4K,
            };

            let device = match model_config.device {
                config::MistralRsDevice::Cpu => MistralRsDevice::Cpu,
                config::MistralRsDevice::Metal => MistralRsDevice::Metal,
                config::MistralRsDevice::Cuda => {
                    eprintln!("  CUDA not available on this platform, using Metal");
                    MistralRsDevice::Metal
                }
            };

            // Create resource monitor before model load
            let mut resource_monitor = ResourceMonitor::new();
            resource_monitor.snapshot_baseline();

            match MistralRsBackend::new(model_type, quant, device).await {
                Ok(embedder) => {
                    match run_model_benchmark(
                        &model_name,
                        strategy.name(),
                        &embedder,
                        &chunks,
                        &chunk_texts,
                        &query_file.queries,
                        resource_monitor,
                    )
                    .await
                    {
                        Ok(metrics) => model_results.push(metrics),
                        Err(e) => eprintln!("  Benchmark failed: {}", e),
                    }
                }
                Err(e) => eprintln!("  Failed to initialize: {}", e),
            }
        }
    }

    // Print summary
    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                      DUAL BENCHMARK SUMMARY                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    // File-level metrics
    println!("┌─ FILE LEVEL (Did we find the right file?) ─────────────────────────────┐");
    println!(
        "{:35} {:>10} {:>10} {:>10} {:>10}",
        "Model", "nDCG@10", "MRR", "R@10", "P@1"
    );
    println!("{}", "─".repeat(75));
    for model in &model_results {
        println!(
            "{:35} {:>10.3} {:>10.3} {:>9.1}% {:>9.0}%",
            &model.model_name[..35.min(model.model_name.len())],
            model.quality.mean_ndcg_at_10,
            model.quality.mrr,
            model.quality.mean_recall_at_10 * 100.0,
            model.quality.mean_precision_at_1 * 100.0,
        );
    }

    // Answer-level metrics
    println!("\n┌─ ANSWER LEVEL (Did we find the right chunk?) ────────────────────────┐");
    println!(
        "{:35} {:>10} {:>10} {:>10} {:>10}",
        "Model", "Hit@1", "Hit@3", "Hit@5", "MRR@10"
    );
    println!("{}", "─".repeat(75));
    for model in &model_results {
        if model.answer.query_count > 0 {
            println!(
                "{:35} {:>9.1}% {:>9.1}% {:>9.1}% {:>10.3}",
                &model.model_name[..35.min(model.model_name.len())],
                model.answer.hit_at_1 * 100.0,
                model.answer.hit_at_3 * 100.0,
                model.answer.hit_at_5 * 100.0,
                model.answer.mrr,
            );
        } else {
            println!(
                "{:35} {:>10}",
                &model.model_name[..35.min(model.model_name.len())],
                "(no answer ground truth)"
            );
        }
    }

    // Resource metrics
    println!("\n┌─ RESOURCE USAGE ─────────────────────────────────────────────────────────┐");
    println!(
        "{:35} {:>10} {:>12} {:>12} {:>12}",
        "Model", "Load(s)", "Embed(s)", "Throughput", "Peak RAM"
    );
    println!("{}", "─".repeat(81));
    for model in &model_results {
        println!(
            "{:35} {:>10.1} {:>12.1} {:>10.1}/s {:>10.0}MB",
            &model.model_name[..35.min(model.model_name.len())],
            model.resources.model_load_time_secs,
            model.resources.total_embed_time_secs,
            model.resources.throughput_per_sec,
            model.resources.peak_memory_mb,
        );
    }

    // Print queries skipped info if any
    let total_skipped: usize = model_results.iter().map(|m| m.quality.queries_skipped).max().unwrap_or(0);
    if total_skipped > 0 {
        println!("\n  ⚠ {} queries skipped due to R=0 (expected files not found in corpus)", total_skipped);
    }

    // Save results
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let result = CodebaseResult {
        codebase: corpus_name,
        corpus_path: corpus_path.to_string_lossy().to_string(),
        query_file: queries_path.to_string_lossy().to_string(),
        strategy: strategy.name().to_string(),
        file_count: corpus.file_count,
        chunk_count,
        query_count: query_file.queries.len(),
        models: model_results,
    };

    #[derive(serde::Serialize)]
    struct BenchmarkOutput {
        timestamp: String,
        config: BenchmarkConfigOutput,
        result: CodebaseResult,
    }

    #[derive(serde::Serialize)]
    struct BenchmarkConfigOutput {
        strategy: String,
        max_chars: usize,
        context_budget: usize,
        code_lines: usize,
        doc_lines: usize,
        overlap: f32,
        max_chunks: usize,
    }

    let output_data = BenchmarkOutput {
        timestamp: chrono::Utc::now().to_rfc3339(),
        config: BenchmarkConfigOutput {
            strategy: strategy.name().to_string(),
            max_chars,
            context_budget,
            code_lines,
            doc_lines,
            overlap,
            max_chunks,
        },
        result,
    };

    let json = serde_json::to_string_pretty(&output_data)?;
    std::fs::write(output, &json)?;
    println!("\nResults saved to {:?}", output);

    Ok(())
}

/// Multi-codebase benchmark result for a single codebase
#[derive(Debug, Clone, serde::Serialize)]
struct CodebaseResult {
    codebase: String,
    corpus_path: String,
    query_file: String,
    strategy: String,
    file_count: usize,
    chunk_count: usize,
    query_count: usize,
    models: Vec<ModelMetrics>,
}

/// Model metrics for a codebase
#[derive(Debug, Clone, serde::Serialize)]
struct ModelMetrics {
    model_name: String,
    strategy: String,
    quality: FileQualityMetrics,
    answer: AnswerMetrics,
    confidence: ModelConfidenceMetrics,
    resources: ResourceMetrics,
}

/// Run benchmark for a single model on a corpus with FILE-LEVEL evaluation
///
/// Key changes from chunk-level evaluation:
/// 1. Builds FileIndex for O(1) file path resolution
/// 2. Filters chunks by query_type (code queries → code chunks, doc queries → doc chunks)
/// 3. Tracks original embedding indices to avoid index misalignment
/// 4. Aggregates chunk scores to file scores (max per file)
/// 5. Computes file-level metrics with correct IDCG (R = total expected files)
async fn run_model_benchmark(
    model_name: &str,
    strategy: &str,
    embedder: &dyn EmbedderBackend,
    chunks: &[&corpus::Chunk],
    chunk_texts: &[String],
    queries: &[BenchmarkQuery],
    mut resource_monitor: ResourceMonitor,
) -> Result<ModelMetrics> {
    use std::collections::HashSet;
    use std::time::Instant;

    // Record model loaded (monitor was created before model, so we capture load time)
    resource_monitor.record_model_loaded(embedder.load_duration());

    // Warmup
    embedder.warmup().await?;

    let chunk_count = chunks.len();

    // =========================================================================
    // STEP 1: Build file indices for code and doc files separately
    // =========================================================================
    let code_files: Vec<String> = chunks
        .iter()
        .filter(|c| c.content_type == ContentType::Code)
        .map(|c| normalize_path(&c.file_path))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    let doc_files: Vec<String> = chunks
        .iter()
        .filter(|c| c.content_type == ContentType::Document)
        .map(|c| normalize_path(&c.file_path))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    let code_file_index = FileIndex::new(code_files.into_iter());
    let doc_file_index = FileIndex::new(doc_files.into_iter());

    // Use eprintln for progress (line-buffered even when piped)
    eprintln!("  File index: {} code files, {} doc files",
        code_file_index.files().len(),
        doc_file_index.files().len()
    );

    // =========================================================================
    // STEP 2: Embed all chunks (once, reused for all queries)
    // =========================================================================
    // Log chunk size statistics for debugging
    let chunk_sizes: Vec<usize> = chunk_texts.iter().map(|t| t.len()).collect();
    let min_size = chunk_sizes.iter().min().copied().unwrap_or(0);
    let max_size = chunk_sizes.iter().max().copied().unwrap_or(0);
    let avg_size = chunk_sizes.iter().sum::<usize>() / chunk_sizes.len().max(1);
    let over_2000 = chunk_sizes.iter().filter(|&&s| s > 2000).count();
    let over_4000 = chunk_sizes.iter().filter(|&&s| s > 4000).count();
    eprintln!("  Chunk sizes: min={}, max={}, avg={}, >2000: {}, >4000: {}",
        min_size, max_size, avg_size, over_2000, over_4000);

    let batch_size = 32;
    let total_batches = (chunk_count + batch_size - 1) / batch_size;
    eprintln!("  Embedding {} chunks in {} batches...", chunk_count, total_batches);

    // Start resource monitoring during embedding
    let _sampling_handle = resource_monitor.start_sampling();
    let embed_start = Instant::now();

    let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(chunk_count);

    for (batch_idx, batch) in chunk_texts.chunks(batch_size).enumerate() {
        // Progress every 10 batches or on first/last batch
        if batch_idx % 10 == 0 || batch_idx == total_batches - 1 {
            eprintln!("    Batch {}/{} ({} chunks embedded)",
                batch_idx + 1, total_batches, all_embeddings.len());
        }
        let result = embedder.embed_batch(&batch.to_vec()).await?;
        all_embeddings.extend(result.embeddings);
    }

    let embed_duration = embed_start.elapsed();
    drop(_sampling_handle); // Stop sampling

    eprintln!("  Embedding complete: {} vectors in {:.1}s ({:.1}/s)",
        all_embeddings.len(),
        embed_duration.as_secs_f64(),
        chunk_count as f64 / embed_duration.as_secs_f64()
    );

    // =========================================================================
    // STEP 3: Evaluate each query at FILE and ANSWER level
    // =========================================================================
    let mut file_query_metrics = Vec::new();
    let mut answer_query_metrics = Vec::new();
    let mut queries_skipped = 0;
    let mut all_warnings: Vec<FileResolutionWarning> = Vec::new();

    for query_def in queries {
        // Select appropriate file index based on query type
        let file_index = match query_def.query_type {
            QueryType::Code => &code_file_index,
            QueryType::Doc => &doc_file_index,
        };

        // Resolve expected files against the appropriate index
        let (resolved_expected, warnings) = file_index.resolve_expected_files(&query_def.expected_files);

        // Track warnings for reporting
        if !warnings.is_empty() {
            all_warnings.extend(warnings.clone());
        }

        // R = total relevant files (resolved successfully AND exist in searchable corpus)
        let total_relevant = resolved_expected.len();

        // Skip queries with R=0 (no expected files could be resolved)
        if total_relevant == 0 {
            queries_skipped += 1;
            eprintln!(
                "  ⚠ Skipping query '{}': R=0 (no expected files resolved)",
                &query_def.query[..query_def.query.len().min(50)]
            );
            continue;
        }

        // Filter chunks by query type, tracking ORIGINAL indices into embeddings array
        // CRITICAL: This prevents embedding index misalignment
        let candidate_indices: Vec<usize> = chunks
            .iter()
            .enumerate()
            .filter(|(_, chunk)| match query_def.query_type {
                QueryType::Code => chunk.content_type == ContentType::Code,
                QueryType::Doc => chunk.content_type == ContentType::Document,
            })
            .map(|(idx, _)| idx)
            .collect();

        if candidate_indices.is_empty() {
            queries_skipped += 1;
            eprintln!(
                "  ⚠ Skipping query '{}': no {} chunks in corpus",
                &query_def.query[..query_def.query.len().min(50)],
                query_def.query_type.name()
            );
            continue;
        }

        // Embed query
        let query_result = embedder.embed(&query_def.query).await?;
        let query_embedding = query_result.embedding;

        // Calculate similarities ONLY for candidate chunks (using original indices)
        let mut scored: Vec<(usize, f32)> = candidate_indices
            .iter()
            .map(|&idx| (idx, cosine_similarity(&query_embedding, &all_embeddings[idx])))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Aggregate chunks to files using zero-copy iterator
        // This avoids cloning full content for every chunk - only creates previews when needed
        let scored_iter = scored.iter().map(|(idx, score)| {
            let chunk = &chunks[*idx];
            (
                chunk.id.as_str(),
                chunk.file_path.as_str(),
                *score,
                chunk.content.as_str(),
            )
        });
        let file_results = aggregate_scored_chunks(scored_iter, &resolved_expected);

        // Compute file-level metrics
        let metrics = FileQueryMetrics::evaluate(
            &query_def.query,
            query_def.difficulty.name(),
            query_def.query_type.name(),
            &file_results,
            total_relevant,
            warnings,
        );
        file_query_metrics.push(metrics);

        // Compute answer-level metrics (if query has answer ground truth)
        let answer_metrics = if query_def.has_answer_ground_truth() {
            // Build chunk matches: (rank, score, matches_answer)
            let chunk_matches: Vec<(usize, f32, bool)> = scored
                .iter()
                .take(10) // Only check top 10 chunks
                .enumerate()
                .map(|(i, (idx, score))| {
                    let chunk = &chunks[*idx];
                    let matches = query_def.chunk_matches_answer(
                        &chunk.content,
                        &chunk.file_path,
                        chunk.start_line,
                        chunk.end_line,
                    );
                    (i + 1, *score, matches) // 1-indexed rank
                })
                .collect();

            AnswerQueryMetrics::evaluate(&query_def.query, query_def.query_type.name(), &chunk_matches)
        } else {
            AnswerQueryMetrics::no_ground_truth(&query_def.query, query_def.query_type.name())
        };
        answer_query_metrics.push(answer_metrics);
    }

    // =========================================================================
    // STEP 4: Aggregate metrics
    // =========================================================================
    let quality = FileQualityMetrics::aggregate(
        model_name,
        strategy,
        file_query_metrics.clone(),
        queries_skipped,
    );

    let answer = AnswerMetrics::aggregate(answer_query_metrics);

    // Calculate confidence metrics (using file-level results converted to format needed)
    let confidence = calculate_file_confidence(model_name, &file_query_metrics);

    // Finalize resource metrics
    let resources = resource_monitor.finalize(embed_duration, chunk_count);

    println!("  File:   {}", quality.format_summary());
    println!("  Answer: {}", answer.format_summary());
    if !all_warnings.is_empty() {
        println!("  ⚠ {} file resolution warnings (see results for details)", all_warnings.len());
    }
    println!("  Confidence: {}", confidence.format_summary());
    println!("  Reliability: {}", confidence.reliability_interpretation());
    println!("  Resources: {}", resources.format_summary());

    Ok(ModelMetrics {
        model_name: model_name.to_string(),
        strategy: strategy.to_string(),
        quality,
        answer,
        confidence,
        resources,
    })
}

/// Calculate confidence metrics from file-level results
///
/// This creates a simplified confidence metric structure that's compatible
/// with the existing ModelConfidenceMetrics format.
fn calculate_file_confidence(
    model_name: &str,
    query_metrics: &[FileQueryMetrics],
) -> ModelConfidenceMetrics {
    let query_count = query_metrics.len();

    if query_count == 0 {
        return ModelConfidenceMetrics {
            model_name: model_name.to_string(),
            query_count: 0,
            mean_score_gap: 0.0,
            std_score_gap: 0.0,
            mean_within_query_std: 0.0,
            mean_top_1_score: 0.0,
            std_top_1_score: 0.0,
            score_relevance_correlation: 0.0,
            ndcg_ci: ConfidenceInterval {
                mean: 0.0,
                lower: 0.0,
                upper: 0.0,
                confidence_level: 0.95,
            },
            ci_width_percent: 0.0,
            difficulty_consistency: 0.0,
            query_confidence: vec![],
        };
    }

    // Calculate statistics from file-level results
    let top_scores: Vec<f32> = query_metrics.iter().map(|m| m.top_score).collect();
    let mean_top_1_score = top_scores.iter().sum::<f32>() / query_count as f32;
    let std_top_1_score = std_dev(&top_scores);

    // Create synthetic QueryConfidenceMetrics from file results
    let query_confidence: Vec<QueryConfidenceMetrics> = query_metrics
        .iter()
        .map(|m| {
            let scores: Vec<f32> = m.top_files.iter().map(|f| f.score).collect();
            let top_1 = scores.first().copied().unwrap_or(0.0);
            let top_2 = scores.get(1).copied().unwrap_or(0.0);
            let (mean, std) = if scores.is_empty() {
                (0.0, 0.0)
            } else {
                let mean = scores.iter().sum::<f32>() / scores.len() as f32;
                let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / scores.len() as f32;
                (mean, variance.sqrt())
            };
            let range = if scores.len() >= 2 {
                scores.first().copied().unwrap_or(0.0) - scores.last().copied().unwrap_or(0.0)
            } else {
                0.0
            };

            // top_1_relevant: check if the FIRST file is relevant, not just any in top-10
            let top_1_relevant = m.top_files.first().map(|f| f.relevant).unwrap_or(false);

            QueryConfidenceMetrics {
                query: m.query.clone(),
                top_1_score: top_1,
                top_2_score: top_2,
                score_gap: top_1 - top_2,
                top_10_mean: mean,
                top_10_std: std,
                top_10_range: range,
                cv: if mean > 0.0 { std / mean } else { 0.0 },
                top_1_relevant,
            }
        })
        .collect();

    // Score gap statistics
    let score_gaps: Vec<f32> = query_confidence.iter().map(|q| q.score_gap).collect();
    let mean_score_gap = score_gaps.iter().sum::<f32>() / query_count as f32;
    let std_score_gap = std_dev(&score_gaps);

    // Within-query std
    let within_stds: Vec<f32> = query_confidence.iter().map(|q| q.top_10_std).collect();
    let mean_within_query_std = within_stds.iter().sum::<f32>() / query_count as f32;

    // Score-relevance correlation
    let relevance: Vec<f32> = query_confidence
        .iter()
        .map(|q| if q.top_1_relevant { 1.0 } else { 0.0 })
        .collect();
    let score_relevance_correlation = pearson_correlation(&top_scores, &relevance);

    // Bootstrap CI for nDCG
    let ndcg_values: Vec<f32> = query_metrics.iter().map(|m| m.ndcg_at_10).collect();
    let ndcg_ci = bootstrap_confidence_interval(&ndcg_values, 0.95, 10_000);
    let ci_width_percent = if ndcg_ci.mean > 0.0 {
        ((ndcg_ci.upper - ndcg_ci.lower) / ndcg_ci.mean) * 100.0
    } else {
        0.0
    };

    // Difficulty consistency
    let difficulty_consistency = calculate_file_difficulty_consistency(query_metrics);

    ModelConfidenceMetrics {
        model_name: model_name.to_string(),
        query_count,
        mean_score_gap,
        std_score_gap,
        mean_within_query_std,
        mean_top_1_score,
        std_top_1_score,
        score_relevance_correlation,
        ndcg_ci,
        ci_width_percent,
        difficulty_consistency,
        query_confidence,
    }
}

/// Calculate difficulty consistency for file-level metrics
fn calculate_file_difficulty_consistency(query_metrics: &[FileQueryMetrics]) -> f32 {
    use std::collections::HashMap;

    let mut by_difficulty: HashMap<&str, Vec<f32>> = HashMap::new();
    for qm in query_metrics {
        by_difficulty
            .entry(&qm.difficulty)
            .or_default()
            .push(qm.ndcg_at_10);
    }

    if by_difficulty.len() < 2 {
        return 1.0;
    }

    let means: Vec<f32> = by_difficulty
        .values()
        .map(|scores| scores.iter().sum::<f32>() / scores.len() as f32)
        .collect();

    let overall_mean = means.iter().sum::<f32>() / means.len() as f32;
    let cv = std_dev(&means) / overall_mean.max(0.001);

    (1.0 - cv).max(0.0).min(1.0)
}

/// Validate a query file
fn validate_queries(path: &PathBuf) -> Result<()> {
    println!("Validating {:?}...", path);

    let query_file = QueryFile::load(path)?;

    println!("✓ Valid query file");
    println!("  Name: {}", query_file.metadata.name);
    println!("  Description: {}", query_file.metadata.description);
    println!("  Queries: {}", query_file.queries.len());

    // Count by difficulty
    let mut simple = 0;
    let mut medium = 0;
    let mut hard = 0;
    let mut extra_hard = 0;

    for q in &query_file.queries {
        match q.difficulty {
            queries::QueryDifficulty::Simple => simple += 1,
            queries::QueryDifficulty::Medium => medium += 1,
            queries::QueryDifficulty::Hard => hard += 1,
            queries::QueryDifficulty::ExtraHard => extra_hard += 1,
        }
    }

    println!("  Difficulty distribution:");
    println!("    Simple: {}", simple);
    println!("    Medium: {}", medium);
    println!("    Hard: {}", hard);
    println!("    ExtraHard: {}", extra_hard);

    Ok(())
}

fn list_models() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              AVAILABLE EMBEDDING MODELS                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("FASTEMBED (ONNX Runtime / CPU):");
    println!("─────────────────────────────────────────────────────────────────");
    for model in FastEmbedModel::all() {
        println!(
            "  {:30} {:4} dims  {}",
            model.name(),
            model.dimensions(),
            model.model_id()
        );
    }

    println!("\nMISTRALRS (Candle / Metal GPU):");
    println!("─────────────────────────────────────────────────────────────────");
    for model in MistralRsModel::all() {
        println!(
            "  {:30} {:4} dims  ~{:.1}GB  {}",
            model.name(),
            model.dimensions(),
            model.size_gb(),
            model.model_id()
        );
    }

    println!("\nQUANTIZATION OPTIONS (mistralrs):");
    println!("─────────────────────────────────────────────────────────────────");
    println!("  none   - F16 (half precision)");
    println!("  q8_0   - 8-bit quantization");
    println!("  q4k    - 4-bit quantization (Q4K)");

    println!("\nCONFIGURATION:");
    println!("─────────────────────────────────────────────────────────────────");
    println!("  Models are configured in models.toml. Example:");
    println!();
    println!("  [[fastembed]]");
    println!("  model = \"BAAI/bge-small-en-v1.5\"");
    println!("  max_tokens = 512");
    println!();
    println!("  [[mistralrs]]");
    println!("  model = \"embedding-gemma\"");
    println!("  quantization = \"q4k\"");
    println!("  device = \"metal\"");
}

async fn test_single_model(model: &str, backend: &str, quantization: &str) -> Result<()> {
    println!(
        "Testing single model: {} (backend: {}, quant: {})\n",
        model, backend, quantization
    );

    let sample_texts = corpus::load_sample_texts();

    match backend.to_lowercase().as_str() {
        "fastembed" => {
            let model_type = match model.to_lowercase().as_str() {
                "bge-small" | "bge-small-en-v1.5" => FastEmbedModel::BgeSmallEnV15,
                "bge-small-q" => FastEmbedModel::BgeSmallEnV15Q,
                "bge-base" => FastEmbedModel::BgeBaseEnV15,
                "bge-large" => FastEmbedModel::BgeLargeEnV15,
                "embedding-gemma" | "embeddinggemma" | "gemma" => FastEmbedModel::EmbeddingGemma300M,
                "jina-code" | "jina" => FastEmbedModel::JinaEmbeddingsV2BaseCode,
                "gte-large" | "gte" => FastEmbedModel::GteLargeEnV15,
                "nomic" => FastEmbedModel::NomicEmbedTextV15,
                "snowflake-m" | "arctic-m" => FastEmbedModel::SnowflakeArcticEmbedM,
                "snowflake-l" | "arctic-l" => FastEmbedModel::SnowflakeArcticEmbedL,
                _ => {
                    println!("Unknown fastembed model: {}", model);
                    println!("Available: bge-small, bge-small-q, bge-base, bge-large, embedding-gemma, jina-code, gte-large, nomic, snowflake-m, snowflake-l");
                    return Ok(());
                }
            };

            let backend = FastEmbedBackend::new(model_type)?;
            println!(
                "Model loaded: {} ({} dims)\n",
                backend.name(),
                backend.dimensions()
            );

            // Test embedding
            let result = backend.embed("test query for embedding").await?;
            println!(
                "Single embedding: {:?} ({} dims)",
                result.duration,
                result.embedding.len()
            );

            // Batch test
            let batch_result = backend.embed_batch(&sample_texts[..10].to_vec()).await?;
            println!(
                "Batch embedding (10): {:?} total, {:?} avg",
                batch_result.duration, batch_result.avg_duration
            );
        }

        "mistralrs" => {
            let model_type = match model.to_lowercase().as_str() {
                "embedding-gemma" | "embeddinggemma" | "gemma" => MistralRsModel::EmbeddingGemma300M,
                "qwen3-0.6b" | "qwen3" => MistralRsModel::Qwen3Embedding06B,
                "qwen3-4b" => MistralRsModel::Qwen3Embedding4B,
                "qwen3-8b" => MistralRsModel::Qwen3Embedding8B,
                _ => {
                    println!("Unknown mistralrs model: {}", model);
                    println!("Available: embedding-gemma, qwen3-0.6b, qwen3-4b, qwen3-8b");
                    return Ok(());
                }
            };

            let quant = match quantization.to_lowercase().as_str() {
                "none" | "f16" => MistralRsQuantization::None,
                "q8_0" | "q8" => MistralRsQuantization::Q8_0,
                "q4k" | "q4_k" => MistralRsQuantization::Q4K,
                "q4_0" | "q4" => MistralRsQuantization::Q4_0,
                _ => MistralRsQuantization::None,
            };

            let backend =
                MistralRsBackend::new(model_type, quant, MistralRsDevice::Metal).await?;
            println!(
                "Model loaded: {} ({} dims)\n",
                backend.name(),
                backend.dimensions()
            );

            // Test embedding
            let result = backend.embed("test query for embedding").await?;
            println!(
                "Single embedding: {:?} ({} dims)",
                result.duration,
                result.embedding.len()
            );

            // Batch test
            let batch_result = backend.embed_batch(&sample_texts[..10].to_vec()).await?;
            println!(
                "Batch embedding (10): {:?} total, {:?} avg",
                batch_result.duration, batch_result.avg_duration
            );
        }

        _ => {
            println!(
                "Unknown backend: {}. Use 'fastembed' or 'mistralrs'",
                backend
            );
        }
    }

    Ok(())
}
