# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Embedding Benchmark is a Rust CLI tool for evaluating embedding models on code search tasks. It compares FastEmbed (ONNX/CPU) and MistralRS (Candle/Metal GPU) backends using file-level and answer-level retrieval metrics.

## Build and Run Commands

```bash
# Build (release recommended for benchmarks)
cargo build --release

# Run a benchmark
cargo run --release -- run \
    --corpus codebases/ripgrep \
    --queries queries/ripgrep.json \
    --strategy semantic \
    --models-config models.toml \
    --output results/benchmark.json

# Run tests
cargo test

# List available models
cargo run --release -- list

# Validate a query file
cargo run --release -- validate-queries --queries queries/ripgrep.json

# Test a single model
cargo run --release -- test embedding-gemma --backend mistralrs --quantization q4k

# Run full benchmark suite (all codebases x strategies)
./run_full_benchmark.sh
```

## Architecture

### Core Modules

- **`src/main.rs`** - CLI entry point with `run`, `list`, `test`, `validate-queries` subcommands
- **`src/embedders/`** - Unified embedding backend abstraction
  - `traits.rs` - `EmbedderBackend` trait defining the interface for all backends
  - `fastembed_backend.rs` - FastEmbed (ONNX Runtime, CPU)
  - `mistralrs_backend.rs` - MistralRS (Candle, Metal GPU)
- **`src/corpus/`** - Corpus loading and chunking
  - `loader.rs` - File loading and `CorpusConfig`
  - `chunker.rs` - Line-based chunking (baseline)
  - `semantic_chunker.rs` - AST-based chunking with context preservation
- **`src/benchmark/quality.rs`** - Retrieval metrics (nDCG@10, MRR, Hit@K, recall, precision)
- **`src/queries/mod.rs`** - Query file format with ground truth (expected files, answer regions)
- **`src/config.rs`** - `ModelsConfig` (TOML) and `Strategy` enum (line vs semantic)
- **`src/resource_monitor.rs`** - Memory and throughput tracking

### Key Data Flow

1. Load models from `models.toml` (FastEmbed and/or MistralRS configs)
2. Load queries from JSON with expected files and optional answer ground truth
3. Load corpus with chosen chunking strategy (line or semantic)
4. Embed all chunks once, then evaluate each query:
   - Filter chunks by query type (code/doc)
   - Aggregate chunk scores to file scores (max per file)
   - Compute file-level metrics (nDCG@10, MRR, recall, precision)
   - Compute answer-level metrics if ground truth exists (Hit@K, MRR)
5. Output JSON with per-model metrics

### Chunking Strategies

- **Line** - Fixed line-count chunks with overlap (baseline)
- **Semantic** - Structure-aware: AST-based for code (injects class/function context), header-based for docs (breadcrumb injection)

### Evaluation Levels

- **File-level** - Did we find the right file? (nDCG@10, MRR, recall@10, precision@1)
- **Answer-level** - Did we find the right chunk? (Hit@1/3/5, MRR@10)

## Configuration Files

- **`models.toml`** - Embedding models to benchmark (FastEmbed and MistralRS sections)
- **`queries/*.json`** - Query files with metadata, queries, expected files, and optional answer ground truth

## Query File Format

```json
{
  "metadata": { "name": "project", "description": "...", "version": "1.0" },
  "queries": [{
    "query": "regex matcher implementation",
    "query_type": "Code",
    "difficulty": "Medium",
    "expected_files": ["src/matcher.rs"],
    "description": "Find regex matcher",
    "answer_keywords": ["RegexMatcher"],
    "answer_lines": [50, 100]
  }]
}
```

## Dependencies Note

Tree-sitter grammars must use compatible ABI versions (currently 0.23 for all). MistralRS requires Metal feature on macOS.
