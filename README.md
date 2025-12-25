# Embedding Benchmark

A Rust CLI tool for evaluating embedding models on code search tasks. Compares FastEmbed (ONNX/CPU) and MistralRS (Candle/Metal GPU) backends using file-level and answer-level retrieval metrics.

## Latest Results

See the full benchmark report: **[docs/public_embedding_benchmark_report.md](docs/public_embedding_benchmark_report.md)**

**TL;DR:** EmbeddingGemma-300M Q8 achieved the best balance of quality (0.891 nDCG@10) and speed (33.5 emb/sec) on Apple Silicon.

## Quick Start

### Prerequisites

- Rust 1.70+
- macOS with Apple Silicon (for Metal GPU acceleration) or any platform for CPU-only mode

### Build

```bash
cargo build --release
```

### Prepare Test Corpus

Clone a codebase into `codebases/` directory:

```bash
git clone https://github.com/BurntSushi/ripgrep codebases/ripgrep
```

### Create Query File

Create a query file (e.g., `queries/ripgrep.json`):

```json
{
  "metadata": {
    "name": "ripgrep",
    "description": "Queries for ripgrep codebase",
    "version": "1.0"
  },
  "queries": [
    {
      "query": "regex matcher implementation",
      "query_type": "Code",
      "difficulty": "Medium",
      "expected_files": ["crates/matcher/src/lib.rs"],
      "description": "Find the regex matcher trait"
    }
  ]
}
```

### Run Benchmark

```bash
cargo run --release -- run \
    --corpus codebases/ripgrep \
    --queries queries/ripgrep.json \
    --strategy line \
    --output results/benchmark.json
```

### Other Commands

```bash
# List available models
cargo run --release -- list

# Validate query file
cargo run --release -- validate-queries --queries queries/ripgrep.json

# Test a single model
cargo run --release -- test embedding-gemma --backend mistralrs --quantization q4k
```

## Configuration

### models.toml

Configure which models to benchmark:

```toml
# FastEmbed (ONNX/CPU)
[[fastembed]]
model = "BAAI/bge-small-en-v1.5"

# MistralRS (Metal GPU)
[[mistralrs]]
model = "embedding-gemma"
quantization = "q8"    # none, q8, q4k
device = "metal"       # metal, cpu
```

### Chunking Strategies

- `--strategy line` - Fixed line-count chunks with overlap (baseline)
- `--strategy semantic` - Structure-aware: AST-based for code, header-based for docs

## Project Structure

```
embedding-benchmark/
├── src/
│   ├── main.rs              # CLI entry point
│   ├── embedders/           # Backend implementations
│   │   ├── fastembed_backend.rs   # ONNX Runtime (CPU)
│   │   └── mistralrs_backend.rs   # Candle + Metal (GPU)
│   ├── corpus/              # Chunking strategies
│   │   ├── chunker.rs       # Line-based chunking
│   │   └── semantic_chunker.rs  # AST-based chunking
│   ├── benchmark/           # Metrics (nDCG, MRR, Hit@K)
│   └── queries/             # Query file format
├── models.toml              # Model configurations
├── queries/                 # Query files (JSON)
├── codebases/               # Test codebases (gitignored)
└── docs/                    # Benchmark reports
```

## Methodology

### How It Works

1. **Chunk the codebase** - Split source files into smaller pieces using a chunking strategy
2. **Embed all chunks** - Convert each chunk into a vector using an embedding model
3. **Embed queries** - Convert search queries into vectors
4. **Retrieve & rank** - Find chunks most similar to each query using cosine similarity
5. **Evaluate** - Compare retrieved results against ground truth using IR metrics

### Evaluation Metrics

We use standard Information Retrieval metrics to measure search quality:

#### nDCG@10 (Normalized Discounted Cumulative Gain)

Measures ranking quality by considering both relevance and position. Higher-ranked relevant results contribute more to the score.

```
DCG@k = Σ (relevance_i / log2(i + 1))  for i = 1 to k
nDCG@k = DCG@k / IDCG@k  (normalized by ideal ranking)
```

- **Score range**: 0.0 to 1.0 (higher is better)
- **Why it matters**: Penalizes relevant results appearing lower in the ranking
- **Reference**: [Wikipedia - nDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)

#### MRR (Mean Reciprocal Rank)

Measures how quickly you find the first relevant result.

```
RR = 1 / rank_of_first_relevant_result
MRR = average RR across all queries
```

- **Score range**: 0.0 to 1.0 (higher is better)
- **Example**: If first relevant result is at rank 3, RR = 1/3 = 0.333
- **Reference**: [Wikipedia - MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)

#### Hit@K (Hit Rate at K)

Binary metric: did a relevant result appear in the top K results?

- **Hit@1**: Was the top result relevant? (precision-focused)
- **Hit@5**: Was there a relevant result in top 5? (recall-focused)
- **Score range**: 0% to 100%

### Chunking Strategies

The chunking strategy determines how source files are split into embeddable pieces.

#### Line-Based Chunking (Baseline)

Simple approach: split files by line count with overlap.

```
Parameters:
- Code files: 50 lines per chunk
- Doc files: 100 lines per chunk
- Overlap: 25%
```

**Example output:**
```
[file: src/config.rs]

use std::path::Path;
use serde::{Deserialize, Serialize};

pub struct Config {
    pub model: String,
    pub max_tokens: usize,
}
// ... (up to 50 lines)
```

**Pros**: Simple, predictable, language-agnostic
**Cons**: May split functions mid-body, loses semantic context

#### Semantic Chunking (Structure-Aware)

AST-based chunking that respects code structure and injects context.

**For code files:**
- Parses source into AST using [tree-sitter](https://tree-sitter.github.io/)
- Chunks at semantic boundaries (functions, classes, impl blocks)
- Injects parent context (class name, module docs) into each chunk

**Example output:**
```
[file: src/auth/middleware.rs] Authentication middleware
[context: impl AuthMiddleware > fn verify_token]

fn verify_token(&self, token: &str) -> Result<Claims> {
    let decoded = jsonwebtoken::decode(token, &self.key)?;
    Ok(decoded.claims)
}
```

**For documentation (Markdown):**
- Splits at header boundaries (##, ###)
- Injects header breadcrumb trail for context

**Example output:**
```
[file: docs/api.md]
[section: # API Reference > ## Authentication > ### JWT Tokens]

JWT tokens are issued by the `/auth/login` endpoint...
```

**Pros**: Preserves semantic units, self-contained chunks with context
**Cons**: Requires language-specific parsers, more complex

### Two-Level Evaluation

We evaluate at two granularity levels:

#### File-Level Evaluation

*"Did we find the right file?"*

- Aggregates chunk scores to file scores (max score per file)
- Compares against expected files from ground truth
- Metrics: nDCG@10, MRR, Recall@10, Precision@1

#### Answer-Level Evaluation

*"Did we find the exact code location?"*

- Evaluates whether retrieved chunks contain the actual answer
- Uses keyword matching and line range overlap
- Metrics: Hit@1, Hit@3, Hit@5, MRR@10

**Why both?** File-level shows overall navigation accuracy. Answer-level shows whether the embedding captured the specific code semantics.

## Metrics Summary

| Level | Metric | Description |
|-------|--------|-------------|
| File | nDCG@10 | Ranking quality of relevant files |
| File | MRR | Reciprocal rank of first relevant file |
| Answer | Hit@K | Found answer chunk in top-K results |
| Answer | MRR@10 | Reciprocal rank of answer chunk |

*For detailed metric definitions, see [Methodology](#methodology) above.*

## Disclaimers

- **Hardware-specific results**: Benchmark results were obtained on Apple Silicon (M-series). Performance will vary on different hardware.
- **Model availability**: Some models require downloading from Hugging Face. Ensure you have sufficient disk space and network access.
- **Not production-ready**: This is a benchmarking tool for research purposes. The embedding backends and chunking strategies are not optimized for production use.
- **Query bias**: Benchmark quality depends heavily on query design. Results may not generalize to different query styles or codebases.
- **Quantization trade-offs**: While Q8/Q4K quantization showed minimal quality loss in our tests, this may vary for other use cases.

## References

### Evaluation Metrics
- [nDCG - Normalized Discounted Cumulative Gain](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) - Standard IR ranking metric
- [MRR - Mean Reciprocal Rank](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) - First relevant result metric
- [BEIR Benchmark](https://github.com/beir-cellar/beir) - Heterogeneous benchmark for information retrieval

### Embedding Models
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) - Massive Text Embedding Benchmark
- [FastEmbed](https://github.com/qdrant/fastembed) - ONNX-based embedding library
- [Mistral.rs](https://github.com/EricLBuehler/mistral.rs) - Rust inference engine with Metal support

### Code Parsing
- [Tree-sitter](https://tree-sitter.github.io/) - Incremental parsing library for code

## License

MIT
