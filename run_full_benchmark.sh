#!/bin/bash
# Full Embedding Benchmark Runner
# Runs all combinations: {ripgrep, fastapi} x {line, semantic} strategies
# All models from models.toml are tested in each run

set -e  # Exit on error

cd "$(dirname "$0")"

# Create results directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo "Full Embedding Benchmark Suite"
echo "Results: $RESULTS_DIR"
echo "=============================================="
echo ""

# Track timing
START_TIME=$(date +%s)

# Define test matrix
CODEBASES=("ripgrep" "fastapi")
STRATEGIES=("line" "semantic")

run_benchmark() {
    local codebase=$1
    local strategy=$2
    local output="${RESULTS_DIR}/${codebase}_${strategy}.json"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Running: $codebase with $strategy chunking"
    echo "Output: $output"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local run_start=$(date +%s)

    ./target/release/embedding-benchmark run \
        --corpus "codebases/$codebase" \
        --queries "queries/${codebase}.json" \
        --strategy "$strategy" \
        --output "$output" \
        --name "$codebase" \
        --models-config models.toml

    local run_end=$(date +%s)
    local run_duration=$((run_end - run_start))
    echo ""
    echo "✓ Completed $codebase/$strategy in ${run_duration}s"
    echo ""
}

# Run all combinations sequentially
for codebase in "${CODEBASES[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
        run_benchmark "$codebase" "$strategy"
    done
done

# Calculate total time
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
MINUTES=$((TOTAL_DURATION / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo "=============================================="
echo "Benchmark Complete!"
echo "=============================================="
echo "Total time: ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results saved to:"
for f in "$RESULTS_DIR"/*.json; do
    echo "  - $f"
done
echo ""

# Generate summary if Python is available
if command -v python3 &> /dev/null; then
    echo "Generating summary..."
    python3 << 'PYTHON_SCRIPT'
import json
import os
from glob import glob

results_dir = os.environ.get('RESULTS_DIR', 'results')
if not results_dir.startswith('results/'):
    # Find most recent results dir
    dirs = sorted(glob('results/*/'), reverse=True)
    if dirs:
        results_dir = dirs[0].rstrip('/')

print(f"\n{'='*70}")
print(f"SUMMARY: {results_dir}")
print(f"{'='*70}\n")

for json_file in sorted(glob(f'{results_dir}/*.json')):
    try:
        with open(json_file) as f:
            data = json.load(f)

        name = os.path.basename(json_file).replace('.json', '')
        print(f"📊 {name}")
        print("-" * 50)

        for model_result in data.get('model_results', []):
            model = model_result.get('model', 'unknown')
            file_metrics = model_result.get('file_quality', {})
            answer_metrics = model_result.get('answer', {})

            file_ndcg = file_metrics.get('summary', {}).get('ndcg_at_10', 0)
            answer_ndcg = answer_metrics.get('ndcg_at_10', 0)

            # Get by_query_type if available
            by_type = answer_metrics.get('by_query_type', {})
            code_ndcg = by_type.get('Code', {}).get('ndcg_at_10', '-')
            doc_ndcg = by_type.get('Doc', {}).get('ndcg_at_10', '-')

            if isinstance(code_ndcg, float):
                code_ndcg = f"{code_ndcg:.3f}"
            if isinstance(doc_ndcg, float):
                doc_ndcg = f"{doc_ndcg:.3f}"

            print(f"  {model:40} File: {file_ndcg:.3f}  Ans: {answer_ndcg:.3f}  (Code: {code_ndcg}, Doc: {doc_ndcg})")
        print()
    except Exception as e:
        print(f"Error reading {json_file}: {e}")

print(f"{'='*70}")
PYTHON_SCRIPT
fi

echo ""
echo "Done! Review results in: $RESULTS_DIR"
