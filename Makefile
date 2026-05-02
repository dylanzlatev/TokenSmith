.PHONY: help env build-llama clean test test-ann test-ann-real report-ann eval-chunk-buffer eval-all run-index run-chat run-index-partial run-add-chapters-partial run-chat-partial install update-env

help:
	@echo "TokenSmith - RAG Application (Conda Dependencies)"
	@echo "Available targets:"
	@echo "  env         - Create conda environment with all dependencies"
	@echo "  update-env  - Update environment from environment.yml"
	@echo "  build-llama - Build llama.cpp (if not found)"
	@echo "  install     - Install package in development mode"
	@echo "  test        - Run all tests"
	@echo "  test-ann         - Run ANN unit + synthetic benchmark tests (no model needed)"
	@echo "  test-ann-real    - Run real-corpus ANN evaluation (requires index + model)"
	@echo "  report-ann       - Write ANN results table to tests/results/ann_report.txt"
	@echo "  eval-chunk-buffer - Evaluate buffer pool on textbook sessions → tests/results/"
	@echo "  eval-all         - Run real ANN eval + chunk buffer eval (requires index + model)"
	@echo "  clean            - Clean build artifacts"
	@echo "  show-deps   - Show installed conda packages"
	@echo "  export-env  - Export current environment"
	@echo "  run-index-partial - Create a partial index (e.g., make run-index-partial CHAPTERS=\"1 2\")"
	@echo "  run-add-chapters-partial - Add chapters to partial index (e.g., make run-add-chapters-partial CHAPTERS=\"3\")"
	@echo "  run-chat-partial - Chat using partial index"

# Environment setup - installs all dependencies via conda
env:
	@echo "Creating TokenSmith conda environment..."
	conda env create -f environment.yml -n tokensmith || conda env update -f environment.yml -n tokensmith
	@echo "Running platform-specific setup..."
	conda run -n tokensmith bash scripts/setup_env.sh

# Update environment from environment.yml
update-env:
	@echo "Updating TokenSmith conda environment..."
	conda env update -f environment.yml -n tokensmith

# Build llama.cpp if needed
build-llama:
	@echo "Checking for existing llama.cpp installation..."
	conda run -n tokensmith python scripts/detect_llama.py || conda run -n tokensmith bash scripts/build_llama.sh

# Install package in development mode (no dependencies, they're from conda)
install:
	conda run -n tokensmith pip install -e . --no-deps

# Full build process
build: env install
	@echo "TokenSmith build complete! Activate environment with: conda activate tokensmith"

# Show installed packages
show-deps:
	@echo "Installed conda packages:"
	conda list -n tokensmith

# Export current environment for sharing
export-env:
	@echo "Exporting environment to environment-lock.yml..."
	conda env export -n tokensmith > environment-lock.yml
	@echo "Environment exported with exact versions."

# Run tests
test:
	conda run -n tokensmith python -m pytest tests/ -v || echo "No tests found"

# ANN benchmark tests — no LLM or textbook index required
test-ann:
	@echo "Running ANN unit + synthetic benchmark tests..."
	conda run -n tokensmith python -m pytest \
		tests/test_ann_integration.py \
		tests/test_ann_benchmark.py \
		tests/test_chunk_buffer.py \
		-s -v $(ARGS)

# Real-corpus ANN evaluation — requires 'make run-index' and the embedding model
test-ann-real:
	@echo "Running real-corpus ANN evaluation (requires index + model)..."
	conda run -n tokensmith python -m pytest tests/test_ann_real_index.py -s -v $(ARGS)

# Aggregate results from tests/results/ann_*.json into a formatted report file
report-ann:
	@echo "Generating ANN evaluation report..."
	conda run -n tokensmith python tests/utils/ann_report.py \
		--out tests/results/ann_report.txt $(ARGS)
	@echo "Report written to tests/results/ann_report.txt"

# Chunk buffer pool evaluation — requires index + embedding model
eval-chunk-buffer:
	@echo "Running chunk buffer pool evaluation..."
	conda run -n tokensmith python tests/eval_chunk_buffer.py \
		--config config/config.yaml $(ARGS)
	@echo "Results written to tests/results/chunk_buffer_eval.json"
	@echo "Report  written to tests/results/chunk_buffer_report.txt"

# Run both real-corpus ANN eval and chunk buffer eval, then generate the ANN report
eval-all: test-ann-real eval-chunk-buffer report-ann
	@echo ""
	@echo "All evaluations complete. Output files:"
	@echo "  tests/results/ann_report.txt"
	@echo "  tests/results/ann_real_index_results.json"
	@echo "  tests/results/ann_per_question_recall.json"
	@echo "  tests/results/chunk_buffer_report.txt"
	@echo "  tests/results/chunk_buffer_eval.json"

# Clean
clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# PDF to Markdown extraction
run-extract:
	@echo "Extracting PDF to markdown (data/chapters/*.pdf -> data/*.md)"
	conda run --no-capture-output -n tokensmith python -m src.preprocessing.extraction
	
# Run modes
run-index:
	@echo "Running TokenSmith index mode with additional CLI args: $(ARGS)"
	conda run --no-capture-output -n tokensmith python -m src.main index $(ARGS)

run-chat:
	@echo "Running TokenSmith chat mode with additional CLI args: $(ARGS)"
	@echo "Note: Chat mode requires interactive terminal. If this fails, use:"
	@echo "  conda activate tokensmith && python -m src.main chat $(ARGS)"
	conda run --no-capture-output -n tokensmith --no-capture-output python -m src.main chat $(ARGS)

run-index-partial:
	@echo "Running TokenSmith partial index mode with chapters: $(CHAPTERS) $(ARGS)"
	conda run --no-capture-output -n tokensmith python -m src.main index --partial --chapters $(CHAPTERS) $(ARGS)

run-add-chapters-partial:
	@echo "Adding chapters $(CHAPTERS) to partial index with ARGS: $(ARGS)"
	conda run --no-capture-output -n tokensmith python -m src.main add-chapters --partial --chapters $(CHAPTERS) $(ARGS)

run-chat-partial:
	@echo "Running TokenSmith chat mode (partial index) with additional CLI args: $(ARGS)"
	conda run --no-capture-output -n tokensmith --no-capture-output python -m src.main chat --partial $(ARGS)
