
# att3ntion Makefile
#
# Usage:
#   make build              Build/install the extension
#   make test               Run quick correctness tests
#   make bench              Benchmark (build + test + run)
#   make bench NOBUILD=1    Benchmark without rebuilding
#   make bench-save NOTE="desc"           Save benchmark with note
#   make bench-save NOTE="desc" NOBUILD=1 Save without rebuilding
#   make history            Show benchmark history with deltas
#   make clean              Clean build artifacts

PYTHON ?= python

# --- Build ---

build:
	pip install -e .

.PHONY: maybe-build
maybe-build:
ifndef NOBUILD
	@$(MAKE) build
endif

# --- Correctness Tests ---

test: maybe-build
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "CORRECTNESS: Quick tests"
	@echo "═══════════════════════════════════════════════════════════════"
	$(PYTHON) tests/test_all_kernels_equivalence.py --quick --continue-on-failure -v

test-quiet: maybe-build
	@$(PYTHON) tests/test_all_kernels_equivalence.py --quick --continue-on-failure 2>&1 | \
		(grep -E "(FAIL|Error|Exception|Traceback)" && exit 1 || echo "  ✓ All correctness tests passed")

test-full: maybe-build
	$(PYTHON) tests/test_all_kernels_equivalence.py --continue-on-failure -v

# --- Benchmarks ---

bench: test-quiet
	$(PYTHON) tests/benchmark_optimizations.py

bench-save: test-quiet
ifndef NOTE
	$(error NOTE is required. Usage: make bench-save NOTE="description")
endif
	$(PYTHON) tests/benchmark_optimizations.py --save --note "$(NOTE)"

bench-quick: maybe-build
	$(PYTHON) tests/benchmark_optimizations.py --quick --forward-only

bench-large: test-quiet
ifndef NOTE
	$(error NOTE is required. Usage: make bench-large NOTE="description")
endif
	$(PYTHON) tests/benchmark_optimizations.py --large --save --note "$(NOTE)"

bench-forward: maybe-build
	$(PYTHON) tests/benchmark_optimizations.py --forward-only

bench-backward: maybe-build
	$(PYTHON) tests/benchmark_optimizations.py --backward-only

bench-complexity:
	$(PYTHON) tests/benchmark_optimizations.py --quick --forward-only --warmup 1 --iters 1 --complexity

# --- Profiling ---

profile-timeline: maybe-build
	@mkdir -p profiling_reports
	nsys profile -o profiling_reports/timeline_$$(date +%Y%m%d_%H%M%S) \
		$(PYTHON) tests/benchmark_optimizations.py --quick --forward-only --warmup 1 --iters 1

profile-kernel: maybe-build
	$(PYTHON) tests/cuda_profiling_script.py

# --- History ---

history:
	@$(PYTHON) tests/benchmark_optimizations.py --show-history

# --- Combined ---

all: test-full bench-save

iterate: test bench-quick

# --- Clean ---

clean:
	pip uninstall -y hyper_attn_extensions 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ __pycache__/ hyper_attn_extensions.egg-info/
	find . -name "*.so" -delete
	@echo "Cleaned build artifacts."

.PHONY: build maybe-build test test-quiet test-full \
        bench bench-save bench-quick bench-large bench-forward bench-backward bench-complexity \
        profile-timeline profile-kernel history all iterate clean
