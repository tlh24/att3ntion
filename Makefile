
# att3ntion Makefile
#
# Usage:
#   make build              			  Build/install the extension
#   make test               		      Run quick correctness tests
#   make test-full        				  Run detailed correctness tests
#   make bench              			  Benchmark (test + run)
#   make bench BUILD=1                    Rebuild before benchmarking
#   make bench-save NOTE="desc"           Save benchmark with note
#   make bench-save NOTE="desc" BUILD=1   Rebuild before saving
#   make bench-scaling                    CUDA scaling analysis
#   make bench-compare                    CUDA vs Torch comparison
#   make history                          Show benchmark history with deltas
#   make clean                            Clean build artifacts

PYTHON ?= python

# --- Build ---

build:
	pip install -e . --no-build-isolation

.PHONY: maybe-build
maybe-build:
ifdef BUILD
	@$(MAKE) build
endif

# --- Correctness Tests ---

test: maybe-build
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "CORRECTNESS: Quick tests"
	@echo "═══════════════════════════════════════════════════════════════"
	$(PYTHON) tests/test_kernel_correctness.py --quick --continue-on-failure -v

test-quiet: maybe-build
	@$(PYTHON) tests/test_kernel_correctness.py --quick --continue-on-failure 2>&1 | \
		(grep -E "(FAIL|Error|Exception|Traceback)" && exit 1 || echo "  ✓ All correctness tests passed")

test-full: maybe-build
	$(PYTHON) tests/test_kernel_correctness.py --continue-on-failure -v

# --- Benchmarks: Regression Tracking ---

bench: test-quiet
	$(PYTHON) benchmarks/bench_regression.py

bench-save: test-quiet
ifndef NOTE
	$(error NOTE is required. Usage: make bench-save NOTE="description")
endif
	$(PYTHON) benchmarks/bench_regression.py --save --note "$(NOTE)"

bench-quick: maybe-build
	$(PYTHON) benchmarks/bench_regression.py --quick --forward-only

bench-large: test-quiet
ifndef NOTE
	$(error NOTE is required. Usage: make bench-large NOTE="description")
endif
	$(PYTHON) benchmarks/bench_regression.py --large --save --note "$(NOTE)"

bench-forward: maybe-build
	$(PYTHON) benchmarks/bench_regression.py --forward-only

bench-backward: maybe-build
	$(PYTHON) benchmarks/bench_regression.py --backward-only

bench-complexity:
	$(PYTHON) benchmarks/bench_regression.py --quick --forward-only --warmup 1 --iters 1 --complexity

# --- Benchmarks: Scaling & Comparison ---

bench-scaling: maybe-build
	$(PYTHON) benchmarks/bench_cuda_scaling.py

bench-scaling-quick: maybe-build
	$(PYTHON) benchmarks/bench_cuda_scaling.py --n-values 32,64,128,256 --no-backward

bench-compare: maybe-build
	$(PYTHON) benchmarks/bench_cuda_vs_torch.py

bench-compare-quick: maybe-build
	$(PYTHON) benchmarks/bench_cuda_vs_torch.py --n-values 32,64,128,256 --no-backward

# --- Profiling ---

profile-timeline: maybe-build
	@mkdir -p profiling_reports
	nsys profile -o profiling_reports/timeline_$$(date +%Y%m%d_%H%M%S) \
		$(PYTHON) benchmarks/bench_regression.py --quick --forward-only --warmup 1 --iters 1

profile-kernel: maybe-build
	$(PYTHON) benchmarks/profile_kernels.py

# --- History ---

history:
	@$(PYTHON) benchmarks/bench_regression.py --show-history

# --- Combined ---

all: test-full bench-save

iterate: test bench-quick

# --- Clean ---

clean:
	pip uninstall -y att3ntion 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ __pycache__/
	find . -name "*.so" -delete
	@echo "Cleaned build artifacts."

.PHONY: build maybe-build test test-quiet test-full \
        bench bench-save bench-quick bench-large bench-forward bench-backward bench-complexity \
        bench-scaling bench-scaling-quick bench-compare bench-compare-quick \
        profile-timeline profile-kernel history all iterate clean

