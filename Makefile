# Makefile for executorch-spyre — host-only developer convenience targets.
# Requires Python 3.12 and cmake.  No Spyre device needed for the default targets.
#
# Usage:
#   make test          # run all host Python tests (spyre + spyre2)
#   make test-spyre    # run spyre partitioner/preprocess tests only
#   make test-spyre2   # run spyre2 (Triton/KTIR) tests only
#   make test-cpp      # build and run C++ payload-parser gtests
#   make test-all      # test + test-cpp
#   make lint          # pre-commit on all files
#   make install       # editable install into the active venv
#   make clean         # remove cmake build dir and pytest caches

PYTHON   ?= python3.12
VENV     := .venv
PY       := $(VENV)/bin/python
PYTEST   := $(VENV)/bin/python -m pytest

# Shared pytest flags: respect pyproject.toml markers, no .pytest_cache writes
PYTEST_FLAGS := -p no:cacheprovider

# cmake build directory for C++ tests
CPP_BUILD := /tmp/spyre-hosttest-build

# ── virtual-env & install ──────────────────────────────────────────────────────

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --quiet --upgrade pip

.PHONY: install
install: $(VENV)/bin/python
	$(VENV)/bin/pip install --quiet "executorch==1.5.1" "torch==2.13.0" pytest
	$(VENV)/bin/pip install --quiet --no-deps -e .

# ── Python tests ──────────────────────────────────────────────────────────────

.PHONY: test
test: install
	PYTHONPATH="$(CURDIR):$(PYTHONPATH)" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
	  $(PYTEST) $(PYTEST_FLAGS) \
	    executorch/backends/spyre/test/ \
	    executorch/backends/spyre2/test/

.PHONY: test-spyre
test-spyre: install
	PYTHONPATH="$(CURDIR):$(PYTHONPATH)" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
	  $(PYTEST) $(PYTEST_FLAGS) executorch/backends/spyre/test/

.PHONY: test-spyre2
test-spyre2: install
	PYTHONPATH="$(CURDIR):$(PYTHONPATH)" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
	  $(PYTEST) $(PYTEST_FLAGS) executorch/backends/spyre2/test/

# ── C++ tests ─────────────────────────────────────────────────────────────────

.PHONY: test-cpp
test-cpp:
	cmake -S executorch/backends/spyre -B $(CPP_BUILD) \
	  -DEXECUTORCH_BUILD_SPYRE=OFF \
	  -DEXECUTORCH_BUILD_SPYRE_TESTS=ON
	cmake --build $(CPP_BUILD)
	ctest --test-dir $(CPP_BUILD) --output-on-failure

# ── Combined ──────────────────────────────────────────────────────────────────

.PHONY: test-all
test-all: test test-cpp

# ── Lint ──────────────────────────────────────────────────────────────────────

.PHONY: lint
lint:
	$(VENV)/bin/pre-commit run --all-files

# ── Housekeeping ──────────────────────────────────────────────────────────────

.PHONY: clean
clean:
	rm -rf $(CPP_BUILD) .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
