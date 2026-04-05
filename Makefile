# Top-level Makefile for x_gemms

.PHONY: all cpp cuda clean test help

all: cpp cuda

cpp:
	@echo "Building C++ benchmarks..."
	$(MAKE) -C src/cpp

cuda:
	@echo "Building CUDA benchmarks..."
	$(MAKE) -C src/cuda

clean:
	@echo "Cleaning builds..."
	$(MAKE) -C src/cpp clean
	$(MAKE) -C src/cuda clean

test: all
	@echo "Running quick benchmarks..."
	py_env/bin/python -m bench run --quick

test-medium: all
	py_env/bin/python -m bench run --medium

test-large: all
	py_env/bin/python -m bench run --large

help:
	@echo "x_gemms build targets:"
	@echo "  make          - Build all (cpp + cuda)"
	@echo "  make cpp      - Build C++ benchmarks only"
	@echo "  make cuda     - Build CUDA benchmarks only"
	@echo "  make clean    - Clean all builds"
	@echo "  make test     - Build and run quick benchmarks"
	@echo "  make test-medium - Build and run medium benchmarks"
	@echo "  make test-large - Build and run large benchmarks"
