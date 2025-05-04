# Compiler
NVCC := nvcc

# CUTLASS directory
CUTLASS_PATH := /home/centos/cutlass/build/install

# Compilation flags
NVCC_FLAGS := -I$(CUTLASS_PATH)/include -lcuda -gencode=arch=compute_80,code=sm_80 -std=c++17 --expt-relaxed-constexpr
CUTE_FLAGS := -I$(CUTLASS_PATH)/include -lcuda -gencode=arch=compute_80,code=sm_80 -std=c++17 --expt-relaxed-constexpr
CUDA_FLAGS := -lcuda -gencode=arch=compute_80,code=sm_80 -std=c++14 -O3
TK_FLAGS := -O3 -arch=sm_80 -std=c++20 --expt-relaxed-constexpr -I/home/centos/triton-ag/manual/ThunderKittens/include -lcublas

# Source and output files
SOURCES := cutlass_matmul.cu cute_matmul.cu cuda_matmul.cu tk_matmul.cu
TARGETS := cutlass_matmul.out cute_matmul.out cuda_matmul.out tk_matmul.out

# Default target
all: $(TARGETS)

# Specific rules for each implementation
cutlass_matmul.out: cutlass_matmul.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

cute_matmul.out: cute_matmul.cu
	$(NVCC) $(CUTE_FLAGS) -o $@ $<

cuda_matmul.out: cuda_matmul.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $<

tk_matmul.out: tk_matmul.cu
	$(NVCC) $(TK_FLAGS) -o $@ $<

# Run all implementations
run_all: $(TARGETS)
	@echo "======================================"
	@echo "Running CUTLASS MatMul..."
	@echo "======================================"
	./cutlass_matmul.out
	@echo "\n======================================"
	@echo "Running CuTe MatMul..."
	@echo "======================================"
	./cute_matmul.out
	@echo "\n======================================"
	@echo "Running CUDA MatMul..."
	@echo "======================================"
	./cuda_matmul.out
	@echo "\n======================================"
	@echo "Running ThunderKittens MatMul..."
	@echo "======================================"
	./tk_matmul.out

# Clean rule
clean:
	rm -f $(TARGETS)

.PHONY: all clean run_all 