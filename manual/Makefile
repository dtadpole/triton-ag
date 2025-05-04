NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80 -std=c++20 --expt-relaxed-constexpr -I/home/centos/triton-ag/manual/ThunderKittens/include
CUTLASS_INCLUDE = -I/home/centos/triton-ag/cutlass/include -I/home/centos/triton-ag/cutlass/tools/util/include
CUTE_FLAGS = -std=c++17 --expt-relaxed-constexpr
CUBLAS_LIBS = -lcublas
TARGETS = thunderkitten_matmul.out cutlass_matmul.out cute_matmul.out cuda_matmul.out

all: $(TARGETS)

thunderkitten_matmul.out: thunderkitten_matmul.cu
	$(NVCC) $(NVCC_FLAGS) $(CUBLAS_LIBS) -o $@ $<

cutlass_matmul.out: cutlass_matmul.cu
	$(NVCC) $(CUTLASS_INCLUDE) --expt-relaxed-constexpr -o $@ $<

cute_matmul.out: cute_matmul.cu
	$(NVCC) $(CUTLASS_INCLUDE) $(CUTE_FLAGS) -o $@ $<

cuda_matmul.out: cuda_matmul.cu
	$(NVCC) -O3 -std=c++14 -o $@ $<

run_comparison: thunderkitten_matmul.out cute_matmul.out cuda_matmul.out
	@echo "======================================"
	@echo "Running ThunderKittens MatMul..."
	@echo "======================================"
	./thunderkitten_matmul.out
	@echo "\n======================================"
	@echo "Running CuTe MatMul..."
	@echo "======================================"
	./cute_matmul.out
	@echo "\n======================================"
	@echo "Running CUDA MatMul..."
	@echo "======================================"
	./cuda_matmul.out
	@echo "\n======================================"
	@echo "Performance Comparison"
	@echo "======================================"
	@echo "All implementations demonstrate matrix multiplication with different approaches:"
	@echo "ThunderKittens: Optimized for NVIDIA hardware with cuBLAS integration."
	@echo "CuTe: Modern tensor abstraction library with clean programming model."
	@echo "CUDA: Direct CUDA implementation with tiling optimization."

clean:
	rm -f $(TARGETS)

.PHONY: all clean run_all run_comparison 