NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_80
CUTLASS_PATH = /home/centos/cutlass/build/install

all: cutlass_matmul.run

cutlass_matmul.run: cutlass_matmul.cu
	$(NVCC) $(NVCC_FLAGS) -I$(CUTLASS_PATH)/include  cutlass_matmul.cu -o cutlass_matmul.run

clean:
	rm -f cutlass_matmul.run 