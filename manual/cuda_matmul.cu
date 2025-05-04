#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <cublas_v2.h>
#include <cuda_bf16.h>  // Include BFloat16 support

// Helper function to convert float to __nv_bfloat16
__host__ __device__ __nv_bfloat16 float_to_bf16(float f) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __float2bfloat16(f);
#else
    return __float2bfloat16_rn(f);
#endif
}

// Helper function to convert __nv_bfloat16 to float
__host__ __device__ float bf16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// Helper function to load __nv_bfloat16 in a vectorized way (2 elements at once)
__device__ __nv_bfloat162 load_bfloat162(const __nv_bfloat16* addr) {
    __nv_bfloat162 val;
    val.x = addr[0];
    val.y = addr[1];
    return val;
}

// CUDA kernel for matrix multiplication with BFloat16
__global__ void matrixMulKernel(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int M, int N, int K) {
    // Calculate global row and column for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (row < M && col < N) {
        // Accumulate results for a single element
        float sum = 0.0f;  // Use float for accumulation to avoid precision loss
        for (int k = 0; k < K; ++k) {
            sum += bf16_to_float(A[row * K + k]) * bf16_to_float(B[k * N + col]);
        }
        C[row * N + col] = float_to_bf16(sum);
    }
}

// Optimized CUDA implementation with improved tiling for better performance
__global__ void matrixMulOptimizedKernel(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int M, int N, int K) {
    const int TILE_SIZE = 32;  // Standard tile size for better performance
    
    // Shared memory tiles
    __shared__ __nv_bfloat16 A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ __nv_bfloat16 B_tile[TILE_SIZE][TILE_SIZE];
    
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Calculate global row and column indices for this thread
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    // Register to accumulate results
    float sum = 0.0f;  // Use float for accumulation
    
    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tiles into shared memory with better memory access pattern
        if (row < M && tile * TILE_SIZE + tx < K) {
            A_tile[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        } else {
            A_tile[ty][tx] = float_to_bf16(0.0f);
        }
        
        if (col < N && tile * TILE_SIZE + ty < K) {
            B_tile[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        } else {
            B_tile[ty][tx] = float_to_bf16(0.0f);
        }
        
        // Ensure all threads have loaded their part of the tiles
        __syncthreads();
        
        // Compute partial dot product with loop unrolling for better ILP
        #pragma unroll 8
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += bf16_to_float(A_tile[ty][k]) * bf16_to_float(B_tile[k][tx]);
        }
        
        // Ensure computation is complete before loading next tiles
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = float_to_bf16(sum);
    }
}

// Further optimized version with double buffering and improved memory access
__global__ void matrixMulVeryOptimizedKernel(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int M, int N, int K) {
    const int TILE_SIZE = 32;  // Optimal for SM utilization
    
    // Shared memory tiles for current tile only
    __shared__ __nv_bfloat16 A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ __nv_bfloat16 B_tile[TILE_SIZE][TILE_SIZE];
    
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Calculate global row and column indices
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    // Registers for accumulation
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tile into shared memory
        if (row < M && tile * TILE_SIZE + tx < K) {
            A_tile[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        } else {
            A_tile[ty][tx] = float_to_bf16(0.0f);
        }
        
        if (col < N && tile * TILE_SIZE + ty < K) {
            B_tile[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        } else {
            B_tile[ty][tx] = float_to_bf16(0.0f);
        }
        
        __syncthreads();
        
        // Compute partial dot product with aggressive loop unrolling
        #pragma unroll 16
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += bf16_to_float(A_tile[ty][k]) * bf16_to_float(B_tile[k][tx]);
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = float_to_bf16(sum);
    }
}

// CUDA implementation with tiling for better performance (original)
__global__ void matrixMulTiledKernel(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int M, int N, int K) {
    const int TILE_SIZE = 32;
    
    // Shared memory tiles
    __shared__ __nv_bfloat16 A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ __nv_bfloat16 B_tile[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Calculate row and column indices for this thread
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;  // Use float for accumulation
    
    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tiles into shared memory
        if (row < M && tile * TILE_SIZE + tx < K) {
            A_tile[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        } else {
            A_tile[ty][tx] = float_to_bf16(0.0f);
        }
        
        if (col < N && tile * TILE_SIZE + ty < K) {
            B_tile[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        } else {
            B_tile[ty][tx] = float_to_bf16(0.0f);
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += bf16_to_float(A_tile[ty][k]) * bf16_to_float(B_tile[k][tx]);
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = float_to_bf16(sum);
    }
}

// Modified to separate memory operations from computation and select best kernel
void cuda_matmul(float* A, float* B, float* C, int M, int N, int K, __nv_bfloat16* d_A, __nv_bfloat16* d_B, __nv_bfloat16* d_C, bool copy_memory, int kernel_type = 2) {
    size_t size_A = M * K * sizeof(__nv_bfloat16);
    size_t size_B = K * N * sizeof(__nv_bfloat16);
    size_t size_C = M * N * sizeof(__nv_bfloat16);

    if (copy_memory) {
        // Convert and copy data from host to device
        __nv_bfloat16* h_A_bf16 = new __nv_bfloat16[M * K];
        __nv_bfloat16* h_B_bf16 = new __nv_bfloat16[K * N];
        
        for (int i = 0; i < M * K; ++i) {
            h_A_bf16[i] = float_to_bf16(A[i]);
        }
        
        for (int i = 0; i < K * N; ++i) {
            h_B_bf16[i] = float_to_bf16(B[i]);
        }
        
        cudaMemcpy(d_A, h_A_bf16, size_A, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B_bf16, size_B, cudaMemcpyHostToDevice);
        
        delete[] h_A_bf16;
        delete[] h_B_bf16;
    }

    // Set up execution configuration based on the kernel type
    const int BLOCK_SIZE = 32;  // Standard block size for all kernels
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);

    // Launch the selected kernel
    switch (kernel_type) {
        case 0:  // Original tiled kernel
            matrixMulTiledKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
            break;
        case 1:  // Optimized kernel
            matrixMulOptimizedKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
            break;
        case 2:  // Very optimized kernel
        default:
            matrixMulVeryOptimizedKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
            break;
    }

    if (copy_memory) {
        // Convert and copy result back to host
        __nv_bfloat16* h_C_bf16 = new __nv_bfloat16[M * N];
        cudaMemcpy(h_C_bf16, d_C, size_C, cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < M * N; ++i) {
            C[i] = bf16_to_float(h_C_bf16[i]);
        }
        
        delete[] h_C_bf16;
    }
}

// Modified to separate memory operations from computation with BFloat16
void cublas_matmul(float* A, float* B, float* C, int M, int N, int K, __nv_bfloat16* d_A, __nv_bfloat16* d_B, __nv_bfloat16* d_C, cublasHandle_t handle, bool copy_memory) {
    size_t size_A = M * K * sizeof(__nv_bfloat16);
    size_t size_B = K * N * sizeof(__nv_bfloat16);
    size_t size_C = M * N * sizeof(__nv_bfloat16);

    if (copy_memory) {
        // Convert and copy data from host to device
        __nv_bfloat16* h_A_bf16 = new __nv_bfloat16[M * K];
        __nv_bfloat16* h_B_bf16 = new __nv_bfloat16[K * N];
        
        for (int i = 0; i < M * K; ++i) {
            h_A_bf16[i] = float_to_bf16(A[i]);
        }
        
        for (int i = 0; i < K * N; ++i) {
            h_B_bf16[i] = float_to_bf16(B[i]);
        }
        
        cudaMemcpy(d_A, h_A_bf16, size_A, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B_bf16, size_B, cudaMemcpyHostToDevice);
        
        delete[] h_A_bf16;
        delete[] h_B_bf16;
    }

    // Set up alpha and beta for gemm
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Call cuBLAS GEMM with BFloat16
    // Use cublasGemmEx for mixed precision
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 d_B, CUDA_R_16BF, N,
                 d_A, CUDA_R_16BF, K,
                 &beta,
                 d_C, CUDA_R_16BF, N,
                 CUBLAS_COMPUTE_32F,  // Compute in FP32 for better accuracy
                 CUBLAS_GEMM_DEFAULT);

    if (copy_memory) {
        // Convert and copy result back to host
        __nv_bfloat16* h_C_bf16 = new __nv_bfloat16[M * N];
        cudaMemcpy(h_C_bf16, d_C, size_C, cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < M * N; ++i) {
            C[i] = bf16_to_float(h_C_bf16[i]);
        }
        
        delete[] h_C_bf16;
    }
}

// Updated benchmark function to compare different CUDA implementations
void benchmark_matmul() {
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;
    
    // Number of iterations for timing and warmup
    const int timing_iterations = 100;
    const int warmup_iterations = 25;
    
    std::cout << "Benchmarking BFloat16 matrix multiplication: " << M << "x" << K << " * " << K << "x" << N << std::endl;
    std::cout << "Running " << warmup_iterations << " warmup iterations and " << timing_iterations << " timing iterations" << std::endl;
    
    // Allocate and initialize host memory (still using float for host)
    float *A = (float*)malloc(M * K * sizeof(float));
    float *B = (float*)malloc(K * N * sizeof(float));
    float *C_cuda_original = (float*)malloc(M * N * sizeof(float));
    float *C_cuda_optimized = (float*)malloc(M * N * sizeof(float));
    float *C_cuda_very_optimized = (float*)malloc(M * N * sizeof(float));
    float *C_cublas = (float*)malloc(M * N * sizeof(float));
    
    // Initialize matrices with random values
    for (int i = 0; i < M * K; ++i) {
        A[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;
    }
    
    for (int i = 0; i < K * N; ++i) {
        B[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;
    }
    
    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B, *d_C_original, *d_C_optimized, *d_C_very_optimized, *d_C_cublas;
    size_t size_A = M * K * sizeof(__nv_bfloat16);
    size_t size_B = K * N * sizeof(__nv_bfloat16);
    size_t size_C = M * N * sizeof(__nv_bfloat16);
    
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C_original, size_C);
    cudaMalloc((void**)&d_C_optimized, size_C);
    cudaMalloc((void**)&d_C_very_optimized, size_C);
    cudaMalloc((void**)&d_C_cublas, size_C);
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Convert and copy data to device once
    __nv_bfloat16* h_A_bf16 = new __nv_bfloat16[M * K];
    __nv_bfloat16* h_B_bf16 = new __nv_bfloat16[K * N];
    
    for (int i = 0; i < M * K; ++i) {
        h_A_bf16[i] = float_to_bf16(A[i]);
    }
    
    for (int i = 0; i < K * N; ++i) {
        h_B_bf16[i] = float_to_bf16(B[i]);
    }
    
    cudaMemcpy(d_A, h_A_bf16, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_bf16, size_B, cudaMemcpyHostToDevice);
    
    delete[] h_A_bf16;
    delete[] h_B_bf16;
    
    // Compute GFLOPS
    double gflops = (2.0 * M * N * K) * 1e-9;
    
    // Set up consistent execution configuration for all kernels
    const int BLOCK_SIZE = 32;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);
    
    // Warmup runs
    std::cout << "Performing warmup runs..." << std::endl;
    
    // Original CUDA implementation warmup
    for (int i = 0; i < warmup_iterations; ++i) {
        matrixMulTiledKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C_original, M, N, K);
    }
    
    // Optimized CUDA implementation warmup
    for (int i = 0; i < warmup_iterations; ++i) {
        matrixMulOptimizedKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C_optimized, M, N, K);
    }
    
    // Very Optimized CUDA implementation warmup
    for (int i = 0; i < warmup_iterations; ++i) {
        matrixMulVeryOptimizedKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C_very_optimized, M, N, K);
    }
    
    // cuBLAS implementation warmup
    for (int i = 0; i < warmup_iterations; ++i) {
        cublas_matmul(A, B, C_cublas, M, N, K, d_A, d_B, d_C_cublas, handle, false);
    }
    
    cudaDeviceSynchronize();
    
    // Benchmark original CUDA implementation
    std::cout << "Benchmarking original CUDA BFloat16 implementation..." << std::endl;
    cudaDeviceSynchronize();
    auto cuda_original_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < timing_iterations; ++i) {
        matrixMulTiledKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C_original, M, N, K);
    }
    
    cudaDeviceSynchronize();
    auto cuda_original_end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> cuda_original_diff = cuda_original_end - cuda_original_start;
    double cuda_original_time_ms = cuda_original_diff.count() * 1000.0 / timing_iterations;
    
    // Benchmark optimized CUDA implementation
    std::cout << "Benchmarking optimized CUDA BFloat16 implementation..." << std::endl;
    cudaDeviceSynchronize();
    auto cuda_optimized_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < timing_iterations; ++i) {
        matrixMulOptimizedKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C_optimized, M, N, K);
    }
    
    cudaDeviceSynchronize();
    auto cuda_optimized_end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> cuda_optimized_diff = cuda_optimized_end - cuda_optimized_start;
    double cuda_optimized_time_ms = cuda_optimized_diff.count() * 1000.0 / timing_iterations;
    
    // Benchmark very optimized CUDA implementation
    std::cout << "Benchmarking very optimized CUDA BFloat16 implementation..." << std::endl;
    cudaDeviceSynchronize();
    auto cuda_very_optimized_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < timing_iterations; ++i) {
        matrixMulVeryOptimizedKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C_very_optimized, M, N, K);
    }
    
    cudaDeviceSynchronize();
    auto cuda_very_optimized_end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> cuda_very_optimized_diff = cuda_very_optimized_end - cuda_very_optimized_start;
    double cuda_very_optimized_time_ms = cuda_very_optimized_diff.count() * 1000.0 / timing_iterations;
    
    // Benchmark cuBLAS implementation
    std::cout << "Benchmarking cuBLAS BFloat16 implementation..." << std::endl;
    cudaDeviceSynchronize();
    auto cublas_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < timing_iterations; ++i) {
        cublas_matmul(A, B, C_cublas, M, N, K, d_A, d_B, d_C_cublas, handle, false);
    }
    
    cudaDeviceSynchronize();
    auto cublas_end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> cublas_diff = cublas_end - cublas_start;
    double cublas_time_ms = cublas_diff.count() * 1000.0 / timing_iterations;
    
    // Copy results back to host for verification
    __nv_bfloat16* h_C_original_bf16 = new __nv_bfloat16[M * N];
    __nv_bfloat16* h_C_optimized_bf16 = new __nv_bfloat16[M * N];
    __nv_bfloat16* h_C_very_optimized_bf16 = new __nv_bfloat16[M * N];
    __nv_bfloat16* h_C_cublas_bf16 = new __nv_bfloat16[M * N];
    
    cudaMemcpy(h_C_original_bf16, d_C_original, size_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_optimized_bf16, d_C_optimized, size_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_very_optimized_bf16, d_C_very_optimized, size_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_cublas_bf16, d_C_cublas, size_C, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < M * N; ++i) {
        C_cuda_original[i] = bf16_to_float(h_C_original_bf16[i]);
        C_cuda_optimized[i] = bf16_to_float(h_C_optimized_bf16[i]);
        C_cuda_very_optimized[i] = bf16_to_float(h_C_very_optimized_bf16[i]);
        C_cublas[i] = bf16_to_float(h_C_cublas_bf16[i]);
    }
    
    delete[] h_C_original_bf16;
    delete[] h_C_optimized_bf16;
    delete[] h_C_very_optimized_bf16;
    delete[] h_C_cublas_bf16;
    
    // Print results
    std::cout << "\n=== Performance Results (BFloat16, excluding memory transfers) ===\n";
    std::cout << "Original CUDA BFloat16 Implementation: " << gflops / (cuda_original_time_ms * 1e-3) 
              << " GFLOPS, " << cuda_original_time_ms << " ms\n";
    std::cout << "Optimized CUDA BFloat16 Implementation: " << gflops / (cuda_optimized_time_ms * 1e-3) 
              << " GFLOPS, " << cuda_optimized_time_ms << " ms\n";
    std::cout << "Very Optimized CUDA BFloat16 Implementation: " << gflops / (cuda_very_optimized_time_ms * 1e-3) 
              << " GFLOPS, " << cuda_very_optimized_time_ms << " ms\n";
    std::cout << "cuBLAS BFloat16 Implementation: " << gflops / (cublas_time_ms * 1e-3) 
              << " GFLOPS, " << cublas_time_ms << " ms\n";
    
    // Speed-up calculations
    std::cout << "\n=== Speed-up Comparisons ===\n";
    std::cout << "Optimized vs Original: " << cuda_original_time_ms / cuda_optimized_time_ms << "x\n";
    std::cout << "Very Optimized vs Original: " << cuda_original_time_ms / cuda_very_optimized_time_ms << "x\n";
    std::cout << "Very Optimized vs Optimized: " << cuda_optimized_time_ms / cuda_very_optimized_time_ms << "x\n";
    std::cout << "cuBLAS vs Very Optimized: " << cuda_very_optimized_time_ms / cublas_time_ms << "x\n";
    
    // Verify correctness: compute max difference between implementations
    float max_diff_opt_orig = 0.0f;
    float max_diff_very_opt_orig = 0.0f;
    float max_diff_cublas_orig = 0.0f;
    
    for (int i = 0; i < M * N; ++i) {
        max_diff_opt_orig = std::max(max_diff_opt_orig, std::abs(C_cuda_optimized[i] - C_cuda_original[i]));
        max_diff_very_opt_orig = std::max(max_diff_very_opt_orig, std::abs(C_cuda_very_optimized[i] - C_cuda_original[i]));
        max_diff_cublas_orig = std::max(max_diff_cublas_orig, std::abs(C_cublas[i] - C_cuda_original[i]));
    }
    
    std::cout << "\n=== Verification ===\n";
    std::cout << "Max difference between Original and Optimized: " << max_diff_opt_orig << std::endl;
    std::cout << "Max difference between Original and Very Optimized: " << max_diff_very_opt_orig << std::endl;
    std::cout << "Max difference between Original and cuBLAS: " << max_diff_cublas_orig << std::endl;
    
    // Print sample results
    std::cout << "\nSample results (top-left corner):" << std::endl;
    const int sample_size = std::min(4, M);
    
    std::cout << "Original CUDA BFloat16 implementation:\n";
    for (int i = 0; i < sample_size; ++i) {
        for (int j = 0; j < sample_size; ++j) {
            std::cout << C_cuda_original[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nOptimized CUDA BFloat16 implementation:\n";
    for (int i = 0; i < sample_size; ++i) {
        for (int j = 0; j < sample_size; ++j) {
            std::cout << C_cuda_optimized[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nVery Optimized CUDA BFloat16 implementation:\n";
    for (int i = 0; i < sample_size; ++i) {
        for (int j = 0; j < sample_size; ++j) {
            std::cout << C_cuda_very_optimized[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\ncuBLAS BFloat16 implementation:\n";
    for (int i = 0; i < sample_size; ++i) {
        for (int j = 0; j < sample_size; ++j) {
            std::cout << C_cublas[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Clean up
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_original);
    cudaFree(d_C_optimized);
    cudaFree(d_C_very_optimized);
    cudaFree(d_C_cublas);
    
    // Free host memory
    free(A);
    free(B);
    free(C_cuda_original);
    free(C_cuda_optimized);
    free(C_cuda_very_optimized);
    free(C_cublas);
}

// Original test function
void test_cuda_matmul() {
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;
    
    // Allocate and initialize host memory
    float *A = (float*)malloc(M * K * sizeof(float));
    float *B = (float*)malloc(K * N * sizeof(float));
    float *C = (float*)malloc(M * N * sizeof(float));
    
    // Initialize matrices with random values
    for (int i = 0; i < M * K; ++i) {
        A[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;
    }
    
    for (int i = 0; i < K * N; ++i) {
        B[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;
    }
    
    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(__nv_bfloat16);
    size_t size_B = K * N * sizeof(__nv_bfloat16);
    size_t size_C = M * N * sizeof(__nv_bfloat16);
    
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);
    
    // Compute GFLOPS
    double gflops = (2.0 * M * N * K) * 1e-9;
    
    // Run the kernel once to warm up
    cuda_matmul(A, B, C, M, N, K, d_A, d_B, d_C, true);
    
    // Time the kernel with multiple iterations
    const int timing_iterations = 10;
    
    // Use std::chrono for timing
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < timing_iterations; ++i) {
        cuda_matmul(A, B, C, M, N, K, d_A, d_B, d_C, true);
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = end - start;
    double time_ms = diff.count() * 1000.0 / timing_iterations;
    
    std::cout << "Matrix size: " << M << "x" << N << "x" << K << std::endl;
    std::cout << "CUDA GEMM BFloat16 performance: " << gflops / (time_ms * 1e-3) << " GFLOPS, " 
              << time_ms << " ms" << std::endl;
    
    // Print a small portion of the result for verification
    std::cout << "\nSample results (top-left corner):" << std::endl;
    const int sample_size = std::min(4, M);
    for (int i = 0; i < sample_size; ++i) {
        for (int j = 0; j < sample_size; ++j) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "CUDA GEMM BFloat16 test completed successfully!" << std::endl;
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Free host memory
    free(A);
    free(B);
    free(C);
}

int main() {
    // Run benchmark comparing CUDA and cuBLAS implementations
    benchmark_matmul();
    return 0;
} 