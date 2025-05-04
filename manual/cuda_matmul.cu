#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

// CUDA kernel for matrix multiplication
__global__ void matrixMulKernel(float* A, float* B, float* C, int M, int N, int K) {
    // Calculate global row and column for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (row < M && col < N) {
        // Accumulate results for a single element
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// CUDA implementation with tiling for better performance
__global__ void matrixMulTiledKernel(float* A, float* B, float* C, int M, int N, int K) {
    const int TILE_SIZE = 32;
    
    // Shared memory tiles
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Calculate row and column indices for this thread
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tiles into shared memory
        if (row < M && tile * TILE_SIZE + tx < K) {
            A_tile[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        } else {
            A_tile[ty][tx] = 0.0f;
        }
        
        if (col < N && tile * TILE_SIZE + ty < K) {
            B_tile[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        } else {
            B_tile[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += A_tile[ty][k] * B_tile[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Function to perform matrix multiplication using CUDA
void cuda_matmul(float* A, float* B, float* C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Allocate device memory
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    // Copy data from host to device
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    // Set up execution configuration
    const int BLOCK_SIZE = 32;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);

    // Launch the tiled kernel
    matrixMulTiledKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);

    // Copy result back to host
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Test function
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
    
    // Compute GFLOPS
    double gflops = (2.0 * M * N * K) * 1e-9;
    
    // Run the kernel once to warm up
    cuda_matmul(A, B, C, M, N, K);
    
    // Time the kernel with multiple iterations
    const int timing_iterations = 10;
    
    // Use std::chrono for timing
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < timing_iterations; ++i) {
        cuda_matmul(A, B, C, M, N, K);
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = end - start;
    double time_ms = diff.count() * 1000.0 / timing_iterations;
    
    std::cout << "Matrix size: " << M << "x" << N << "x" << K << std::endl;
    std::cout << "CUDA GEMM performance: " << gflops / (time_ms * 1e-3) << " GFLOPS, " 
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
    
    std::cout << "CUDA GEMM test completed successfully!" << std::endl;
    
    // Free host memory
    free(A);
    free(B);
    free(C);
}

int main() {
    test_cuda_matmul();
    return 0;
} 