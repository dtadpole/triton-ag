#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <iostream>
#include <chrono>
#include <iomanip> // For prettier debug output

#include <cuda_runtime.h>
#include <cuda_bf16.h> // Include bfloat16 headers
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cublas_v2.h>

// Include CuTe headers
#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/numeric_conversion.h" 

// Helper to convert between float and __nv_bfloat16
float bf16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

__nv_bfloat16 float_to_bf16(float f) {
    return __float2bfloat16(f);
}

// Helper kernel to convert float to bfloat16
__global__ void convertFloatToBFloat16(__nv_bfloat16* out, float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(in[idx]);
    }
}

// Helper kernel to convert bfloat16 to float
__global__ void convertBFloat16ToFloat(float* out, __nv_bfloat16* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __bfloat162float(in[idx]);
    }
}

// Debug function to validate conversion round trips
void validate_bf16_conversion() {
    const int n = 10;
    float test_values[n] = {
        1.0f, -1.0f, 0.1234f, -0.1234f, 123.456f, 
        -123.456f, 0.0f, 1e-3f, 1e3f, -1e3f
    };
    
    std::cout << "\n==== BF16 Conversion Test ====" << std::endl;
    std::cout << std::fixed << std::setprecision(7);
    std::cout << "Value\t\tFloat→BF16→Float\tDifference" << std::endl;
    
    for (int i = 0; i < n; i++) {
        float original = test_values[i];
        __nv_bfloat16 as_bf16 = float_to_bf16(original);
        float roundtrip = bf16_to_float(as_bf16);
        float diff = std::abs(original - roundtrip);
        
        std::cout << original << "\t" << roundtrip << "\t" << diff << std::endl;
    }
}

// Reference CPU matrix multiplication for validation
void cpu_matmul(int m, int n, int k, 
                float alpha,
                float* A, int ldA,
                float* B, int ldB, 
                float beta,
                float* C, int ldC) {
    // Matrix multiplication C = alpha * A * B + beta * C
    // A is m x k, B is k x n, C is m x n
    // All matrices in row-major format
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A[i * ldA + p] * B[p * ldB + j];  // Note: B is accessed as B[p, j]
            }
            C[i * ldC + j] = alpha * sum + beta * C[i * ldC + j];
        }
    }
}

// CUDA kernel for matrix transpose (B from row-major to column-major)
__global__ void transposeMatrix(float* input, float* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        // Input is row-major: input[row*cols + col]
        // Output is column-major: output[col*rows + row]
        output[col*rows + row] = input[row*cols + col];
    }
}

// CuTe matrix multiplication kernel
template <class ProblemShape, class TileShape>
__global__ void cute_matmul_kernel(
    ProblemShape problem_shape,
    TileShape tile_shape,
    float const* A,     // M x K in row-major layout
    float const* B,     // K x N in column-major layout (already transposed)
    float* C,           // M x N in row-major layout
    float alpha, float beta)
{
    using namespace cute;
    
    // Extract dimensions
    int M = get<0>(problem_shape);
    int N = get<1>(problem_shape);
    int K = get<2>(problem_shape);
    
    // Get tile sizes
    int BM = get<0>(tile_shape);
    int BN = get<1>(tile_shape);
    int BK = get<2>(tile_shape);
    
    // This kernel uses block tiling and shared memory
    extern __shared__ float shared_mem[];
    
    // Partition shared memory
    float* sA = shared_mem;
    float* sB = shared_mem + BM * BK;
    
    // Calculate global position
    int m_block = blockIdx.y;
    int n_block = blockIdx.x;
    int m_thread = threadIdx.y;
    int n_thread = threadIdx.x;
    
    // Calculate global indices
    int m_start = m_block * BM;
    int n_start = n_block * BN;
    
    // Register for accumulation
    float acc[16][16] = {0.0f};
    
    // Loop over K tiles
    for (int k_offset = 0; k_offset < K; k_offset += BK) {
        // Load A tile into shared memory (row-major)
        for (int i = threadIdx.y; i < BM; i += blockDim.y) {
            for (int j = threadIdx.x; j < BK; j += blockDim.x) {
                int m_global = m_start + i;
                int k_global = k_offset + j;
                
                if (m_global < M && k_global < K) {
                    sA[i * BK + j] = A[m_global * K + k_global];
                } else {
                    sA[i * BK + j] = 0.0f;
                }
            }
        }
        
        // Load B tile into shared memory (column-major from transposed B)
        for (int i = threadIdx.y; i < BK; i += blockDim.y) {
            for (int j = threadIdx.x; j < BN; j += blockDim.x) {
                int k_global = k_offset + i;
                int n_global = n_start + j;
                
                if (k_global < K && n_global < N) {
                    // B is already transposed (column-major), so access pattern is different
                    sB[i * BN + j] = B[n_global * K + k_global];
                } else {
                    sB[i * BN + j] = 0.0f;
                }
            }
        }
        
        __syncthreads();
        
        // Each thread computes a part of the output tile
        #pragma unroll 4
        for (int k = 0; k < BK; k++) {
            // Each thread in the block is responsible for computing elements 
            // of the output matrix based on its 2D thread index
            float aValue = sA[m_thread * BK + k];
            float bValue = sB[k * BN + n_thread];
            acc[0][0] += aValue * bValue;
        }
        
        __syncthreads();
    }
    
    // Store result back to global memory
    int m_global = m_start + m_thread;
    int n_global = n_start + n_thread;
    
    if (m_global < M && n_global < N) {
        // Apply alpha and beta
        C[m_global * N + n_global] = alpha * acc[0][0] + beta * C[m_global * N + n_global];
    }
}

// Setup and run matrix multiplication using CuTe with bfloat16
template <class TA, class TB, class TC, class Alpha, class Beta>
void cute_matmul(int m, int n, int k,
                 Alpha alpha,
                 TA const* A_bf16, int ldA,
                 TB const* B_bf16, int ldB,
                 Beta beta,
                 TC* C_bf16, int ldC,
                 cudaStream_t stream = 0)
{
  using namespace cute;

  // Convert BF16 inputs to float for processing
  float* A_float;
  float* B_float;
  float* C_float;
  
  size_t size_a = m * k * sizeof(float);
  size_t size_b = k * n * sizeof(float);
  size_t size_c = m * n * sizeof(float);
  
  cudaMalloc(&A_float, size_a);
  cudaMalloc(&B_float, size_b);
  cudaMalloc(&C_float, size_c);
  
  // Convert BF16 inputs to FP32
  int blockSize = 256;
  int gridSize_a = (m * k + blockSize - 1) / blockSize;
  int gridSize_b = (k * n + blockSize - 1) / blockSize;
  int gridSize_c = (m * n + blockSize - 1) / blockSize;
  
  convertBFloat16ToFloat<<<gridSize_a, blockSize, 0, stream>>>(A_float, (__nv_bfloat16*)A_bf16, m * k);
  convertBFloat16ToFloat<<<gridSize_b, blockSize, 0, stream>>>(B_float, (__nv_bfloat16*)B_bf16, k * n);
  
  // Create a temporary matrix B that's properly transposed for our CuTe kernel
  float* B_transposed;
  cudaMalloc(&B_transposed, size_b);
  
  // Transpose B on GPU (row-major to column-major)
  dim3 blockDim(16, 16);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (k + blockDim.y - 1) / blockDim.y);
  transposeMatrix<<<gridDim, blockDim, 0, stream>>>(B_float, B_transposed, k, n);
  cudaDeviceSynchronize();
  
  // Define problem shape and tile shape using CuTe's make_shape
  auto problem_shape = make_shape(m, n, k);
  auto tile_shape = make_shape(32, 32, 8); // Tune these tile sizes
  
  // Calculate thread block size and grid size
  dim3 threadsPerBlock(32, 32);
  dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
  
  // Calculate shared memory size
  size_t smem_size = (32*8 + 8*32) * sizeof(float); // For A and B tiles
  
  // Launch the CuTe kernel
  float alpha_f = alpha;
  float beta_f = beta;
  
  cute_matmul_kernel<<<blocksPerGrid, threadsPerBlock, smem_size, stream>>>(
      problem_shape, tile_shape,
      A_float,        // A: M x K in row-major
      B_transposed,   // B: K x N in column-major (transposed)
      C_float,        // C: M x N in row-major
      alpha_f, beta_f);
  
  // Convert results back to BF16
  convertFloatToBFloat16<<<gridSize_c, blockSize, 0, stream>>>((__nv_bfloat16*)C_bf16, C_float, m * n);
  
  // Free temporary memory
  cudaFree(A_float);
  cudaFree(B_float);
  cudaFree(B_transposed);
  cudaFree(C_float);
}

// Setup and run matrix multiplication using cuBLAS
template <class TA, class TB, class TC, class Alpha, class Beta>
void cublas_matmul(int m, int n, int k,
                  Alpha alpha,
                  TA const* A, int ldA,
                  TB const* B, int ldB,
                  Beta beta,
                  TC* C, int ldC,
                  cudaStream_t stream = 0)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    if (stream != 0) {
        cublasSetStream(handle, stream);
    }
    
    // Using cublasGemmEx for bfloat16 support
    float alpha_f = alpha;
    float beta_f = beta;
    
    // cuBLAS assumes column-major layout but we're using row-major
    // To compute C = A*B in row-major, we can compute C' = B'*A' in column-major
    // This is equivalent to computing B*A with OP_N in column-major order
    // where B has dimensions (k,n) and A has dimensions (m,k)
    cublasGemmEx(handle,
                 CUBLAS_OP_N, CUBLAS_OP_N,   // No additional transpose needed
                 n, m, k,                    // Dimensions: n=cols(C), m=rows(C), k=common dimension
                 &alpha_f,
                 B, CUDA_R_16BF, ldB,        // B first (k,n) with leading dimension ldB
                 A, CUDA_R_16BF, ldA,        // A second (m,k) with leading dimension ldA
                 &beta_f,
                 C, CUDA_R_16BF, ldC,        // C output (m,n) with leading dimension ldC
                 CUDA_R_32F,                 // Compute in FP32
                 CUBLAS_GEMM_DEFAULT);
                 
    cublasDestroy(handle);
}

// Debug test with a small matrix
void debug_small_matmul() {
    const int M = 4;  // Use a small size for easier debugging
    const int N = 4;
    const int K = 4;
    
    std::cout << "\n==== Small Matrix Debug Test ====" << std::endl;
    
    // Create test matrices with known values (row-major)
    float h_A_float[M*K] = {
        1.0f, 2.0f, 3.0f, 4.0f,     // A[0,0] to A[0,3]
        5.0f, 6.0f, 7.0f, 8.0f,     // A[1,0] to A[1,3]
        9.0f, 10.0f, 11.0f, 12.0f,  // A[2,0] to A[2,3]
        13.0f, 14.0f, 15.0f, 16.0f  // A[3,0] to A[3,3]
    };
    
    float h_B_float[K*N] = {
        1.0f, 2.0f, 3.0f, 4.0f,     // B[0,0] to B[0,3]
        5.0f, 6.0f, 7.0f, 8.0f,     // B[1,0] to B[1,3]
        9.0f, 10.0f, 11.0f, 12.0f,  // B[2,0] to B[2,3]
        13.0f, 14.0f, 15.0f, 16.0f  // B[3,0] to B[3,3]
    };
    
    // Do a reference CPU calculation
    float h_C_ref_float[M*N] = {0.0f};
    cpu_matmul(M, N, K, 1.0f, h_A_float, K, h_B_float, N, 0.0f, h_C_ref_float, N);
    
    // Print the expected result
    std::cout << "\nExpected CPU Result (row-major):" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << std::setw(8) << h_C_ref_float[i*N + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Convert to bfloat16
    __nv_bfloat16 h_A_bf16[M*K];
    __nv_bfloat16 h_B_bf16[K*N];
    __nv_bfloat16 h_C_cute_bf16[M*N];
    __nv_bfloat16 h_C_cublas_bf16[M*N];
    
    for (int i = 0; i < M*K; i++) {
        h_A_bf16[i] = float_to_bf16(h_A_float[i]);
    }
    
    for (int i = 0; i < K*N; i++) {
        h_B_bf16[i] = float_to_bf16(h_B_float[i]);
    }
    
    for (int i = 0; i < M*N; i++) {
        h_C_cute_bf16[i] = float_to_bf16(0.0f);
        h_C_cublas_bf16[i] = float_to_bf16(0.0f);
    }
    
    // Allocate device memory
    __nv_bfloat16 *d_A_bf16, *d_B_bf16, *d_C_cute_bf16, *d_C_cublas_bf16;
    cudaMalloc(&d_A_bf16, M*K*sizeof(__nv_bfloat16));
    cudaMalloc(&d_B_bf16, K*N*sizeof(__nv_bfloat16));
    cudaMalloc(&d_C_cute_bf16, M*N*sizeof(__nv_bfloat16));
    cudaMalloc(&d_C_cublas_bf16, M*N*sizeof(__nv_bfloat16));
    
    // Copy to device
    cudaMemcpy(d_A_bf16, h_A_bf16, M*K*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_bf16, h_B_bf16, K*N*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_cute_bf16, h_C_cute_bf16, M*N*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_cublas_bf16, h_C_cublas_bf16, M*N*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    
    // Run CuTe implementation
    cute_matmul(M, N, K, 1.0f, d_A_bf16, K, d_B_bf16, N, 0.0f, d_C_cute_bf16, N);
    
    // Run cuBLAS implementation
    cublas_matmul(M, N, K, 1.0f, d_A_bf16, K, d_B_bf16, N, 0.0f, d_C_cublas_bf16, N);
    
    // Copy results back to host
    cudaMemcpy(h_C_cute_bf16, d_C_cute_bf16, M*N*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_cublas_bf16, d_C_cublas_bf16, M*N*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    
    // Convert results to float for comparison
    float h_C_cute_float[M*N];
    float h_C_cublas_float[M*N];
    
    for (int i = 0; i < M*N; i++) {
        h_C_cute_float[i] = bf16_to_float(h_C_cute_bf16[i]);
        h_C_cublas_float[i] = bf16_to_float(h_C_cublas_bf16[i]);
    }
    
    // Print matrices with proper indexing for clarity
    std::cout << "\nOriginal A Matrix (row-major):" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            std::cout << std::setw(8) << h_A_float[i*K + j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nOriginal B Matrix (row-major):" << std::endl;
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << std::setw(8) << h_B_float[i*N + j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nCuTe BF16 Result (row-major):" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << std::setw(8) << h_C_cute_float[i*N + j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\ncuBLAS BF16 Result (row-major):" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << std::setw(8) << h_C_cublas_float[i*N + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Check differences
    float max_diff_cpu_cute = 0.0f;
    float max_diff_cpu_cublas = 0.0f;
    float max_diff_cute_cublas = 0.0f;
    
    for (int i = 0; i < M*N; i++) {
        max_diff_cpu_cute = std::max(max_diff_cpu_cute, std::abs(h_C_ref_float[i] - h_C_cute_float[i]));
        max_diff_cpu_cublas = std::max(max_diff_cpu_cublas, std::abs(h_C_ref_float[i] - h_C_cublas_float[i]));
        max_diff_cute_cublas = std::max(max_diff_cute_cublas, std::abs(h_C_cute_float[i] - h_C_cublas_float[i]));
    }
    
    std::cout << "\nMax Differences:" << std::endl;
    std::cout << "CPU vs. CuTe: " << max_diff_cpu_cute << std::endl;
    std::cout << "CPU vs. cuBLAS: " << max_diff_cpu_cublas << std::endl;
    std::cout << "CuTe vs. cuBLAS: " << max_diff_cute_cublas << std::endl;
    
    // Free device memory
    cudaFree(d_A_bf16);
    cudaFree(d_B_bf16);
    cudaFree(d_C_cute_bf16);
    cudaFree(d_C_cublas_bf16);
}

// Benchmark function to compare CuTe and cuBLAS implementations with bfloat16
void benchmark_matmul() {
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;
    
    using T = __nv_bfloat16;
    
    // Allocate host memory for float data
    thrust::host_vector<float> h_A_float(M * K);
    thrust::host_vector<float> h_B_float(K * N);
    
    // Fill matrices with random values
    for (int i = 0; i < M * K; ++i) {
        h_A_float[i] = 2.0f * (rand() / double(RAND_MAX)) - 1.0f;
    }
    
    for (int i = 0; i < K * N; ++i) {
        h_B_float[i] = 2.0f * (rand() / double(RAND_MAX)) - 1.0f;
    }
    
    // Convert to bfloat16
    thrust::host_vector<T> h_A(M * K);
    thrust::host_vector<T> h_B(K * N);
    
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = float_to_bf16(h_A_float[i]);
    }
    
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = float_to_bf16(h_B_float[i]);
    }
    
    // Create device vectors
    thrust::device_vector<T> d_A = h_A;
    thrust::device_vector<T> d_B = h_B;
    thrust::device_vector<T> d_C_cute(M * N, T(0.0f));
    thrust::device_vector<T> d_C_cublas(M * N, T(0.0f));
    
    // Compute GFLOPS
    double gflops = (2.0 * M * N * K) * 1e-9;
    
    std::cout << "==== BF16 Matrix Multiplication Benchmark ====" << std::endl;
    std::cout << "Matrix size: " << M << "x" << N << "x" << K << std::endl;
    
    // Warmup runs
    cute_matmul(M, N, K, 
                1.0f,
                d_A.data().get(), K,
                d_B.data().get(), N,
                0.0f,
                d_C_cute.data().get(), N);
                
    cublas_matmul(M, N, K, 
                 1.0f,
                 d_A.data().get(), K,
                 d_B.data().get(), N,
                 0.0f,
                 d_C_cublas.data().get(), N);
    
    cudaDeviceSynchronize();
    
    // Time the CuTe kernel
    const int timing_iterations = 10;
    
    auto start_cute = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < timing_iterations; ++i) {
        cute_matmul(M, N, K, 
                    1.0f,
                    d_A.data().get(), K,
                    d_B.data().get(), N,
                    0.0f,
                    d_C_cute.data().get(), N);
    }
    
    cudaDeviceSynchronize();
    auto end_cute = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff_cute = end_cute - start_cute;
    double time_ms_cute = diff_cute.count() * 1000.0 / timing_iterations;
    double perf_cute = gflops / (time_ms_cute * 1e-3);
    
    // Time the cuBLAS kernel
    auto start_cublas = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < timing_iterations; ++i) {
        cublas_matmul(M, N, K, 
                     1.0f,
                     d_A.data().get(), K,
                     d_B.data().get(), N,
                     0.0f,
                     d_C_cublas.data().get(), N);
    }
    
    cudaDeviceSynchronize();
    auto end_cublas = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff_cublas = end_cublas - start_cublas;
    double time_ms_cublas = diff_cublas.count() * 1000.0 / timing_iterations;
    double perf_cublas = gflops / (time_ms_cublas * 1e-3);
    
    // Print performance results
    std::cout << "\n=== Performance Results ===" << std::endl;
    std::cout << "CuTe BF16 GEMM:   " << perf_cute << " GFLOPS, " << time_ms_cute << " ms" << std::endl;
    std::cout << "cuBLAS BF16 GEMM: " << perf_cublas << " GFLOPS, " << time_ms_cublas << " ms" << std::endl;
    std::cout << "Speedup of cuBLAS over CuTe: " << perf_cublas / perf_cute << "x" << std::endl;
    
    // Verify results
    cudaDeviceSynchronize();
    thrust::host_vector<T> result_cute = d_C_cute;
    thrust::host_vector<T> result_cublas = d_C_cublas;
    
    // Check if results match within a small tolerance
    const float tolerance = 1e-1; // Increase tolerance for bfloat16
    float max_diff = 0.0f;
    bool results_match = true;
    
    for (int i = 0; i < 10; ++i) { // Just check first few elements
        for (int j = 0; j < 10; ++j) {
            float diff = std::abs(bf16_to_float(result_cute[i * N + j]) - bf16_to_float(result_cublas[i * N + j]));
            max_diff = std::max(max_diff, diff);
            if (diff > tolerance) {
                results_match = false;
                break;
            }
        }
        if (!results_match) break;
    }
    
    std::cout << "\n=== Result Verification ===" << std::endl;
    std::cout << "Max difference between implementations: " << max_diff << std::endl;
    std::cout << "Results " << (results_match ? "match" : "do not match") 
              << " within tolerance of " << tolerance << std::endl;
    
    // Print a small portion of the results for verification
    std::cout << "\nSample results (top-left corner):" << std::endl;
    const int sample_size = std::min(4, M);
    
    std::cout << "CuTe BF16:" << std::endl;
    for (int i = 0; i < sample_size; ++i) {
        for (int j = 0; j < sample_size; ++j) {
            std::cout << bf16_to_float(result_cute[i * N + j]) << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\ncuBLAS BF16:" << std::endl;
    for (int i = 0; i < sample_size; ++i) {
        for (int j = 0; j < sample_size; ++j) {
            std::cout << bf16_to_float(result_cublas[i * N + j]) << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nBenchmark completed successfully!" << std::endl;
}

// Additional debugging function to validate cuBLAS transpose handling
void debug_transpose_issue() {
    const int M = 4;
    const int N = 4;
    const int K = 4;
    
    std::cout << "\n==== Transpose Handling Debug ====" << std::endl;
    
    // Create test matrices with recognizable pattern
    float h_A_float[M*K];
    float h_B_float[N*K];
    
    // Fill A with row-major pattern (A[i,j] = i*K + j)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            h_A_float[i*K + j] = i*K + j + 1;
        }
    }
    
    // Fill B with row-major pattern (B[i,j] = i*K + j)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            h_B_float[i*K + j] = i*K + j + 101; // offset for clarity
        }
    }
    
    // Convert to bfloat16
    __nv_bfloat16 h_A_bf16[M*K];
    __nv_bfloat16 h_B_bf16[N*K];
    __nv_bfloat16 h_C_cute_bf16[M*N] = {0};
    __nv_bfloat16 h_C_cublas_bf16[M*N] = {0};
    
    for (int i = 0; i < M*K; i++) {
        h_A_bf16[i] = float_to_bf16(h_A_float[i]);
    }
    
    for (int i = 0; i < N*K; i++) {
        h_B_bf16[i] = float_to_bf16(h_B_float[i]);
    }
    
    // Allocate device memory
    __nv_bfloat16 *d_A_bf16, *d_B_bf16, *d_C_cute_bf16, *d_C_cublas_bf16;
    cudaMalloc(&d_A_bf16, M*K*sizeof(__nv_bfloat16));
    cudaMalloc(&d_B_bf16, N*K*sizeof(__nv_bfloat16));
    cudaMalloc(&d_C_cute_bf16, M*N*sizeof(__nv_bfloat16));
    cudaMalloc(&d_C_cublas_bf16, M*N*sizeof(__nv_bfloat16));
    
    // Copy to device
    cudaMemcpy(d_A_bf16, h_A_bf16, M*K*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_bf16, h_B_bf16, N*K*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemset(d_C_cute_bf16, 0, M*N*sizeof(__nv_bfloat16));
    cudaMemset(d_C_cublas_bf16, 0, M*N*sizeof(__nv_bfloat16));
    
    // Print the original matrices for reference
    std::cout << "\nMatrix A (Row Major):" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            std::cout << std::setw(5) << h_A_float[i*K + j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nMatrix B (Row Major):" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            std::cout << std::setw(5) << h_B_float[i*K + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Run implementations
    cute_matmul(M, N, K, 1.0f, d_A_bf16, K, d_B_bf16, K, 0.0f, d_C_cute_bf16, N);
    cublas_matmul(M, N, K, 1.0f, d_A_bf16, K, d_B_bf16, K, 0.0f, d_C_cublas_bf16, N);
    
    // Copy results back
    cudaMemcpy(h_C_cute_bf16, d_C_cute_bf16, M*N*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_cublas_bf16, d_C_cublas_bf16, M*N*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    
    // Convert and print results
    float h_C_cute_float[M*N];
    float h_C_cublas_float[M*N];
    
    for (int i = 0; i < M*N; i++) {
        h_C_cute_float[i] = bf16_to_float(h_C_cute_bf16[i]);
        h_C_cublas_float[i] = bf16_to_float(h_C_cublas_bf16[i]);
    }
    
    std::cout << "\nSimple CUDA Result:" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << std::setw(8) << h_C_cute_float[i*N + j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\ncuBLAS Result:" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << std::setw(8) << h_C_cublas_float[i*N + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Compute reference matrix multiplication with transposition
    float h_C_ref_float[M*N] = {0};
    
    // Manual implementation for verification
    std::cout << "\nComputing reference with row-major storage:" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += h_A_float[i*K + k] * h_B_float[j*K + k];  // B is transposed 
            }
            h_C_ref_float[i*N + j] = sum;
        }
    }
    
    std::cout << "\nReference CPU Result:" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << std::setw(8) << h_C_ref_float[i*N + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Free device memory
    cudaFree(d_A_bf16);
    cudaFree(d_B_bf16);
    cudaFree(d_C_cute_bf16);
    cudaFree(d_C_cublas_bf16);
}

int main() {
    std::cout << "Running BF16 conversion validation..." << std::endl;
    validate_bf16_conversion();
    
    std::cout << "\nRunning small matrix debug test..." << std::endl;
    debug_small_matmul();
    
    std::cout << "\nRunning CuTe vs cuBLAS BF16 benchmark..." << std::endl;
    benchmark_matmul();
    
    return 0;
} 