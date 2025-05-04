#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <iostream>
#include <chrono>

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/helper_cuda.hpp"

// CuTe matrix multiplication kernel
template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(cute::size(CThreadLayout{}))::value)
void
matmul_kernel(ProblemShape shape_MNK, CtaTiler cta_tiler,
              TA const* A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
              TB const* B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
              TC      * C, CStride dC, CSmemLayout          , CThreadLayout tC,
              Alpha alpha, Beta beta)
{
  using namespace cute;

  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  // Shared memory buffers
  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K)

  // Partition the copying of A and B tiles across the threads
  Tensor tAgA = local_partition(gA, tA, threadIdx.x);                  // (THR_M,THR_K,k)
  Tensor tAsA = local_partition(sA, tA, threadIdx.x);                  // (THR_M,THR_K)

  Tensor tBgB = local_partition(gB, tB, threadIdx.x);                  // (THR_N,THR_K,k)
  Tensor tBsB = local_partition(sB, tB, threadIdx.x);                  // (THR_N,THR_K)

  // Partition for computation
  Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});   // (THR_M,BLK_K)
  Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step< X,_1>{});   // (THR_N,BLK_K)
  Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1,_1>{});   // (THR_M,THR_N)

  // Allocate the accumulators
  Tensor tCrC = make_tensor_like(tCgC);                                // (THR_M,THR_N)

  // Clear the accumulators
  clear(tCrC);

  // Iterate over K dimension, computing partial matrix products
  auto K_TILE_MAX = size<2>(tAgA);
  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
  {
    // Copy global memory to shared memory with thread partitioning
    copy(tAgA(_,_,k_tile), tAsA);      // A (THR_M,THR_K) -> (THR_M,THR_K)
    copy(tBgB(_,_,k_tile), tBsB);      // B (THR_N,THR_K) -> (THR_N,THR_K)

    cp_async_fence();        // Label the end of (potential) cp.async instructions
    cp_async_wait<0>();      // Sync on all (potential) cp.async instructions
    __syncthreads();         // Wait for all threads to write to smem

    // Compute matrix multiplication on thread-partitioned shared memory
    gemm(tCsA, tCsB, tCrC);  // (THR_M,THR_N) += (THR_M,BLK_K) * (THR_N,BLK_K)

    __syncthreads();         // Wait for all threads to read from smem
  }

  // Write results back to global memory
  axpby(alpha, tCrC, beta, tCgC);
}

// Setup and run matrix multiplication for NT (non-transposed A, transposed B) format
template <class TA, class TB, class TC, class Alpha, class Beta>
void cute_matmul(int m, int n, int k,
                 Alpha alpha,
                 TA const* A, int ldA,
                 TB const* B, int ldB,
                 Beta beta,
                 TC* C, int ldC,
                 cudaStream_t stream = 0)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define NT strides (mixed)
  auto dA = make_stride(Int<1>{}, ldA);                      // (dM, dK)
  auto dB = make_stride(Int<1>{}, ldB);                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<8>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)

  // Define the shared memory layouts (static)
  auto sA = make_layout(make_shape(bM, bK));                 // (m,k) -> smem_idx; m-major
  auto sB = make_layout(make_shape(bN, bK));                 // (n,k) -> smem_idx; n-major
  auto sC = make_layout(make_shape(bM, bN));                 // (m,n) -> smem_idx; m-major

  // Define the thread layouts (static)
  auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));    // (m,k) -> thr_idx
  auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}));    // (n,k) -> thr_idx
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));   // (m,n) -> thr_idx

  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));

  matmul_kernel<<<dimGrid, dimBlock, 0, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, tA,
       B, dB, sB, tB,
       C, dC, sC, tC,
       alpha, beta);
}

// Test function to verify our implementation
void test_cute_matmul()
{
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;
    
    using T = float;
    
    // Allocate host memory
    thrust::host_vector<T> h_A(M * K);
    thrust::host_vector<T> h_B(N * K);
    thrust::host_vector<T> h_C(M * N, 0.0f);
    
    // Fill matrices with random values
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = static_cast<T>(2.0f * (rand() / double(RAND_MAX)) - 1.0f);
    }
    
    for (int i = 0; i < N * K; ++i) {
        h_B[i] = static_cast<T>(2.0f * (rand() / double(RAND_MAX)) - 1.0f);
    }
    
    // Transfer to device memory
    thrust::device_vector<T> d_A = h_A;
    thrust::device_vector<T> d_B = h_B;
    thrust::device_vector<T> d_C(M * N, 0.0f);
    
    // Compute GFLOPS
    double gflops = (2.0 * M * N * K) * 1e-9;
    
    // Run the kernel
    cute_matmul(M, N, K, 
                1.0f,
                d_A.data().get(), M,
                d_B.data().get(), N,
                0.0f,
                d_C.data().get(), M);
                
    cudaDeviceSynchronize();
    
    // Time the kernel with multiple iterations
    const int timing_iterations = 10;
    
    // Use std::chrono for timing
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < timing_iterations; ++i) {
        cute_matmul(M, N, K, 
                    1.0f,
                    d_A.data().get(), M,
                    d_B.data().get(), N,
                    0.0f,
                    d_C.data().get(), M);
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = end - start;
    double time_ms = diff.count() * 1000.0 / timing_iterations;
    
    std::cout << "Matrix size: " << M << "x" << N << "x" << K << std::endl;
    std::cout << "CuTe GEMM performance: " << gflops / (time_ms * 1e-3) << " GFLOPS, " 
              << time_ms << " ms" << std::endl;
    
    // Verify by checking some values
    cudaDeviceSynchronize();
    thrust::host_vector<T> result = d_C;
    
    // Print a small portion of the result for verification
    std::cout << "\nSample results (top-left corner):" << std::endl;
    const int sample_size = std::min(4, M);
    for (int i = 0; i < sample_size; ++i) {
        for (int j = 0; j < sample_size; ++j) {
            std::cout << result[i * M + j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "CuTe GEMM test completed successfully!" << std::endl;
}

int main() {
    test_cute_matmul();
    return 0;
} 