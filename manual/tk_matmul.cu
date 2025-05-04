#include "kittens.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <vector>
#include <random>

using namespace kittens;

// Forward declaration for benchmark
void benchmark_tk_vs_cublas();

// Forward declaration for fast matmul
void thunderkitten_fast_matmul(const __nv_bfloat16 *A,
                               const __nv_bfloat16 *B,
                               __nv_bfloat16       *C,
                               int M,int N,int K, cudaStream_t stream=0);

// Define tile sizes and layout for matrix multiplication
constexpr int TILE_DIM = 16;  // Tile dimension

//-----------------------------------------------------------------------------
// Warp-level GEMM kernel powered by ThunderKittens primitives
// Each CUDA block == one warp (32 threads) computes a single 16×16 output tile.
//-----------------------------------------------------------------------------

template<typename GlA, typename GlB, typename GlC>
__global__ void thunderkitten_kernel(GlA Ag, GlB Bg, GlC Cg,
                                     int M, int N, int K) {

    // Tile index that this warp is responsible for
    const int tile_m = blockIdx.y;        //   M-tile index
    const int tile_n = blockIdx.x;        //   N-tile index

    // Guard against partial tiles (expects M,N multiples of 16 for simplicity)
    if (tile_m * TILE_DIM >= M || tile_n * TILE_DIM >= N) return;

    //---------------------------------------------------------------------
    // Aliases / type shortcuts
    //---------------------------------------------------------------------
    using bf16    = kittens::bf16;
    using rtA_t   = kittens::rt_bf<TILE_DIM, TILE_DIM>;                                           // row-major  A
    using rtB_t   = kittens::rt_bf<TILE_DIM, TILE_DIM, kittens::ducks::rt_layout::col>;           // col-major  B
    using acc_t   = kittens::rt_fl<TILE_DIM, TILE_DIM>;                                           // fp32 accum

    //---------------------------------------------------------------------
    // Main MMA accumulation loop over K dimension (in 16-element chunks)
    //---------------------------------------------------------------------
    acc_t accum;   kittens::zero(accum);

    const int num_k_tiles = (K + TILE_DIM - 1) / TILE_DIM;
    for (int ktile = 0; ktile < num_k_tiles; ++ktile) {
        // Load one 16×16 tile of A and B from GMEM straight into registers
        rtA_t a_reg;  kittens::load(a_reg, Ag, kittens::coord<rtA_t>(tile_m, ktile));
        rtB_t b_reg;  kittens::load(b_reg, Bg, kittens::coord<rtB_t>(ktile  , tile_n));

        // HMMA-based multiply-accumulate
        kittens::mma_AB(accum, a_reg, b_reg, accum);
    }

    //---------------------------------------------------------------------
    // Convert FP32 accumulator → BF16 and store back to global memory
    //---------------------------------------------------------------------
    kittens::rt_bf<TILE_DIM, TILE_DIM> c_bf;
    c_bf = accum;                                      // implicit fp32→bf16 conversion
    kittens::store(Cg, c_bf, kittens::coord<decltype(c_bf)>(tile_m, tile_n));
}

// Function to launch the kernel
void thunderkitten_matmul(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M, int N, int K,
    cudaStream_t stream = 0) {
    
    // Calculate grid dimensions
    dim3 block(kittens::WARP_THREADS);   // 1 warp (32 threads)
    dim3 grid(
        (N + TILE_DIM - 1) / TILE_DIM,
        (M + TILE_DIM - 1) / TILE_DIM
    );
    
    using bf16 = kittens::bf16;
    using glA_t = kittens::gl<bf16, -1, -1, -1, -1>;
    using glB_t = kittens::gl<bf16, -1, -1, -1, -1>;
    using glC_t = kittens::gl<bf16, -1, -1, -1, -1>;

    glA_t Ag(const_cast<bf16*>(reinterpret_cast<const bf16*>(A)), 1, 1, M, K);
    glB_t Bg(const_cast<bf16*>(reinterpret_cast<const bf16*>(B)), 1, 1, K, N);
    glC_t Cg(reinterpret_cast<bf16*>(C),                          1, 1, M, N);

    thunderkitten_kernel<<<grid, block, 0, stream>>>(Ag, Bg, Cg, M, N, K);
}

// Test function
void test_thunderkitten_matmul() {
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    
    // Use bf16 for operations
    using compute_t = __nv_bfloat16;
    
    // Allocate host memory
    compute_t* h_A = new compute_t[M * K];
    compute_t* h_B = new compute_t[K * N];
    compute_t* h_C = new compute_t[M * N];
    
    // Initialize matrices
    for (int i = 0; i < M * K; ++i) h_A[i] = __float2bfloat16(1.0f);
    for (int i = 0; i < K * N; ++i) h_B[i] = __float2bfloat16(1.0f);
    
    // Allocate device memory
    compute_t* d_A;
    compute_t* d_B;
    compute_t* d_C;
    cudaMalloc(&d_A, M * K * sizeof(compute_t));
    cudaMalloc(&d_B, K * N * sizeof(compute_t));
    cudaMalloc(&d_C, M * N * sizeof(compute_t));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(compute_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(compute_t), cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);
    
    // Perform matrix multiplication
    thunderkitten_matmul(d_A, d_B, d_C, M, N, K);
    
    // Record stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(compute_t), cudaMemcpyDeviceToHost);
    
    // Verify result
    bool correct = true;
    for (int i = 0; i < M * N; ++i) {
        float val = __bfloat162float(h_C[i]);
        if (abs(val - float(K)) > 0.1f) {
            correct = false;
            break;
        }
    }
    
    std::cout << "ThunderKittens Matrix Multiplication Test: " 
              << (correct ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Execution time: " << milliseconds << " ms" << std::endl;
    
    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    test_thunderkitten_matmul();
    benchmark_tk_vs_cublas();
    return 0;
}

// -------------  BENCHMARK SECTION  ----------------------------------------

void benchmark_tk_vs_cublas() {
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    constexpr int ITERS = 100;

    using compute_t = __nv_bfloat16;

    // Allocate host memory
    std::vector<float>  h_A_fp32(M * K);
    std::vector<float>  h_B_fp32(K * N);
    std::vector<float>  h_C_fp32(M * N, 0.f);

    // Fill with random values
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : h_A_fp32) v = dist(gen);
    for (auto &v : h_B_fp32) v = dist(gen);

    // Convert to bf16
    std::vector<compute_t> h_A_bf16(M*K), h_B_bf16(K*N);
    for (size_t i = 0; i < h_A_fp32.size(); ++i) h_A_bf16[i] = __float2bfloat16(h_A_fp32[i]);
    for (size_t i = 0; i < h_B_fp32.size(); ++i) h_B_bf16[i] = __float2bfloat16(h_B_fp32[i]);

    // Device allocations
    compute_t *d_A, *d_B; compute_t *d_C_tk, *d_C_cu;
    cudaMalloc(&d_A, h_A_bf16.size()*sizeof(compute_t));
    cudaMalloc(&d_B, h_B_bf16.size()*sizeof(compute_t));
    cudaMalloc(&d_C_tk, h_C_fp32.size()*sizeof(compute_t));
    cudaMalloc(&d_C_cu, h_C_fp32.size()*sizeof(compute_t));

    cudaMemcpy(d_A, h_A_bf16.data(), h_A_bf16.size()*sizeof(compute_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_bf16.data(), h_B_bf16.size()*sizeof(compute_t), cudaMemcpyHostToDevice);

    // ---------------- ThunderKittens -------------------------------
    {
        cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
        thunderkitten_fast_matmul(d_A, d_B, d_C_tk, M, N, K); // warmup
        cudaEventRecord(start);
        for (int i=0;i<ITERS;i++) thunderkitten_fast_matmul(d_A, d_B, d_C_tk, M, N, K);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms = 0.f; cudaEventElapsedTime(&ms, start, stop);
        float avg_ms = ms/ITERS;
        double tflops = 2.0 * M * N * K / (avg_ms*1e6);
        std::cout << "ThunderKittens  avg " << avg_ms << " ms  -> " << tflops << " TFLOPS\n";
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    // ---------------- cuBLAS -------------------------------
    {
        cublasHandle_t handle; cublasCreate(&handle);
        const float alpha = 1.f, beta = 0.f;
        cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
        // Warmup
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                      N, M, K,
                      &alpha,
                      d_B, CUDA_R_16BF, N,
                      d_A, CUDA_R_16BF, K,
                      &beta,
                      d_C_cu, CUDA_R_16BF, N,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaEventRecord(start);
        for (int i=0;i<ITERS;i++)
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                          N, M, K,
                          &alpha,
                          d_B, CUDA_R_16BF, N,
                          d_A, CUDA_R_16BF, K,
                          &beta,
                          d_C_cu, CUDA_R_16BF, N,
                          CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms=0.f; cudaEventElapsedTime(&ms, start, stop);
        float avg_ms = ms/ITERS;
        double tflops = 2.0 * M * N * K / (avg_ms*1e6);
        std::cout << "cuBLAS          avg " << avg_ms << " ms  -> " << tflops << " TFLOPS\n";
        cudaEventDestroy(start); cudaEventDestroy(stop);
        cublasDestroy(handle);
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_tk); cudaFree(d_C_cu);
}

#ifdef KITTENS_HOPPER
// ---------------------------------------------------------------------------
//  Warp-group (4-warp) WGMMA kernel – much higher arithmetic intensity
//  Computes a 64×128×16 macro-tile per warp-group
// ---------------------------------------------------------------------------

template<typename GlA, typename GlB, typename GlC>
__global__ void tk_wgmma_kernel(GlA Ag, GlB Bg, GlC Cg,
                                int M, int N, int K) {
    using namespace kittens;
    using group4 = group<4>;

    constexpr int TM = 64;   // 4 sub tiles (rows)
    constexpr int TN = 128;  // 8 sub tiles (cols)

    const int tile_m = blockIdx.y;
    const int tile_n = blockIdx.x;
    if (tile_m*TM >= M || tile_n*TN >= N) return;

    // Shared tiles (double buffered not needed for simplicity)
    extern __shared__ __align__(16) char smem[];
    using stA_t = st_bf<TM, 16>;   // 64×16
    using stB_t = st_bf<16, TN>;   // 16×128
    stA_t &sA = *reinterpret_cast<stA_t*>(smem);
    stB_t &sB = *reinterpret_cast<stB_t*>(smem + sizeof(stA_t));

    // Accumulator
    rt_fl<TM, TN> acc;  kittens::zero(acc);

    for (int kt=0; kt<K; kt+=16) {
        // Cooperative load GMEM -> SMEM
        group4::load_async(sA, Ag, coord<stA_t>(tile_m, kt/16));
        group4::load_async(sB, Bg, coord<stB_t>(kt/16, tile_n));
        group4::sync(0);

        // MMA
        group4::mm_AB(acc, sA, sB);
        group4::sync(0);
    }

    // Store result
    rt_bf<TM, TN> c_bf; c_bf = acc;
    group4::store(Cg, c_bf, coord<decltype(c_bf)>(tile_m, tile_n));
}
#endif // KITTENS_HOPPER

#ifndef KITTENS_HOPPER
// Fallback for pre-Hopper architectures: just call the regular warp-level kernel
void thunderkitten_fast_matmul(const __nv_bfloat16 *A,
                               const __nv_bfloat16 *B,
                               __nv_bfloat16       *C,
                               int M,int N,int K, cudaStream_t stream) {
    thunderkitten_matmul(A,B,C,M,N,K,stream);
}
#endif 