#include <iostream>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/gemm.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>

// Define the CUTLASS GEMM operation
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
    ElementC,
    128 / cutlass::sizeof_bits<ElementC>::value,
    ElementAccumulator,
    ElementAccumulator
>;

using Gemm = cutlass::gemm::device::Gemm<
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3
>;

// Function to perform matrix multiplication using CUTLASS
void cutlass_matmul(
    const ElementA* A,
    const ElementB* B,
    ElementC* C,
    int M,
    int N,
    int K,
    cudaStream_t stream = 0
) {
    // Create arguments for the GEMM operation
    typename Gemm::Arguments arguments{
        {M, N, K},
        {A, K},
        {B, N},
        {C, N},
        {C, N},
        {ElementAccumulator(1), ElementAccumulator(0)}
    };

    // Initialize the GEMM operation
    Gemm gemm_op;
    cutlass::Status status = gemm_op.initialize(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to initialize CUTLASS GEMM operation" << std::endl;
        return;
    }

    // Run the GEMM operation
    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to run CUTLASS GEMM operation" << std::endl;
        return;
    }
}

// Test function
void test_cutlass_matmul() {
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    // Allocate host memory
    cutlass::HostTensor<ElementA, LayoutA> A({M, K});
    cutlass::HostTensor<ElementB, LayoutB> B({K, N});
    cutlass::HostTensor<ElementC, LayoutC> C({M, N});
    cutlass::HostTensor<ElementC, LayoutC> C_ref({M, N});

    // Fill matrices with random values
    cutlass::reference::host::TensorFillRandomUniform(
        A.host_view(),
        1,
        ElementA(1),
        ElementA(-1),
        0
    );

    cutlass::reference::host::TensorFillRandomUniform(
        B.host_view(),
        1,
        ElementB(1),
        ElementB(-1),
        0
    );

    // Copy to device
    A.sync_device();
    B.sync_device();
    C.sync_device();
    C_ref.sync_device();

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Run CUTLASS GEMM
    cutlass_matmul(
        A.device_data(),
        B.device_data(),
        C.device_data(),
        M,
        N,
        K,
        stream
    );

    // Run reference GEMM
    cutlass::reference::device::Gemm<
        ElementA,
        LayoutA,
        ElementB,
        LayoutB,
        ElementC,
        LayoutC,
        ElementAccumulator,
        ElementAccumulator
    > gemm_ref;

    gemm_ref(
        {M, N, K},
        ElementAccumulator(1),
        A.device_ref(),
        B.device_ref(),
        ElementAccumulator(0),
        C_ref.device_ref(),
        ElementAccumulator(0)  // beta
    );

    // Wait for completion
    cudaStreamSynchronize(stream);

    // Copy results back to host
    C.sync_host();
    C_ref.sync_host();

    // Apply ReLU activation
    for (int i = 0; i < M * N; ++i) {
        C_ref.host_data()[i] = (float(C_ref.host_data()[i]) > 0) ? C_ref.host_data()[i] : ElementC(0);
    }

    // Compare results
    bool passed = true;
    for (int i = 0; i < M * N; ++i) {
        if (std::abs(float(C.host_data()[i]) - float(C_ref.host_data()[i])) > 1e-2) {
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "✅ CUTLASS GEMM test passed" << std::endl;
    } else {
        std::cout << "❌ CUTLASS GEMM test failed" << std::endl;
    }

    // Cleanup
    cudaStreamDestroy(stream);
}

int main() {
    test_cutlass_matmul();
    return 0;
} 