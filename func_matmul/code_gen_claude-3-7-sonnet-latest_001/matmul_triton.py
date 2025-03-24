import torch
import triton
import triton.language as tl
import time


@triton.autotune(
    configs=[
        # Basic configurations for different matrix sizes
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how to access the next element
    # of the matrix along a particular dimension.
    stride_am, stride_ak,  # Strides for A
    stride_bk, stride_bn,  # Strides for B
    stride_cm, stride_cn,  # Strides for C
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr = ""
):
    """
    Compute the matrix multiplication C = A @ B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)

    This kernel uses a block-based algorithm with:
    - BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K: tile sizes for the computation
    - GROUP_SIZE_M: number of rows of blocks to compute together (super-blocking)
    """
    # -----------------------------------------------------------
    # Map program ID to the block of C it should compute
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create block pointers for the first blocks of A and B
    # A block is of shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
    # B block is of shape (BLOCK_SIZE_K, BLOCK_SIZE_N)

    # Offset of the first block of A based on the block id (pid_m)
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # Filter out out-of-bounds accesses for A
    # offs_am = tl.where(offs_am < M, offs_am, 0)
    # Offset in K dimension (start from 0)
    offs_ak = tl.arange(0, BLOCK_SIZE_K)
    # Compute the pointer to the first block of A
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak

    # Offset of the first block of B based on the block id (pid_n)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # Filter out out-of-bounds accesses for B
    # offs_bn = tl.where(offs_bn < N, offs_bn, 0)
    # Offset in K dimension (start from 0)
    offs_bk = tl.arange(0, BLOCK_SIZE_K)
    # Compute the pointer to the first block of B
    b_ptrs = b_ptr + offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix
    # We accumulate into a block of BLOCK_SIZE_M x BLOCK_SIZE_N
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over k blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, using the masks to handle boundary conditions
        k_mask = tl.arange(0, BLOCK_SIZE_K) < K - k * BLOCK_SIZE_K

        # Load blocks of A and B from DRAM into SRAM
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)

        # We compute a block of C by multiplying a block of A with a block of B
        # using a GEMM operation. We accumulate the results in the accumulator.
        accumulator += tl.dot(a, b, out_dtype=tl.float32)

        # Move the pointers to the next k-block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Optional: Apply activation function to output (not used in matrix multiply)
    if ACTIVATION == "relu":
        accumulator = tl.where(accumulator >= 0.0, accumulator, 0.0)
    elif ACTIVATION == "sigmoid":
        accumulator = 1.0 / (1.0 + tl.exp(-accumulator))

    # convert accumulator to original data type
    # accumulator = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the results to C
    # We need to compute the target addresses for the block of C we computed
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Compute the actual bounds to guard against out-of-bounds accesses
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # Compute the pointer to the block of C where we'll write our results
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    # Write the results back to DRAM
    tl.store(c_ptrs, accumulator, mask=c_mask)


def matmul(a, b, activation=""):
    """
    Compute the matrix multiplication C = A @ B using Triton.

    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)
        activation: Optional activation function to apply to the output

    Returns:
        Output tensor of shape (M, N)
    """
    # Check constraints
    assert a.shape[1] == b.shape[0], f"Incompatible dimensions: {a.shape} and {b.shape}"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"

    # Get dimensions
    M, K = a.shape
    K, N = b.shape

    # Allocate output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Calculate the strides for each matrix
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b.stride(0), b.stride(1)
    stride_cm, stride_cn = c.stride(0), c.stride(1)

    # Calculate grid size
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    # Launch the kernel with the appropriate configuration
    _matmul_kernel[grid](
        a_ptr=a, b_ptr=b, c_ptr=c,
        M=M, N=N, K=K,
        stride_am=stride_am, stride_ak=stride_ak,
        stride_bk=stride_bk, stride_bn=stride_bn,
        stride_cm=stride_cm, stride_cn=stride_cn,
        ACTIVATION=activation
    )

    return c


def generated_torch_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation using PyTorch.

    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)

    Returns:
        Output tensor of shape (M, N)
    """
    return torch.matmul(a, b)


if __name__ == "__main__":
    M_sizes = [256, 1024, 4096]
    N_sizes = [256, 1024, 4096]
    K_sizes = [256, 512, 1024, 2048, 4096]
    def input_generator(M, N, K, device='cuda', dtype=torch.float16):
        A = torch.randn(M, K, device=device, dtype=dtype)
        A = torch.nn.functional.normalize(A, dim=-1)
        B = torch.randn(K, N, device=device, dtype=dtype)
        B = torch.nn.functional.normalize(B, dim=-1)
        return A, B

    from benchmark import verify_correctness_func, benchmark_performance_func
    verify_correctness_func(matmul, generated_torch_func, input_generator, [M_sizes, N_sizes, K_sizes])
    benchmark_performance_func(matmul, generated_torch_func, input_generator, [M_sizes, N_sizes, K_sizes])
