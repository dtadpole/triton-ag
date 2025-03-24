import torch
import triton
import triton.language as tl

"""
Optimized matrix multiplication using Triton with advanced features:
- Block-based computation for better memory access patterns
- Autotuning for block sizes (M, N, K)
- Super-blocking with autotuned GROUP_SIZE_M
- Memory coalescing and register optimizations
"""

@triton.autotune(
    configs=[
        # Basic configs varying block sizes
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}),
        # Additional configs with different GROUP_SIZE_M
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}),
        # Large matrix configs
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how to access the next element along a
    # particular dimension. For example, `stride_am` represents how to access
    # the next element along the M dimension of the A matrix.
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute C = A @ B using Triton with advanced blocking strategies

    Args:
        a_ptr: pointer to A matrix of shape (M, K)
        b_ptr: pointer to B matrix of shape (K, N)
        c_ptr: pointer to C matrix of shape (M, N)
        M, N, K: dimensions of matrices
        stride_am, stride_ak: strides of the A matrix
        stride_bk, stride_bn: strides of the B matrix
        stride_cm, stride_cn: strides of the C matrix
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K: block sizes for tiling
        GROUP_SIZE_M: super-block size for M dimension
    """
    # -----------------------------------------------------------
    # Map program ids to the blocks of C being computed
    # -----------------------------------------------------------

    # Program ID
    pid = tl.program_id(axis=0)

    # Number of blocks in N dimension
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Number of blocks in M dimension
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)

    # Number of blocks in a super-block
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Group ID (super-block ID)
    group_id = pid // num_pid_in_group

    # Local ID within the group
    first_pid_m = group_id * GROUP_SIZE_M

    # Get the block positions within the super-block
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Calculate the starting indices for this block
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    # Create block ranges
    rm = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    rn = block_start_n + tl.arange(0, BLOCK_SIZE_N)

    # Apply bounds checking (for edge blocks)
    # rm = tl.where(rm < M, rm, 0)
    # rn = tl.where(rn < N, rn, 0)

    # -----------------------------------------------------------
    # Create pointers for the block of data to be loaded
    # -----------------------------------------------------------

    # Pointers for A matrix
    a_ptrs = a_ptr + rm[:, None] * stride_am + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_ak

    # Pointers for B matrix
    b_ptrs = b_ptr + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_bk + rn[None, :] * stride_bn

    # -----------------------------------------------------------
    # Initialize accumulator with zeros
    # -----------------------------------------------------------
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix
    # -----------------------------------------------------------

    # Iterate through k-dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Boundary check for K
        # k_remaining = K - k * BLOCK_SIZE_K
        # k_size = min(k_remaining, BLOCK_SIZE_K)

        # Load A and B blocks from global memory
        # Apply masking to handle edge cases
        a_mask = (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        b_mask = (offs_k[:, None] < K - k * BLOCK_SIZE_K)

        # Load the blocks from DRAM
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Convert to higher precision for accumulation
        # a = a.to(tl.float32)
        # b = b.to(tl.float32)

        # Perform the matrix multiplication
        acc += tl.dot(a, b)

        # Advance the pointers for the next iteration
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # -----------------------------------------------------------
    # Write the result to global memory
    # -----------------------------------------------------------

    # Create pointers for C matrix
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn

    # Create a mask for valid elements of the result (handle edge cases)
    mask = (rm[:, None] < M) & (rn[None, :] < N)

    # Write the result back to global memory with the mask
    tl.store(c_ptrs, acc, mask=mask)

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the matrix product C = A @ B

    Args:
        a: tensor of shape (M, K)
        b: tensor of shape (K, N)

    Returns:
        c: tensor of shape (M, N)
    """
    # Extract dimensions
    M, K = a.shape
    K_b, N = b.shape

    # Check dimensions
    assert K == K_b, f"Incompatible dimensions: {a.shape} and {b.shape}"

    # Check device and data type
    assert a.device == b.device, "Inputs must be on the same device"
    assert a.dtype == b.dtype, "Inputs must have the same data type"

    # Allocate output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Calculate memory strides (for addressing linearized memory)
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b.stride(0), b.stride(1)
    stride_cm, stride_cn = c.stride(0), c.stride(1)

    # Launch the CUDA kernel with optimal grid configuration
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
    )

    return c

def generated_torch_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation for matrix multiplication.

    Args:
        a: tensor of shape (M, K)
        b: tensor of shape (K, N)

    Returns:
        c: tensor of shape (M, N)
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
