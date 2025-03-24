import torch
import numpy as np
import json
import matplotlib.pyplot as plt

from benchmark import verify_correctness_func, benchmark_performance_func

if __name__ == '__main__':
    # Example test configuration
    batch_sizes = [16, 32]
    input_dims = [128, 256]
    output_dims = [64, 128]
    dimensions = [batch_sizes, input_dims, output_dims]

    # Example input generator
    def input_generator(batch_size, input_dim, output_dim, device='cuda', dtype=torch.float16):
        return torch.randn(batch_size, input_dim, device=device, dtype=dtype)

    # Example functions to compare
    def triton_func(x):
        return torch.nn.functional.relu(x)

    def torch_func(x):
        return torch.nn.functional.relu(x)

    # Run correctness test
    print("Running correctness test...")
    correctness = verify_correctness_func(
        triton_func,
        torch_func,
        input_generator,
        dimensions,
        device='cuda',
        dtype=torch.float32
    )
    print(f"Correctness tests: {correctness['passed']}/{correctness['total']} passed")

    # Run performance benchmark
    print("\nRunning performance benchmark...")
    triton_time, torch_time, speedup, perf = benchmark_performance_func(
        triton_func,
        torch_func,
        input_generator,
        dimensions,
        device='cuda',
        dtype=torch.float32
    )
    print(f"\nMedian times:\n  Triton: {triton_time:.2f} us\n  PyTorch: {torch_time:.2f} us\n  Speedup: {speedup:.2f}x")
