# filename: verifier/correctness_test.py

import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F


# Example functions to compare
def triton_func(x):
    return torch.nn.functional.relu(x)


def torch_func(x):
    return torch.nn.functional.relu(x)


class TritonModule(nn.Module):
    def __init__(self, I, O, device="cuda", dtype=torch.float16):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(O, I, device=device, dtype=dtype))

    def forward(self, x):
        x = x @ self.weight.T
        return torch.nn.functional.relu(x)


class TorchModule(nn.Module):
    def __init__(self, I, O, device="cuda", dtype=torch.float16):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(O, I, device=device, dtype=dtype))

    def forward(self, x):
        x = x @ self.weight.T
        return torch.nn.functional.relu(x)


def correctness_func():
    # Example test configuration
    M_dims = [16, 64, 256]
    N_dims = [128, 256]
    K_dims = [64, 128]
    dimensions = [M_dims, N_dims, K_dims]

    # Example input generator
    def input_generator(M_size, N_size, K_size, device="cuda", dtype=torch.float16):
        result = torch.randn(M_size, N_size, K_size, device=device, dtype=dtype)
        result = F.normalize(result, p=2, dim=-1)
        return result

    from verifier.correctness import verify_correctness_func

    print("Running correctness test...")
    correctness = verify_correctness_func(
        triton_func,
        torch_func,
        input_generator,
        dimensions,
        device="cuda",
        dtype=torch.float16,
    )
    print(f"Correctness tests: {correctness['passed']}/{correctness['total']} passed")


def correctness_module():
    from verifier.correctness import (
        verify_correctness_forward,
        verify_correctness_backward,
    )

    # Example test configuration
    batch_dims = [16, 64, 256]
    input_dims = [128, 512, 2048]
    output_dims = [64, 256, 1024]
    dimensions = [batch_dims, input_dims, output_dims]

    def module_generator(B, I, O, device="cuda", dtype=torch.float16):
        triton_module = TritonModule(I, O, device=device, dtype=dtype)
        torch_module = TorchModule(I, O, device=device, dtype=dtype)
        return triton_module, torch_module

    def forward_generator(B, I, O, device="cuda", dtype=torch.float16):
        input = torch.randn(B, I, device=device, dtype=dtype)
        input = torch.nn.functional.normalize(input, dim=-1)
        return input

    def backward_generator(B, I, O, device="cuda", dtype=torch.float16):
        input = torch.randn(B, I, device=device, dtype=dtype)
        input = torch.nn.functional.normalize(input, dim=-1)
        d_output = torch.randn(B, O, device=device, dtype=dtype)
        d_output = torch.nn.functional.normalize(d_output, dim=-1)
        return input, d_output

    from verifier.correctness import verify_correctness_forward

    print("Running correctness test (forward)...")
    correctness = verify_correctness_forward(
        module_generator, forward_generator, dimensions
    )
    print(f"Correctness tests: {correctness['passed']}/{correctness['total']} passed")

    from verifier.correctness import verify_correctness_backward

    print("Running correctness test (backward)...")
    correctness = verify_correctness_backward(
        module_generator, backward_generator, dimensions
    )
    print(f"Correctness tests: {correctness['passed']}/{correctness['total']} passed")


if __name__ == "__main__":

    correctness_func()
    correctness_module()
