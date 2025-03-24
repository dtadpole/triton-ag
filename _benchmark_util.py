import torch
import triton
import numpy as np
import json
import matplotlib.pyplot as plt

def _generate_configs(dims):
    """
    Generate all possible combinations of dimensions for benchmarking.

    Args:
        dims (List[List[int]]): List of lists containing possible values for each dimension.
            Example: [[16, 32], [128, 256], [512, 1024]] represents batch_sizes, input_dims, output_dims.

    Returns:
        List[List[int]]: List of all possible dimension combinations.
            Example: [[16, 128, 512], [16, 128, 1024], ..., [32, 256, 1024]]

    Example:
        >>> dims = [[16, 32], [128, 256], [512, 1024]]
        >>> _generate_configs(dims)
        [[16, 128, 512], [16, 128, 1024], [16, 256, 512], [16, 256, 1024],
         [32, 128, 512], [32, 128, 1024], [32, 256, 512], [32, 256, 1024]]
    """
    configs = []
    indices = np.zeros(len(dims), dtype=np.int32)

    # increment indices
    def increment_indices():
        for i in range(len(dims) - 1, -1, -1):
            if indices[i] < len(dims[i]) - 1:
                indices[i] += 1
                return True
            indices[i] = 0
        return False

    while True:
        stack = []
        for i, dim in enumerate(dims):
            stack.append(dim[indices[i]])
        configs.append(stack.copy())
        if not increment_indices():
            break

    return configs

def _check_outputs_match(config, correctness, triton_output, torch_output, dtype, atol=1e-2, rtol=1e-2, label=""):
    """
    Check if outputs from Triton and PyTorch implementations match within tolerance.

    Args:
        config (List[int]): Current configuration being tested (e.g., [batch_size, input_dim, output_dim]).
        correctness (Dict): Dictionary to track test results with keys:
            - passed (int): Number of passed tests
            - failed (int): Number of failed tests
            - total (int): Total number of tests
            - results (List): Detailed results for each test
        triton_output (torch.Tensor | Tuple[torch.Tensor]): Output from Triton implementation.
        torch_output (torch.Tensor | Tuple[torch.Tensor]): Output from PyTorch implementation.
        dtype (torch.dtype): Data type used for tensors.
        atol (float, optional): Absolute tolerance. Defaults to 1e-2.
        rtol (float, optional): Relative tolerance. Defaults to 1e-2.
        label (str, optional): Label for the test (e.g., "forward", "backward"). Defaults to "".

    Returns:
        bool: True if outputs match within tolerance, False otherwise.
    """
    mismatch = False
    if isinstance(torch_output, tuple):
        for i in range(len(torch_output)):
            if torch.allclose(triton_output[i], torch_output[i], atol=atol, rtol=rtol):
                pass
            else:
                mismatch = True
                break
    else:
        if torch.allclose(triton_output, torch_output, atol=atol, rtol=rtol):
            pass
        else:
            mismatch = True

    if not mismatch:
        print(f"✅ Correctness test passed for shape ({config}, {dtype}) [{label}]")
        correctness["passed"] += 1
        correctness["total"] += 1
        correctness["results"].append({
            "shape": config,
            "dtype": str(dtype),
            "correctness": True
        })
    else:
        print(f"❌ Correctness test failed for shape ({config}, {dtype}) [{label}]")
        correctness["failed"] += 1
        correctness["total"] += 1
        correctness["results"].append({
            "shape": config,
            "dtype": str(dtype),
            "correctness": False
        })

    return not mismatch

def _copy_parameters(source_module, target_module):
    """
    Copy parameters from source PyTorch module to target Triton module.

    Args:
        source_module (torch.nn.Module): Source PyTorch module to copy parameters from.
        target_module (torch.nn.Module): Target Triton module to copy parameters to.

    Note:
        Both modules must have the same parameter names and shapes.
        The copy is performed in-place on the target module's parameters.
    """
    # Iterate over all parameters in the source module
    for name, source_param in source_module.named_parameters():
        # Retrieve the corresponding parameter in the target module by name
        target_param = target_module.get_parameter(name)
        # Copy the data from source to target (in-place)
        target_param.data.copy_(source_param.data)

def _backward_pass(module, inputs, dy):
    """
    Perform backward pass through a module with given inputs and gradients.

    Args:
        module (torch.nn.Module): Neural network module to perform backward pass through.
        inputs (torch.Tensor | Tuple[torch.Tensor]): Input tensor(s) to the module.
        dy (torch.Tensor | Tuple[torch.Tensor]): Gradient tensor(s) for backward pass.

    Returns:
        Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]: Two tuples containing:
            1. Gradients with respect to inputs (dx)
            2. Gradients with respect to module parameters (dw)
    """
    detached_inputs = tuple(x.clone().detach().requires_grad_() for x in inputs) if isinstance(inputs, tuple) else (inputs.clone().detach().requires_grad_(),)
    outputs = module(*detached_inputs)

    if isinstance(outputs, tuple):
        for t_out, t_dy in zip(outputs, dy):
            t_out.backward(t_dy, retain_graph=True)
    else:
        outputs.backward(dy)

    dx, dw = tuple([x.grad for x in detached_inputs]), tuple([p.grad for p in module.parameters()])
    return dx, dw

def _benchmark_implementations(config, triton_fn, torch_fn, inputs, timings_triton, timings_pytorch, speedups, performance, num_warmup=25, num_repeats=100, device='cuda', dtype=torch.float16):
    """
    Benchmark Triton and PyTorch implementations and measure execution times.

    Args:
        config (List[int]): Current configuration being tested (e.g., [batch_size, input_dim, output_dim]).
        triton_fn (Callable): Triton implementation to benchmark.
        torch_fn (Callable): PyTorch implementation to benchmark.
        inputs (torch.Tensor | Tuple[torch.Tensor]): Input tensor(s) for the functions.
        timings_triton (List[float]): List to store Triton execution times.
        timings_pytorch (List[float]): List to store PyTorch execution times.
        speedups (List[float]): List to store speedup ratios (PyTorch/Triton).
        performance (Dict): Dictionary to store detailed performance results.
        num_warmup (int, optional): Number of warmup iterations. Defaults to 25.
        num_repeats (int, optional): Number of timing iterations. Defaults to 100.
        device (str, optional): Device to run on. Defaults to 'cuda'.
        dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float16.

    Returns:
        Tuple[float, float, float]: Tuple containing:
            1. Triton execution time (ms)
            2. PyTorch execution time (ms)
            3. Speedup ratio (PyTorch/Triton)
    """
    quantiles = [0.5]

    # Ensure inputs is a tuple
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    # Time Triton implementation
    time_triton = triton.testing.do_bench(
        lambda: triton_fn(inputs),
        warmup=num_warmup,
        rep=num_repeats,
        quantiles=quantiles
    ) * 1e3

    # Time PyTorch implementation
    time_pytorch = triton.testing.do_bench(
        lambda: torch_fn(inputs),
        warmup=num_warmup,
        rep=num_repeats,
        quantiles=quantiles
    ) * 1e3

    timings_triton.append(time_triton)
    timings_pytorch.append(time_pytorch)

    speedup = time_pytorch / time_triton
    speedups.append(speedup)

    print(f"Shape: ({config}, {dtype}), Triton: {time_triton:.1f} us, PyTorch: {time_pytorch:.1f} us, Speedup: {speedup:.2f}x")

    performance["results"].append({
        "shape": config,
        "dtype": str(dtype),
        "triton_time": time_triton,
        "pytorch_time": time_pytorch,
        "speedup": speedup
    })

    return time_triton, time_pytorch, speedup

def _plot_benchmark_results(configs, timings_triton, timings_pytorch, speedups, label=""):
    """
    Plot performance comparison between Triton and PyTorch implementations.

    Args:
        configs (List[List[int]]): List of configurations tested.
        timings_triton (List[float]): List of Triton execution times.
        timings_pytorch (List[float]): List of PyTorch execution times.
        speedups (List[float]): List of speedup ratios (PyTorch/Triton).
        label (str, optional): Label for the plot. Defaults to "".

    Note:
        Saves the plot to 'code.performance.png' and includes:
        - Speedup ratio for each configuration
        - Break-even line at y=1
        - Grid and legend
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    x_labels = [f"({config})" for config in configs]
    x = np.arange(len(x_labels))
    width = 0.35

    # Plot timing bars
    ax1.bar(x - width/2, timings_triton, width, label='Triton', color='blue')
    ax1.bar(x + width/2, timings_pytorch, width, label='PyTorch', color='red')

    ax1.set_ylabel('Time (us)')
    ax1.set_xlabel('Tensor Shape')
    ax1.set_title('Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=45)
    ax1.legend(loc='upper left')

    # Plot speedup line
    ax2 = ax1.twinx()
    ax2.plot(x, speedups, label='Speedup (PyTorch / Triton)', color='green', marker='o')
    ax2.set_ylabel('Speedup (x)')
    ax2.legend(loc='upper right')

    filename = f"code.performance.{label}.png"

    fig.tight_layout()
    plt.savefig(filename)
    print(f"Benchmark graph saved to {filename}")
    plt.close(fig)
