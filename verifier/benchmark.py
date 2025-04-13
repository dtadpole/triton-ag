# filename: verifier/benchmark.py

import torch
import numpy as np
import json
from ._verifier_util import (
    _generate_configs,
    _backward_pass,
    _benchmark_implementations,
    _plot_benchmark_results,
)


def benchmark_performance_func(
    func_triton,
    func_torch,
    input_generator,
    dimensions,
    num_warmup=25,
    num_repeats=100,
    device="cuda",
    dtype=torch.float16,
):
    """
    Benchmarks the performance of the Triton implementation against PyTorch.
    """
    torch.manual_seed(0)
    configs = _generate_configs(dimensions)
    pytorch_compiled = torch.compile(func_torch, mode="max-autotune")

    timings_triton = []
    timings_pytorch = []
    speedups = []
    device_name = torch.cuda.get_device_name(0)

    performance = {"speedup": 0.0, "device": device_name, "results": []}

    for config in configs:
        inputs = input_generator(*config, device=device, dtype=dtype)

        _benchmark_implementations(
            config,
            lambda inputs: (
                func_triton(*inputs)
                if isinstance(inputs, tuple)
                else func_triton(inputs)
            ),
            lambda inputs: (
                pytorch_compiled(*inputs)
                if isinstance(inputs, tuple)
                else pytorch_compiled(inputs)
            ),
            inputs,
            timings_triton,
            timings_pytorch,
            speedups,
            performance,
            num_warmup=num_warmup,
            num_repeats=num_repeats,
        )

    performance["speedup"] = np.median(speedups)

    # write performance results to file
    with open("code.performance.func.json", "w") as f:
        f.write(json.dumps(performance, indent=4))

    # Generate performance comparison graph
    _plot_benchmark_results(
        configs, timings_triton, timings_pytorch, speedups, label="func"
    )

    return (
        np.median(timings_triton),
        np.median(timings_pytorch),
        np.median(speedups),
        performance,
    )


def benchmark_performance_forward(
    module_generator,
    forward_generator,
    dimensions,
    num_warmup=25,
    num_repeats=100,
    device="cuda",
    dtype=torch.float16,
):
    """
    Benchmarks the performance of forward pass
    """
    configs = _generate_configs(dimensions)

    timings_triton = []
    timings_pytorch = []
    speedups = []
    device_name = torch.cuda.get_device_name(0)

    performance = {"speedup": 0.0, "device": device_name, "results": []}

    for config in configs:
        triton_module, torch_module = module_generator(
            *config, device=device, dtype=dtype
        )
        triton_compiled = torch.compile(triton_module, mode="max-autotune")
        pytorch_compiled = torch.compile(torch_module, mode="max-autotune")

        inputs = forward_generator(*config, device=device, dtype=dtype)

        _benchmark_implementations(
            config,
            lambda inputs: (
                triton_compiled(*inputs)
                if isinstance(inputs, tuple)
                else triton_compiled(inputs)
            ),
            lambda inputs: (
                pytorch_compiled(*inputs)
                if isinstance(inputs, tuple)
                else pytorch_compiled(inputs)
            ),
            inputs,
            timings_triton,
            timings_pytorch,
            speedups,
            performance,
            num_warmup=num_warmup,
            num_repeats=num_repeats,
        )

    performance["speedup"] = np.median(speedups)

    # write performance results to file
    with open("code.performance.forward.json", "w") as f:
        f.write(json.dumps(performance, indent=4))

    # Generate performance comparison graph
    _plot_benchmark_results(
        configs, timings_triton, timings_pytorch, speedups, label="forward"
    )

    return (
        np.median(timings_triton),
        np.median(timings_pytorch),
        np.median(speedups),
        performance,
    )


def benchmark_performance_backward(
    module_generator,
    input_generator,
    dimensions,
    num_warmup=25,
    num_repeats=100,
    device="cuda",
    dtype=torch.float16,
):
    """
    Benchmarks the performance of backward pass
    """
    configs = _generate_configs(dimensions)

    timings_triton = []
    timings_pytorch = []
    speedups = []
    device_name = torch.cuda.get_device_name(0)

    performance = {"speedup": 0.0, "device": device_name, "results": []}

    for config in configs:
        triton_module, torch_module = module_generator(
            *config, device=device, dtype=dtype
        )
        triton_compiled = torch.compile(triton_module, mode="max-autotune")
        pytorch_compiled = torch.compile(torch_module, mode="max-autotune")

        inputs, dy = input_generator(*config, device=device, dtype=dtype)

        _benchmark_implementations(
            config,
            lambda x: _backward_pass(triton_compiled, x[0], x[1]),
            lambda x: _backward_pass(pytorch_compiled, x[0], x[1]),
            (inputs, dy),
            timings_triton,
            timings_pytorch,
            speedups,
            performance,
            num_warmup=num_warmup,
            num_repeats=num_repeats,
        )

    performance["speedup"] = np.median(speedups)

    # write performance results to file
    with open("code.performance.backward.json", "w") as f:
        f.write(json.dumps(performance, indent=4))

    # Generate performance comparison graph
    _plot_benchmark_results(
        configs, timings_triton, timings_pytorch, speedups, label="backward"
    )

    return (
        np.median(timings_triton),
        np.median(timings_pytorch),
        np.median(speedups),
        performance,
    )
