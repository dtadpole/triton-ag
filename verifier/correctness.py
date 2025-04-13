# filename: verifier/correctness.py

import torch
import json
from ._verifier_util import (
    _generate_configs,
    _check_outputs_match,
    _copy_parameters,
    _backward_pass,
)


def verify_correctness_func(
    func_triton,
    func_torch,
    input_generator,
    dimensions,
    device="cuda",
    dtype=torch.float16,
    atol=2e-2,
    rtol=2e-2,
    seed=None,
):
    """
    Verify the function correctness of triton implementation against torch implementation
    """
    if seed is not None:
        torch.manual_seed(seed)
    device_name = torch.cuda.get_device_name(0)

    correctness = {
        "passed": 0,
        "failed": 0,
        "total": 0,
        "device": device_name,
        "results": [],
    }

    for config in _generate_configs(dimensions):
        inputs = input_generator(*config, device=device, dtype=dtype)
        triton_output = (
            func_triton(*inputs) if isinstance(inputs, tuple) else func_triton(inputs)
        )
        torch_output = (
            func_torch(*inputs) if isinstance(inputs, tuple) else func_torch(inputs)
        )
        _check_outputs_match(
            config,
            correctness,
            triton_output,
            torch_output,
            dtype,
            atol=atol,
            rtol=rtol,
            label="func",
        )

    # write correctness results to file
    with open("code.correctness.func.json", "w") as f:
        f.write(json.dumps(correctness, indent=4))

    return correctness


def verify_correctness_forward(
    module_generator,
    input_generator,
    dimensions,
    device="cuda",
    dtype=torch.float16,
    atol=2e-2,
    rtol=2e-2,
    seed=None,
):
    """
    Verify the forward pass
    """
    if seed is not None:
        torch.manual_seed(seed)
    device_name = torch.cuda.get_device_name(0)

    correctness = {
        "passed": 0,
        "failed": 0,
        "total": 0,
        "device": device_name,
        "results": [],
    }

    for config in _generate_configs(dimensions):
        triton_module, torch_module = module_generator(
            *config, device=device, dtype=dtype
        )
        _copy_parameters(torch_module, triton_module)

        inputs = input_generator(*config, device=device, dtype=dtype)
        triton_output = (
            triton_module(*inputs)
            if isinstance(inputs, tuple)
            else triton_module(inputs)
        )
        torch_output = (
            torch_module(*inputs) if isinstance(inputs, tuple) else torch_module(inputs)
        )
        _check_outputs_match(
            config,
            correctness,
            triton_output,
            torch_output,
            dtype,
            atol=atol,
            rtol=rtol,
            label="forward",
        )

    # write correctness results to file
    with open("code.correctness.forward.json", "w") as f:
        f.write(json.dumps(correctness, indent=4))

    return correctness


def verify_correctness_backward(
    module_generator,
    backward_generator,
    dimensions,
    device="cuda",
    dtype=torch.float16,
    atol=2e-2,
    rtol=2e-2,
    seed=None,
):
    """
    Verify the backward pass
    """
    if seed is not None:
        torch.manual_seed(seed)
    device_name = torch.cuda.get_device_name(0)

    correctness = {
        "passed": 0,
        "failed": 0,
        "total": 0,
        "device": device_name,
        "results": [],
    }

    for config in _generate_configs(dimensions):
        # input generator returns params (for Module initialization) and inputs (for Module forward)
        triton_module, torch_module = module_generator(
            *config, device=device, dtype=dtype
        )
        _copy_parameters(torch_module, triton_module)

        inputs, dy = backward_generator(*config, device=device, dtype=dtype)

        dx_triton, dw_triton = _backward_pass(triton_module, inputs, dy)
        dx_torch, dw_torch = _backward_pass(torch_module, inputs, dy)

        # Compare gradients
        _check_outputs_match(
            config,
            correctness,
            dx_triton,
            dx_torch,
            dtype,
            atol=atol,
            rtol=rtol,
            label="backward dx",
        )
        _check_outputs_match(
            config,
            correctness,
            dw_triton,
            dw_torch,
            dtype,
            atol=atol,
            rtol=rtol,
            label="backward dw",
        )

    # write correctness results to file
    with open("code.correctness.backward.json", "w") as f:
        f.write(json.dumps(correctness, indent=4))

    return correctness
