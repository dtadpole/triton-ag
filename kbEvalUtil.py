import socket
import traceback
import yaml
import torch
import torch.nn as nn
from pydantic import BaseModel
import hashlib
import os
import fcntl
import time
import psutil
from logger import logger
import numpy as np
import sys
import subprocess
import ast
import re
import signal
import argparse
from collections import defaultdict
from typing import Optional


MAX_LOCK_AGE = 20 # seconds

def on_critical_alarm(signum, frame):
    # logger.error(f"⏰ Critical timeout reached [{signum}] [{frame.f_code.co_name}], exiting.")
    logger.error(f"⏰ Critical alarm invoked")
    logger.error(f"⏰ [signal={signum}] [function={frame.f_code.co_name}] [file={frame.f_code.co_filename}:{frame.f_lineno}]")
    # kill the process
    os.kill(os.getpid(), signal.SIGKILL)
    # sleep for 1 second
    time.sleep(1)
    # exit with code 5 to indicate timer expired
    os._exit(5)

def on_critical_timeout():
    logger.error(f"⏰ Critical timeout invoked")
    os.kill(os.getpid(), signal.SIGKILL)
    # sleep for 1 second
    time.sleep(1)
    # exit with code 5 to indicate timer expired
    os._exit(5)

def on_process_timeout():
    logger.error(f"⛔ Process timeout invoked, exiting.")
    os.kill(os.getpid(), signal.SIGKILL)
    # sleep for 1 second
    time.sleep(1)
    # exit with code 6 to indicate process timeout
    os._exit(6)


def from_kbEval_yaml(yaml_file: str="kbEval.yaml"):
    with open(yaml_file, 'r') as f:
        yaml_data = yaml.safe_load(f)
    # get my hostname
    hostname = socket.gethostname()
    return yaml_data.get("kbEvalCli", {}).get(hostname, {})

def format_exception(e: Exception):
    # return "\n".join(traceback.format_exception(type(e), e, e.__traceback__)
    message = f"{type(e).__name__}: {e}"
    if e.__cause__ is not None or e.__context__ is not None:
        # return message + "\n  caused by: " + format_exception(e.__cause__ or e.__context__)
        return format_exception(e.__cause__ or e.__context__)
    else:
        return message
    # return f"{type(e).__name__}: {e.message}"

def set_seed(seed: int):
    torch.manual_seed(seed)
    # NOTE: this only sets on current cuda device
    torch.cuda.manual_seed(seed)


class KernelExecResult(BaseModel):
    """
    Kernel Execution Result
    """
    compiled: bool = False
    correctness: bool = False
    runtime: float = -1.0  # in us, only recorded if we decide to measure performance
    metadata: dict = {}
    runtime_stats: dict = {}  # only recorded if we decide to measure performance

class CorrectnessResult(BaseModel):
    trials: str = "unknown"
    total_trials: int = 0
    passed_trials: int = 0
    output_shape: str = "unknown"
    max_diff: float = -1.0
    avg_diff: float = -1.0


class CompileError(Exception):
    def __init__(self, message: str | None = None) -> None:
        self.message = message or f"Could not compile code"
        super().__init__(self.message)          # pass text to base class

class CompileSyntaxError(CompileError):
    def __init__(self, message: str | None = None) -> None:
        self.message = message or f"Syntax error in code"
        super().__init__(self.message)          # pass text to base class

class CompileLoadError(CompileError):
    def __init__(self, message: str | None = None) -> None:
        self.message = message or f"Could not load code"
        super().__init__(self.message)          # pass text to base class

class CompileMissingComponentError(CompileError):
    def __init__(self, message: str | None = None) -> None:
        self.message = message or f"Could not find component in code"
        super().__init__(self.message)          # pass text to base class

class CompileResolveComponentError(CompileError):
    def __init__(self, message: str | None = None) -> None:
        self.message = message or f"Could not find component in code"
        super().__init__(self.message)          # pass text to base class

class CompileModifiedComponentError(CompileError):
    def __init__(self, message: str | None = None) -> None:
        self.message = message or f"Component has been modified"
        super().__init__(self.message)          # pass text to base class

class CompileInstantiationError(CompileError):
    def __init__(self, message: str | None = None) -> None:
        self.message = message or f"Could not instantiate custom model"
        super().__init__(self.message)          # pass text to base class

class CompileRuntimeError(CompileError):
    def __init__(self, message: str | None = None) -> None:
        self.message = message or f"Could not run custom model"
        super().__init__(self.message)          # pass text to base class

class CorrectnessError(Exception):
    def __init__(self, message: str | None = None) -> None:
        self.message = message or f"Correctness error"
        super().__init__(self.message)          # pass text to base class

class CorrectnessShapeMismatchError(CorrectnessError):
    def __init__(self, message: str | None = None) -> None:
        self.message = message or f"Correctness shape mismatch"
        super().__init__(self.message)          # pass text to base class

class CorrectnessValueMismatchError(CorrectnessError):
    def __init__(self, message: str | None = None) -> None:
        self.message = message or f"Correctness value mismatch"
        super().__init__(self.message)          # pass text to base class

class CorrectnessProcessingError(CorrectnessError):
    def __init__(self, message: str | None = None) -> None:
        self.message = message or f"Correctness processing error"
        super().__init__(self.message)          # pass text to base class


def _compile_and_load_model(model_src: str, context: dict, filename: str = "<string>"):
    """
    Load class from original NN.module pytorch code
    this is pytorch reference and we feed that to model to see if there will be any improvement
    """
    try:
        code = compile(model_src, filename, "exec")
        # print(code)
    except SyntaxError as e:
        raise CompileSyntaxError(str(e)) from e

    try:
        exec(code, context)  # expose to current namespace
    except Exception as e:
        raise CompileLoadError(str(e)) from e

    return context

def load_model_and_inputs(
    model_src: str, context: dict, filename: str = "<string>"
) -> tuple[nn.Module, callable, callable]:
    """
    Load class from original NN.module pytorch code
    this is pytorch reference and we feed that to model to see if there will be any improvement
    """
    _compile_and_load_model(model_src, context, filename)

    # check "Model" exists in the context
    Model = context.get("Model")
    if not Model:
        raise CompileMissingComponentError("class [Model] not found")
    elif not isinstance(Model, type):
        raise CompileMissingComponentError(f"class [Model] is not a class, but {type(Model)}")
    elif not issubclass(Model, nn.Module):
        raise CompileMissingComponentError(f"class [Model] is not a subclass of nn.Module, but {type(Model)}")

    # check "get_init_inputs" exists in the context
    get_init_inputs_fn = context.get("get_init_inputs")
    if not get_init_inputs_fn:
        raise CompileMissingComponentError("function [get_init_inputs] not found")
    elif not callable(get_init_inputs_fn):
        raise CompileMissingComponentError(f"function [get_init_inputs] is not callable, but {type(get_init_inputs_fn)}")

    # check "get_inputs" exists in the context
    get_inputs_fn = context.get("get_inputs")
    if not get_inputs_fn:
        raise CompileMissingComponentError("function [get_inputs] not found")
    elif not callable(get_inputs_fn):
        raise CompileMissingComponentError(f"function [get_inputs] is not callable, but {type(get_inputs_fn)}")

    # return the model class, get_init_inputs function, and get_inputs function
    return Model, get_init_inputs_fn, get_inputs_fn

def load_custom_model(
    model_custom_src: str, context: dict, build_directory: str = None, filename: str = "<string>", code_type: str = "triton", check_get_inputs: bool = True,
) -> nn.Module:
    """
    Load class from custom NN.module pytorch code
    this is the code output by LLM with calls to custom cuda kernels
    """
    if code_type == "triton":
        # nothing to do for triton
        pass
    elif code_type == "cuda":
        # TODO: add custom CUDA handling here
        if build_directory:
            context["BUILD_DIRECTORY"] = build_directory
            # Add import at the start of the source code
            model_custom_src = (
                "import os\n" f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_directory}'\n"
            ) + model_custom_src
    else:
        # TODO: add other code types handling here
        raise ValueError(f"Invalid code type: {code_type}")

    # preserve a copy of the original model and inputs functions
    original_Model = context.get("Model")
    original_get_init_inputs_fn = context.get("get_init_inputs")
    original_get_inputs_fn = context.get("get_inputs")
    if not original_Model or not original_get_init_inputs_fn or not original_get_inputs_fn:
        raise CompileMissingComponentError("class [Model] or function [get_init_inputs] or function [get_inputs] not found in context")

    _compile_and_load_model(model_custom_src, context, filename)


    # Redefine the eq of two function objects. If the get_init_inputs and get_inputs have the same format, they should be equal
    # The old comparison won't be equal event if the generated code has identical definition as reference code for the two functions
    def compare_functions_objects(func1, func2):
        return (
            func1.__code__.co_code == func2.__code__.co_code and
            func1.__code__.co_names == func2.__code__.co_names and
            func1.__code__.co_varnames == func2.__code__.co_varnames
        )

    # check if any of the original components have been modified
    afterwards_Model = context.get("Model")
    afterwards_get_init_inputs_fn = context.get("get_init_inputs")
    afterwards_get_inputs_fn = context.get("get_inputs")
    if afterwards_Model != original_Model:
        raise CompileModifiedComponentError("class [Model] has been modified")
    if check_get_inputs is True:
        if compare_functions_objects(afterwards_get_init_inputs_fn,original_get_init_inputs_fn) is False:
            raise CompileModifiedComponentError("function [get_init_inputs] has been modified")
        if compare_functions_objects(afterwards_get_inputs_fn, original_get_inputs_fn) is False:
            raise CompileModifiedComponentError("function [get_inputs] has been modified")

    # check "ModelNew" exists in the context
    ModelNew = context.get("ModelNew")
    if not ModelNew:
        raise CompileMissingComponentError("class [ModelNew] not found")
    elif not isinstance(ModelNew, type):
        raise CompileMissingComponentError(f"class [ModelNew] is not a class, but {type(ModelNew)}")
    elif not issubclass(ModelNew, nn.Module):
        raise CompileMissingComponentError(f"class [ModelNew] is not a subclass of nn.Module, but {type(ModelNew)}")

    # return the ModelNew class
    return ModelNew

def get_timing_stats(elapsed_times: list[float], device: torch.device = None) -> dict:
    """Get timing statistics from a list of elapsed times.

    Args:
        elapsed_times: List of elapsed times in milliseconds
        device: CUDA device, record device info
    Returns:
        Dict containing mean, std, min, max and num_trials
        all timing are in ms
    """

    stats = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
    }

    if device:
        stats["hardware"] = torch.cuda.get_device_name(device=device)
        stats["device"] = str(device)  # for debugging

    return stats

def time_execution_with_cuda_event(
    kernel_fn: callable,
    *args,
    num_warmups: int = 25,
    num_trials: int = 100,
    device: torch.device = None,
) -> list[float]:
    """
    Time a CUDA kernel function over multiple trials using torch.cuda.Event

    Args:
        kernel_fn: Function to time
        *args: Arguments to pass to kernel_fn
        num_trials: Number of timing trials to run
        device: CUDA device to use, if None, use current device

    Returns:
        List of elapsed times in milliseconds
    """
    if device is None:
        device = torch.cuda.current_device()

    elapsed_times = []
    # stream = torch.cuda.Stream(device=device) # comment out to use default stream
    with torch.cuda.device(device):
        # Warm ups
        for _ in range(num_warmups):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            # use exact same device as the actual trials
            start_event.record()
            kernel_fn(*args)
            end_event.record()
            # Synchronize to ensure the events have completed
            torch.cuda.synchronize(device=device)

        # Actual trials
        for trial in range(num_trials):
            # create event marker default is not interprocess
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            # start_event.device = device
            # end_event.device = device
            start_event.record()
            kernel_fn(*args)
            end_event.record()
            # Synchronize to ensure the events have completed
            torch.cuda.synchronize(device=device)
            # end_event.synchronize()

            # Calculate the elapsed time in milliseconds
            elapsed_time_ms = start_event.elapsed_time(end_event)
            # print(f"Trial {trial} elapsed_time_ms: {elapsed_time_ms}")
            elapsed_times.append(elapsed_time_ms)

    return elapsed_times

def graceful_eval_cleanup(curr_context: dict, device: torch.device):
    """
    Clean up env, gpu cache, and compiled CUDA extensions after evaluation
    """  # delete ran-specific function definitions before next eval run
    del curr_context
    # Clear CUDA cache and reset GPU state
    if device is not None:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

            # does this help?
            torch.cuda.reset_peak_memory_stats(device=device)

            torch.cuda.synchronize(
                device=device
            )  # Wait for all CUDA operations to complete

    # _cleanup_cuda_extensions() # SIMON NOTE: is this necessary?

class FunctionCallMapper(ast.NodeVisitor):
    def __init__(self):
        self.scope_stack = ['__global__']  # Tracks class/function nesting
        self.call_graph = defaultdict(set)  # caller → set of callees
        self.defined_functions = {}  # name → fully qualified name (e.g., helper → MyClass.helper)
        self.triton_jit_functions = set()
        self.model_new_class_count = 0
        self.model_new_forward_count = 0
        print(f"Processing Function Call Mapper [{self._current_scope()}]...")

    def _current_scope(self):
        return ".".join(self.scope_stack)

    def visit_ClassDef(self, node):
        print(f'{"  "*len(self.scope_stack)} Inside class: [{".".join(self.scope_stack + [node.name])}]')
        self.scope_stack.append(node.name)
        if node.name == "ModelNew":
            self.model_new_class_count += 1
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node):
        print(f'{"  "*len(self.scope_stack)} Inside function: [{".".join(self.scope_stack + [node.name])}]')
        full_name = ".".join(self.scope_stack + [node.name])
        self.defined_functions[node.name] = full_name  # record mapping for function lookup
        self.scope_stack.append(node.name)
        if node.name == "forward" and len(self.scope_stack) >= 2 and self.scope_stack[-2] == "ModelNew":
            self.model_new_forward_count += 1
        for dec in node.decorator_list:
            if isinstance(dec, ast.Attribute) and \
            isinstance(dec.value, ast.Name) and \
            dec.value.id == 'triton' and dec.attr == 'jit':
                self.triton_jit_functions.add(full_name)
                print(f'{"  "*len(self.scope_stack)} Function [{full_name}] is [@triton.jit]')
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_AsyncFunctionDef(self, node):
        print(f'{"  "*len(self.scope_stack)} Inside async function: [{".".join(self.scope_stack + [node.name])}]')
        full_name = ".".join(self.scope_stack + [node.name])
        self.defined_functions[node.name] = full_name
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_Call(self, node):
        func_name = self._get_call_name(node.func)
        caller = self._current_scope()
        if caller and func_name:
            # Upgrade to full qualified name if we know it
            qualified_callee = self.defined_functions.get(func_name, func_name)
            self.call_graph[caller].add(qualified_callee)
            print(f'{"  "*len(self.scope_stack)} Found function call: [{qualified_callee}]')
        self.generic_visit(node)

    def _get_call_name(self, node):
        """Extract base name from the called function expression."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            # e.g., self.helper → helper
            return node.attr
        elif isinstance(node, ast.Subscript):
            return self._get_call_name(node.value)
        elif isinstance(node, ast.Call):
            return self._get_call_name(node.func)
        return None

def resolve_custom_cuda_kernel(code: str):
    """
    Resolve the custom CUDA kernel code, check for valid kernel implementation
    and raise corresponding errors based on validation results.
    """
    validation_results = validate_custom_cuda_kernel(code)
    if not validation_results["has_kernel_function"]:
        raise CompileResolveComponentError("No __global__ kernel function found")
    if not validation_results["has_kernel_launch"]:
        raise CompileResolveComponentError("No kernel launch syntax (<<< >>>) found")
    if not validation_results["has_thread_indexing"]:
        raise CompileResolveComponentError("No CUDA thread indexing (threadIdx, blockIdx, etc.) found")
    if validation_results["library_shortcuts"]:
        shortcuts = ", ".join(validation_results["library_shortcuts"][:3])
        if len(validation_results["library_shortcuts"]) > 3:
            shortcuts += f" (and {len(validation_results['library_shortcuts']) - 3} more)"
        raise CompileResolveComponentError(f"Custom CUDA kernel uses torch or trensorrt library shortcuts: {shortcuts}, which is not allowed")
    if validation_results["has_heavy_pytorch_ops"]:
        raise CompileResolveComponentError(f"The code directly calls torch nn library, instead of writing CUDA kernel, which is not allowed")
    if not validation_results["is_valid_custom_kernel"]:
        raise CompileResolveComponentError("Invalid custom CUDA kernel implementation")
    logger.info(f"Custom CUDA kernel validation passed")
    return True

def check_triton_disallowed_nn_modules(code: str) -> list[str]:
    """
    Check for disallowed torch.nn module, functional API, and torch.* operation usage in Triton code.

    Allowed exceptions:
    - nn.Parameter / Parameter
    - Containers: nn.Module, nn.ModuleList, nn.ModuleDict, nn.Sequential, nn.ParameterList, nn.ParameterDict
    - Initialization: nn.init.*
    - Tensor creation: torch.empty, torch.zeros, torch.ones, torch.full, torch.arange, torch.linspace
    - Shape manipulation: view, reshape, permute, transpose, contiguous, cat, stack, split, chunk,
      squeeze, unsqueeze, expand, flatten
    - Type casting: .to(), .float(), .half()
    - Device operations: .cuda(), .to(device)

    Returns:
        List of found disallowed module/function usages
    """
    # Disallowed nn modules (heavy PyTorch operations)
    disallowed_nn_modules = [
        # Convolution Operations
        'nn.Conv1d', 'nn.Conv2d', 'nn.Conv3d',
        'nn.ConvTranspose1d', 'nn.ConvTranspose2d', 'nn.ConvTranspose3d',
        'nn.LazyConv1d', 'nn.LazyConv2d', 'nn.LazyConv3d',
        'nn.LazyConvTranspose1d', 'nn.LazyConvTranspose2d', 'nn.LazyConvTranspose3d',

        # Pooling Operations
        'nn.MaxPool1d', 'nn.MaxPool2d', 'nn.MaxPool3d',
        'nn.AvgPool1d', 'nn.AvgPool2d', 'nn.AvgPool3d',
        'nn.AdaptiveAvgPool1d', 'nn.AdaptiveAvgPool2d', 'nn.AdaptiveAvgPool3d',
        'nn.AdaptiveMaxPool1d', 'nn.AdaptiveMaxPool2d', 'nn.AdaptiveMaxPool3d',
        'nn.MaxUnpool1d', 'nn.MaxUnpool2d', 'nn.MaxUnpool3d',
        'nn.FractionalMaxPool2d', 'nn.FractionalMaxPool3d',
        'nn.LPPool1d', 'nn.LPPool2d',

        # Activation Functions
        'nn.ReLU', 'nn.LeakyReLU', 'nn.GELU', 'nn.SiLU', 'nn.Mish',
        'nn.Sigmoid', 'nn.Tanh', 'nn.Softmax', 'nn.LogSoftmax',
        'nn.ELU', 'nn.SELU', 'nn.PReLU', 'nn.Softplus', 'nn.Softsign',
        'nn.Hardtanh', 'nn.HardSigmoid', 'nn.Hardswish',
        'nn.ReLU6', 'nn.RReLU', 'nn.CELU', 'nn.GLU',
        'nn.Softmax2d', 'nn.Softmin', 'nn.Tanhshrink', 'nn.Softshrink', 'nn.Hardshrink',
        'nn.Threshold', 'nn.MultiheadAttention',

        # Normalization Operations
        'nn.BatchNorm1d', 'nn.BatchNorm2d', 'nn.BatchNorm3d',
        'nn.LayerNorm', 'nn.GroupNorm',
        'nn.InstanceNorm1d', 'nn.InstanceNorm2d', 'nn.InstanceNorm3d',
        'nn.LocalResponseNorm', 'nn.SyncBatchNorm',
        'nn.LazyBatchNorm1d', 'nn.LazyBatchNorm2d', 'nn.LazyBatchNorm3d',
        'nn.LazyInstanceNorm1d', 'nn.LazyInstanceNorm2d', 'nn.LazyInstanceNorm3d',
        'nn.RMSNorm',

        # Linear/Recurrent Operations
        'nn.Linear', 'nn.LazyLinear', 'nn.Bilinear',
        'nn.LSTM', 'nn.GRU', 'nn.RNN',
        'nn.LSTMCell', 'nn.GRUCell', 'nn.RNNCell',

        # Transformer Operations
        'nn.Transformer', 'nn.TransformerEncoder', 'nn.TransformerDecoder',
        'nn.TransformerEncoderLayer', 'nn.TransformerDecoderLayer',

        # Embedding Operations
        'nn.Embedding', 'nn.EmbeddingBag',

        # Dropout
        'nn.Dropout', 'nn.Dropout1d', 'nn.Dropout2d', 'nn.Dropout3d',
        'nn.AlphaDropout', 'nn.FeatureAlphaDropout',

        # Upsampling/Interpolation
        'nn.Upsample', 'nn.UpsamplingNearest2d', 'nn.UpsamplingBilinear2d',

        # Padding
        'nn.ReflectionPad1d', 'nn.ReflectionPad2d', 'nn.ReflectionPad3d',
        'nn.ReplicationPad1d', 'nn.ReplicationPad2d', 'nn.ReplicationPad3d',
        'nn.ZeroPad1d', 'nn.ZeroPad2d', 'nn.ZeroPad3d',
        'nn.ConstantPad1d', 'nn.ConstantPad2d', 'nn.ConstantPad3d',
        'nn.CircularPad1d', 'nn.CircularPad2d', 'nn.CircularPad3d',

        # Loss Functions
        'nn.CrossEntropyLoss', 'nn.NLLLoss', 'nn.MSELoss', 'nn.L1Loss',
        'nn.SmoothL1Loss', 'nn.BCELoss', 'nn.BCEWithLogitsLoss',
        'nn.KLDivLoss', 'nn.CosineEmbeddingLoss', 'nn.CTCLoss',
        'nn.HuberLoss', 'nn.PoissonNLLLoss', 'nn.GaussianNLLLoss',
        'nn.HingeEmbeddingLoss', 'nn.MarginRankingLoss', 'nn.MultiLabelMarginLoss',
        'nn.MultiLabelSoftMarginLoss', 'nn.MultiMarginLoss', 'nn.SoftMarginLoss',
        'nn.TripletMarginLoss', 'nn.TripletMarginWithDistanceLoss',

        # Fold/Unfold
        'nn.Fold', 'nn.Unfold',

        # Distance/Similarity
        'nn.CosineSimilarity', 'nn.PairwiseDistance',

        # Utility
        'nn.Flatten', 'nn.Unflatten',
        'nn.PixelShuffle', 'nn.PixelUnshuffle',
        'nn.ChannelShuffle',
    ]

    # Disallowed functional API operations (F.* — same cuDNN/cuBLAS kernels as nn.* modules)
    disallowed_functional = [
        # Convolution
        'F.conv1d', 'F.conv2d', 'F.conv3d',
        'F.conv_transpose1d', 'F.conv_transpose2d', 'F.conv_transpose3d',

        # Pooling
        'F.max_pool1d', 'F.max_pool2d', 'F.max_pool3d',
        'F.avg_pool1d', 'F.avg_pool2d', 'F.avg_pool3d',
        'F.adaptive_avg_pool1d', 'F.adaptive_avg_pool2d', 'F.adaptive_avg_pool3d',
        'F.adaptive_max_pool1d', 'F.adaptive_max_pool2d', 'F.adaptive_max_pool3d',
        'F.lp_pool1d', 'F.lp_pool2d',

        # Activation
        'F.relu', 'F.leaky_relu', 'F.gelu', 'F.silu', 'F.mish',
        'F.sigmoid', 'F.tanh', 'F.softmax', 'F.log_softmax',
        'F.elu', 'F.selu', 'F.prelu', 'F.softplus', 'F.softsign',
        'F.hardtanh', 'F.hardsigmoid', 'F.hardswish',
        'F.relu6', 'F.rrelu', 'F.celu', 'F.glu',
        'F.softmin', 'F.tanhshrink', 'F.softshrink', 'F.hardshrink',
        'F.threshold',

        # Normalization
        'F.batch_norm', 'F.layer_norm', 'F.group_norm', 'F.instance_norm',
        'F.local_response_norm',

        # Linear
        'F.linear', 'F.bilinear',

        # Dropout
        'F.dropout', 'F.dropout1d', 'F.dropout2d', 'F.dropout3d',
        'F.alpha_dropout', 'F.feature_alpha_dropout',

        # Embedding
        'F.embedding', 'F.embedding_bag',

        # Loss
        'F.cross_entropy', 'F.nll_loss', 'F.mse_loss', 'F.l1_loss',
        'F.smooth_l1_loss', 'F.binary_cross_entropy', 'F.binary_cross_entropy_with_logits',
        'F.kl_div', 'F.cosine_embedding_loss', 'F.ctc_loss',
        'F.huber_loss', 'F.poisson_nll_loss', 'F.gaussian_nll_loss',
        'F.hinge_embedding_loss', 'F.margin_ranking_loss', 'F.multi_margin_loss',
        'F.multilabel_margin_loss', 'F.multilabel_soft_margin_loss',
        'F.soft_margin_loss', 'F.triplet_margin_loss', 'F.triplet_margin_with_distance_loss',

        # Upsampling/Interpolation
        'F.interpolate', 'F.upsample',

        # Fold/Unfold
        'F.fold', 'F.unfold',

        # Padding
        'F.pad',

        # Attention
        'F.scaled_dot_product_attention',
        'F.multi_head_attention_forward',
    ]

    # Disallowed torch.* operations (cuBLAS, cuDNN, or heavy compute)
    disallowed_torch_ops = [
        # cuBLAS matmul operations
        'torch.matmul', 'torch.mm', 'torch.bmm',
        'torch.addmm', 'torch.addmv', 'torch.baddbmm',
        'torch.einsum',

        # Activations (call same CUDA kernels as F.* versions)
        'torch.relu', 'torch.sigmoid', 'torch.tanh', 'torch.softmax',

        # Internal cuDNN RNN ops
        'torch._VF',
    ]

    found_disallowed = []
    for module in disallowed_nn_modules:
        if module in code:
            found_disallowed.append(module)

    for func in disallowed_functional:
        if func in code:
            found_disallowed.append(func)

    for op in disallowed_torch_ops:
        if op in code:
            found_disallowed.append(op)

    return found_disallowed


def resolve_triton_code(code: str):
    """
    Resolve the triton code, check there is function call from ModelNew.forward to @triton.jit function(s)
    """
    # check if the code is valid
    try:
        tree = ast.parse(code)
        # print(ast.dump(tree, indent=2))
    except Exception as e:
        raise CompileSyntaxError(str(e)) from e

    mapper = FunctionCallMapper()
    mapper.visit(tree)
    if len(mapper.call_graph) == 0:
        raise CompileResolveComponentError("No function calls found")
    if len(mapper.triton_jit_functions) == 0:
        raise CompileResolveComponentError("No @triton.jit functions found")
    logger.info(f"Found [{len(mapper.triton_jit_functions)}] @triton.jit functions: {mapper.triton_jit_functions}")
    # check that ModelNew is defined as a class, and only one ModelNew is defined
    # check that ModelNew has a forward method, and only one forward method is defined
    if mapper.model_new_class_count == 0:
        raise CompileResolveComponentError("ModelNew is not defined as a class")
    if mapper.model_new_class_count != 1:
        raise CompileResolveComponentError(f"ModelNew is defined multiple times: [{mapper.model_new_class_count}]")
    logger.info(f"Class [ModelNew] is defined [{mapper.model_new_class_count}] time(s)")
    if mapper.model_new_forward_count == 0:
        raise CompileResolveComponentError("ModelNew does not have a forward method")
    if mapper.model_new_forward_count != 1:
        raise CompileResolveComponentError(f"ModelNew has multiple forward methods: [{mapper.model_new_forward_count}]")
    logger.info(f"Method [ModelNew.forward] is defined [{mapper.model_new_forward_count}] time(s)")

    # Check for disallowed nn module usage (heavy PyTorch operations)
    # Allowed: nn.Parameter, nn.Module containers (ModuleList, ModuleDict, Sequential, ParameterList, ParameterDict), nn.init
    disallowed_modules = check_triton_disallowed_nn_modules(code)
    if disallowed_modules:
        modules_str = ", ".join(disallowed_modules[:5])
        if len(disallowed_modules) > 5:
            modules_str += f" (and {len(disallowed_modules) - 5} more)"
        raise CompileResolveComponentError(
            f"Triton code uses disallowed torch.nn modules: {modules_str}. "
            "Only nn.Parameter, nn.Module containers, and nn.init are allowed."
        )
    logger.info(f"No disallowed torch.nn modules found in Triton code")

    # === Reward hacking detection checks ===

    # Check for getattr(nn, ...) bypass for disallowed modules
    getattr_nn_pattern = re.compile(r'getattr\s*\(\s*(?:nn|torch\.nn)\s*,')
    if getattr_nn_pattern.search(code):
        raise CompileResolveComponentError(
            "Code uses getattr(nn, ...) to bypass module restrictions. "
            "Use nn.Parameter + functional API instead."
        )

    # Check for torch.compile / torch.jit
    compile_pattern = re.compile(r'torch\.compile|torch\.jit\.script|torch\.jit\.trace')
    if compile_pattern.search(code):
        raise CompileResolveComponentError(
            "Code uses torch.compile or torch.jit which is not allowed. Write Triton kernels directly."
        )

    # Check for CUDA Graphs
    cuda_graph_pattern = re.compile(r'CUDAGraph|cuda\.graph\s*\(|graph\.replay\s*\(|cuda_graph')
    if cuda_graph_pattern.search(code):
        raise CompileResolveComponentError(
            "Code uses CUDA Graphs which is not allowed. Optimize with Triton kernels instead."
        )

    # Check for noop/identity Triton kernels (conservative: only catches pure load-store)
    def _check_triton_kernels_have_computation(source_tree):
        """Check that @triton.jit kernels have real computation, not just load/store."""
        for node in ast.walk(source_tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            # Check if this function has @triton.jit decorator
            is_triton_jit = False
            for dec in node.decorator_list:
                if isinstance(dec, ast.Attribute) and \
                   isinstance(dec.value, ast.Name) and \
                   dec.value.id == 'triton' and dec.attr == 'jit':
                    is_triton_jit = True
                    break
            if not is_triton_jit:
                continue
            # Dump the body AST and check for compute operations
            body_src = ast.dump(node)
            has_compute = bool(re.search(
                r'tl\.(dot|sum|max|min|exp|sigmoid|where|sqrt|log|abs|zeros|full|cdiv|math)',
                body_src
            ))
            has_binop = 'BinOp' in body_src
            if not has_compute and not has_binop:
                return False, node.name
        return True, None

    has_computation, noop_func_name = _check_triton_kernels_have_computation(tree)
    if not has_computation:
        raise CompileResolveComponentError(
            f"Triton kernel '{noop_func_name}' appears to be a noop/identity kernel "
            "(no arithmetic or compute operations found). Every @triton.jit kernel must "
            "perform meaningful computation."
        )

    # Check for reference Model(...) instantiation (not ModelNew)
    model_instantiation_pattern = re.compile(r'(?<!\w)Model\s*\(')
    for match in model_instantiation_pattern.finditer(code):
        line_num = code[:match.start()].count('\n')
        line_text = code.split('\n')[line_num]
        if 'class ' not in line_text and 'ModelNew' not in line_text:
            raise CompileResolveComponentError(
                "Code instantiates the reference Model class. Write your own implementation."
            )

    # Check for F.scaled_dot_product_attention
    if 'scaled_dot_product_attention' in code:
        raise CompileResolveComponentError(
            "Code uses F.scaled_dot_product_attention which delegates to Flash Attention. "
            "Write the attention computation in Triton."
        )

    # === End reward hacking detection checks ===

    # check ModelNew.forward calls at least one triton.jit function, using call_graph
    forward_method = "__global__.ModelNew.forward"
    if forward_method not in mapper.call_graph:
        raise CompileResolveComponentError(f"Forward method [{forward_method}] is not found in call graph")
    # recursivel wall through the call graph, check all function call names
    def get_function_call_recursive(func_name: str, parent_path: list[str] = [], visited: dict = {}) -> dict:
        if func_name in visited:
            return visited
        visited[func_name] = parent_path.copy()
        for callee in mapper.call_graph[func_name]:
            visited = get_function_call_recursive(callee, parent_path + [func_name], visited)
        return visited
    function_calls = get_function_call_recursive(forward_method)
    jit_triton_function_called = 0
    for func_name, path in function_calls.items():
        if func_name in mapper.triton_jit_functions:
            logger.info(f"Function [@triton.jit] [{func_name}] is called by [{' -> '.join(path)}]")
            jit_triton_function_called += 1
    if jit_triton_function_called == 0:
        raise CompileResolveComponentError("ModelNew.forward does not call any @triton.jit functions")

    # we are here if everything checks out
    return True

def get_nvcc_version() -> str:
    """Get NVCC compiler version."""
    try:
        result = subprocess.run(['nvcc', '--version'],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # Parse output like: "Cuda compilation tools, release 12.0, V12.0.76"
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    # Extract version like "12.0" from the line
                    parts = line.split('release')
                    if len(parts) > 1:
                        version_part = parts[1].split(',')[0].strip()
                        return version_part
            # Fallback: return full version output
            return result.stdout.strip().replace('\n', ' ')[:50]
        else:
            return f"nvcc_error_{result.returncode}"
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        return f"nvcc_unknown_{str(e)[:20]}"


def get_gpu_card_type() -> str:
    """Get GPU card type (e.g., A100, H100, V100)."""
    try:
        # Try nvidia-ml-py first (more reliable)
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Get first GPU
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            return name.split()[0][:20]
        except ImportError:
            # Fallback to nvidia-smi
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                gpu_name = result.stdout.strip().split('\n')[0]
                # Same normalization as above
                return gpu_name.split()[0][:20]
            else:
                return "gpu_unknown"
    except Exception as e:
        return f"gpu_error_{str(e)[:10]}"


def get_hostname() -> str:
    """Get system hostname."""
    try:
        hostname = socket.gethostname()
        return hostname[:30]
    except Exception as e:
        return f"host_unknown_{str(e)[:10]}"


def get_pytorch_version() -> str:
    """Get PyTorch version."""
    try:
        return torch.__version__
    except Exception as e:
        return f"torch_unknown_{str(e)[:10]}"


def get_compute_capability() -> str:
    """Get CUDA compute capability for the current GPU."""
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return f"{major}.{minor}"
        else:
            return "no_cuda"
    except Exception as e:
        return f"compute_unknown_{str(e)[:10]}"


def apply_black_formatter(generated_code: str) -> str:
    """
    Format the given Python code string using the 'black' code formatter.

    Args:
        generated_code: The Python code as a string.

    Returns:
        The formatted Python code as a string.

    Raises:
        RuntimeError: If black is not installed or formatting fails.
    """
    try:
        import black
        # Mode: use default Black mode (PEP8, line length 88)
        mode = black.FileMode()
        formatted_code = black.format_str(generated_code, mode=mode)
        return formatted_code
    except ImportError:
        raise RuntimeError("Black formatter is not installed. Please install with 'pip install black'.")
    except Exception as e:
        raise RuntimeError(f"Black formatting failed: {e}")


def generate_cache_hash(generated_code: str, file_path: str, hash_length: int=50) -> str:
    """
    Generate a deterministic cache hash for CUDA compilation.

    Args:
        generated_code: The generated CUDA code content
        file_path: Path to the generated code file

    Returns:
        A deterministic hash string suitable for use as a directory name
    """
    # Collect all hash components
    components = {
        'generated_code': apply_black_formatter(generated_code),
        'nvcc_version': get_nvcc_version(),
        'gpu_card_type': get_gpu_card_type(),
        # 'hostname': get_hostname(), # Remove hostname to leverage CPU from other hosts to compile the code
        'pytorch_version': get_pytorch_version(),
        'compute_capability': get_compute_capability(),
    }

    # Create a deterministic string representation
    hash_input_parts = []
    for key in sorted(components.keys()):  # Sort keys for deterministic order
        value = str(components[key])
        hash_input_parts.append(f"{key}:{value}")

    hash_input = '\n'.join(hash_input_parts)

    # Generate SHA256 hash and take first 16 characters for manageable directory names
    hash_obj = hashlib.sha256(hash_input.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()[:hash_length]

    return hash_hex


def get_cache_build_directory(generated_code: str, file_path: str,
                            shared_cache_parent: str = "/tmp/cuda_shared_cache") -> str:
    """
    Get the build directory path for caching based on content and system configuration.

    Args:
        generated_code: The generated CUDA code content
        file_path: Path to the generated code file
        shared_cache_parent: Parent directory for all cache folders

    Returns:
        Full path to the build directory for this specific configuration
    """
    cache_hash = generate_cache_hash(generated_code, file_path)
    build_dir = os.path.join(shared_cache_parent, cache_hash)

    # Create directory if it doesn't exist
    os.makedirs(build_dir, exist_ok=True)

    return build_dir


# ===== AOTI (Ahead-Of-Time Inductor) Compilation =====

def generate_aoti_cache_hash(
    model_code: str,
    hash_length: int = 50
) -> str:
    """
    Generate a deterministic cache hash for AOTI compilation.

    The cache key is based on:
    - Model source code (which includes get_inputs/get_init_inputs definitions)
    - GPU card type
    - PyTorch version
    - CUDA compute capability

    Note: Input shapes are NOT included because they are deterministic
    based on the get_inputs() function defined in the model code.

    Args:
        model_code: The model source code
        hash_length: Length of the hash string

    Returns:
        A deterministic hash string suitable for use as a directory/file name
    """
    components = {
        'model_code': apply_black_formatter(model_code),
        'hostname': get_hostname(),
        'pytorch_version': get_pytorch_version(),
        'compute_capability': get_compute_capability(),
    }

    hash_input_parts = []
    for key in sorted(components.keys()):
        value = str(components[key])
        hash_input_parts.append(f"{key}:{value}")

    hash_input = '\n'.join(hash_input_parts)

    hash_obj = hashlib.sha256(hash_input.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()[:hash_length]

    return hash_hex


def get_aoti_cache_path(
    model_code: str,
    shared_cache_parent: str = "shared/.kbeval/reference_cache"
) -> tuple[str, str]:
    """
    Get the AOTI cache directory and .pt2 file path.

    Args:
        model_code: The model source code
        shared_cache_parent: Parent directory for all AOTI cache folders

    Returns:
        Tuple of (cache_directory, pt2_file_path)
    """
    cache_hash = generate_aoti_cache_hash(model_code)

    # Convert to absolute path to ensure it works from any working directory
    # (PyTorch's aot_compile uses /tmp/torchinductor_root/ as working dir)
    cache_dir = os.path.abspath(os.path.join(shared_cache_parent, cache_hash))
    pt2_file_path = os.path.join(cache_dir, "model.pt2")

    os.makedirs(cache_dir, exist_ok=True)

    return cache_dir, pt2_file_path


def compile_model_aoti(
    model: nn.Module,
    example_inputs: tuple,
    cache_dir: str,
    pt2_file_path: str,
    device: torch.device = None,
) -> nn.Module:
    """
    Compile a PyTorch model using AOTI (Ahead-Of-Time Inductor).

    This function exports the model and compiles it to a .pt2 package file,
    then loads and returns the compiled model.

    Args:
        model: The PyTorch model instance to compile
        example_inputs: Example inputs for tracing (tuple of tensors)
        cache_dir: Directory to store compilation artifacts
        pt2_file_path: Path where the .pt2 package file will be saved
        device: CUDA device to use

    Returns:
        The AOTI-compiled model callable

    Note:
        This uses static shapes for compilation. The compiled model will
        only work with inputs of the exact same shape. This is appropriate
        for KernelBench where get_inputs() returns deterministic shapes
        for each model.
    """
    import torch._inductor
    from torch.export import export

    if device is None:
        device = torch.cuda.current_device()

    # Check if cached .pt2 file exists
    if os.path.exists(pt2_file_path):
        logger.info(f"🎯 Loading cached AOTI model from [{pt2_file_path}]")
        try:
            # Load AOTI model within the correct CUDA device context
            with torch.cuda.device(device):
                compiled_model = torch._inductor.aoti_load_package(pt2_file_path)
            return compiled_model
        except Exception as e:
            logger.warning(
                f"⚠️ Failed to load cached AOTI model: {e}. Recompiling..."
            )
            # Remove corrupted cache and recompile
            try:
                os.remove(pt2_file_path)
            except OSError:
                pass

    logger.info(f"🔨 Compiling model with AOTI to [{pt2_file_path}]")

    # Ensure model is in eval mode for export
    model.eval()

    try:
        # Export and compile within the correct CUDA device context
        with torch.cuda.device(device):
            # Export the model with static shapes
            with torch.no_grad():
                exported_model = export(model, example_inputs)

            # AOT compile and package to .pt2 file
            pt2_path = torch._inductor.aoti_compile_and_package(
                exported_model,
                package_path=pt2_file_path,
            )

            logger.info(f"✅ AOTI compilation successful: [{pt2_path}]")

            # Load the compiled model from .pt2 package within the same device context
            compiled_model = torch._inductor.aoti_load_package(pt2_path)

        return compiled_model

    except Exception as e:
        logger.error(f"❌ AOTI compilation failed: {e}")
        raise CompileError(f"AOTI compilation failed: {e}") from e


def get_or_compile_aoti_model(
    model: nn.Module,
    example_inputs: tuple,
    model_code: str,
    device: torch.device = None,
    shared_cache_parent: str = "shared/.kbeval/reference_cache",
) -> nn.Module:
    """
    Get a cached AOTI model or compile and cache it.

    This is the main entry point for AOTI compilation with caching.

    Args:
        model: The PyTorch model instance to compile
        example_inputs: Example inputs for tracing (tuple of tensors)
        model_code: The model source code (for cache key generation)
        device: CUDA device to use
        shared_cache_parent: Parent directory for AOTI cache

    Returns:
        The AOTI-compiled model callable
    """
    cache_dir, pt2_file_path = get_aoti_cache_path(
        model_code,
        shared_cache_parent,
    )

    logger.info(f"🔍 AOTI cache directory: [{cache_dir}]")

    return compile_model_aoti(
        model,
        example_inputs,
        cache_dir,
        pt2_file_path,
        device,
    )


def compile_model_torch_compile(
    model: nn.Module,
    mode: str = "reduce-overhead",
    backend: str = "inductor",
) -> nn.Module:
    """
    Compile a PyTorch model using torch.compile (JIT compilation).

    This is a simpler alternative to AOTI that doesn't cache to disk.

    Args:
        model: The PyTorch model instance to compile
        mode: Compilation mode. Options:
            - "default": Default mode
            - "reduce-overhead": Reduces overhead, good for small models
            - "max-autotune": Maximum autotuning (slower compile, faster run)
            - "max-autotune-no-cudagraphs": Max autotune without CUDA graphs
        backend: Backend to use (default: "inductor")

    Returns:
        The torch.compile'd model
    """
    logger.info(f"🔨 Compiling model with torch.compile (mode={mode})")

    try:
        compiled_model = torch.compile(model, mode=mode, backend=backend)
        logger.info(f"✅ torch.compile successful")
        return compiled_model
    except Exception as e:
        logger.error(f"❌ torch.compile failed: {e}")
        raise CompileError(f"torch.compile failed: {e}") from e


def print_cache_info(generated_code: str, file_path: str) -> None:
    """
    Print detailed information about cache hash components for debugging.
    """
    print("Cache Hash Components:")
    print("=" * 50)

    components = {
        'file_path': os.path.abspath(file_path),
        'generated_code_length': len(generated_code),
        'generated_code_hash': hashlib.md5(generated_code.encode()).hexdigest()[:8],
        'nvcc_version': get_nvcc_version(),
        'gpu_card_type': get_gpu_card_type(),
        'hostname': get_hostname(),
        'pytorch_version': get_pytorch_version(),
        'compute_capability': get_compute_capability(),
    }

    for key, value in components.items():
        print(f"{key:20}: {value}")

    cache_hash = generate_cache_hash(generated_code, file_path)
    print(f"{'cache_hash':20}: {cache_hash}")

    build_dir = get_cache_build_directory(generated_code, file_path)
    print(f"{'build_directory':20}: {build_dir}")


def validate_custom_cuda_kernel(cuda_source: str) -> dict:
    """
    Comprehensive validation for custom CUDA kernels based on KernelBench operations.
    Detects library shortcuts that bypass actual kernel implementation.
    """
    results = {
        "has_kernel_function": False,
        "has_kernel_launch": False,
        "has_thread_indexing": False,
        "library_shortcuts": [],
        "is_valid_custom_kernel": False
    }

    # Check for kernel definition
    if "__global__" in cuda_source:
        results["has_kernel_function"] = True

    # Check for kernel launch
    if "<<<" in cuda_source and ">>>" in cuda_source:
        results["has_kernel_launch"] = True

    # Check for thread indexing
    thread_patterns = ["threadIdx", "blockIdx", "blockDim", "gridDim"]
    if any(pattern in cuda_source for pattern in thread_patterns):
        results["has_thread_indexing"] = True

    # EXPANDED: Comprehensive library shortcuts based on KernelBench Level 1 & 2
    shortcuts = {
        # ===== PyTorch ATen Library (Most Common Hack!) =====
        "torch::": [
            # Level 1: Matrix Operations
            "torch::matmul", "torch::mm", "torch::bmm", "torch::addmm",
            "torch::baddbmm", "torch::dot", "torch::mv", "torch::ger",

            # Level 1: Convolution Operations
            "torch::conv1d", "torch::conv2d", "torch::conv3d",
            "torch::conv_transpose1d", "torch::conv_transpose2d", "torch::conv_transpose3d",

            # Level 1: Pooling Operations
            "torch::max_pool1d", "torch::max_pool2d", "torch::max_pool3d",
            "torch::avg_pool1d", "torch::avg_pool2d", "torch::avg_pool3d",
            "torch::adaptive_avg_pool1d", "torch::adaptive_avg_pool2d", "torch::adaptive_avg_pool3d",
            "torch::adaptive_max_pool1d", "torch::adaptive_max_pool2d", "torch::adaptive_max_pool3d",

            # Level 1: Activation Functions
            "torch::relu", "torch::gelu", "torch::silu", "torch::sigmoid",
            "torch::tanh", "torch::softmax", "torch::log_softmax",
            "torch::leaky_relu", "torch::elu", "torch::selu", "torch::prelu",
            "torch::softplus", "torch::softsign", "torch::hardtanh", "torch::hardsigmoid",
            "torch::hardswish", "torch::mish", "torch::swish",

            # Level 1: Normalization Layers
            "torch::layer_norm", "torch::batch_norm", "torch::group_norm",
            "torch::instance_norm", "torch::local_response_norm",

            # Level 1: Loss Functions
            "torch::cross_entropy", "torch::nll_loss", "torch::mse_loss",
            "torch::l1_loss", "torch::smooth_l1_loss", "torch::binary_cross_entropy",
            "torch::binary_cross_entropy_with_logits", "torch::kl_div",
            "torch::cosine_embedding_loss", "torch::ctc_loss",

            # Level 1: Reduction Operations ??
            "torch::sum", "torch::mean", "torch::prod", "torch::max", "torch::min",
            "torch::argmax", "torch::argmin", "torch::median", "torch::std", "torch::var",
            "torch::norm", "torch::dist", "torch::logsumexp",

            # Level 1: Element-wise Operations ??
            "torch::add", "torch::sub", "torch::mul", "torch::div",
            "torch::pow", "torch::exp", "torch::log", "torch::sqrt",
            "torch::abs", "torch::neg", "torch::reciprocal",
            "torch::clamp", "torch::clip",

            # Level 1: Dropout & Regularization
            "torch::dropout", "torch::alpha_dropout", "torch::feature_alpha_dropout",

            # Level 1: Embedding Operations
            "torch::embedding", "torch::embedding_bag",

            # Level 1: Attention Operations
            "torch::scaled_dot_product_attention", "torch::multi_head_attention_forward",

            # Level 1: Upsampling/Interpolation
            "torch::upsample_nearest1d", "torch::upsample_nearest2d", "torch::upsample_nearest3d",
            "torch::upsample_linear1d", "torch::upsample_bilinear2d", "torch::upsample_trilinear3d",
            "torch::interpolate",

            # Level 2: Fused Operations (Common Patterns)
            "torch::addmm", "torch::addbmm", "torch::addmv",
            "torch::addr", "torch::baddbmm",
        ],

        # # ===== ATen Native & CUDA Implementations =====
        # "at::native::": [
        #     "at::native::", # Catch-all for any native implementation
        # ],
        # "at::cuda::": [
        #     "at::cuda::", # Catch-all for CUDA implementations
        # ],
        # "at::": [
        #     "at::matmul", "at::conv", "at::batch_norm", "at::layer_norm",
        #     "at::softmax", "at::log_softmax", "at::relu", "at::gelu",
        # ],

        # # ===== cuBLAS (BLAS Library) =====
        # "cublas": [
        #     # GEMM variants (Matrix Multiply)
        #     "cublasSgemm", "cublasDgemm", "cublasHgemm", "cublasCgemm", "cublasZgemm",
        #     "cublasGemmEx", "cublasGemmBatchedEx", "cublasGemmStridedBatchedEx",

        #     # Other BLAS operations
        #     "cublasSaxpy", "cublasDaxpy", "cublasSdot", "cublasDdot",
        #     "cublasSscal", "cublasDscal", "cublasSnrm2", "cublasDnrm2",
        #     "cublasSgemv", "cublasDgemv", "cublasSger", "cublasDger",
        # ],

        # # ===== cuDNN (Deep Neural Network Library) =====
        # "cudnn": [
        #     # Convolution
        #     "cudnnConvolutionForward", "cudnnConvolutionBackwardData", "cudnnConvolutionBackwardFilter",
        #     "cudnnConvolutionBiasActivationForward",

        #     # Pooling
        #     "cudnnPoolingForward", "cudnnPoolingBackward",

        #     # Activation
        #     "cudnnActivationForward", "cudnnActivationBackward",

        #     # Softmax
        #     "cudnnSoftmaxForward", "cudnnSoftmaxBackward",

        #     # Normalization
        #     "cudnnBatchNormalizationForward", "cudnnBatchNormalizationBackward",
        #     "cudnnNormalizationForward", "cudnnNormalizationBackward",

        #     # RNN
        #     "cudnnRNNForward", "cudnnRNNBackward",

        #     # Dropout
        #     "cudnnDropoutForward", "cudnnDropoutBackward",
        # ],

        # # ===== Thrust (High-level CUDA C++ Library) =====
        # "thrust::": [
        #     "thrust::reduce", "thrust::transform", "thrust::sort",
        #     "thrust::copy", "thrust::fill", "thrust::sequence",
        #     "thrust::transform_reduce", "thrust::inclusive_scan", "thrust::exclusive_scan",
        #     "thrust::gather", "thrust::scatter", "thrust::partition",
        #     "thrust::unique", "thrust::remove", "thrust::count",
        # ],

        # # ===== CUB (CUDA Unbound Library) =====
        # "cub::": [
        #     "cub::DeviceReduce", "cub::DeviceScan", "cub::DeviceHistogram",
        #     "cub::BlockReduce", "cub::BlockScan", "cub::WarpReduce",
        #     "cub::DeviceSelect", "cub::DevicePartition",
        # ],

        # # ===== cuFFT (Fast Fourier Transform) =====
        # "cufft": [
        #     "cufftExecC2C", "cufftExecR2C", "cufftExecC2R",
        #     "cufftPlan1d", "cufftPlan2d", "cufftPlan3d",
        # ],

        # # ===== cuSPARSE (Sparse Matrix Operations) =====
        # "cusparse": [
        #     "cusparseSpMV", "cusparseSpMM", "cusparseSpGEMM",
        #     "cusparseCsrgemm", "cusparseCsrmv",
        # ],

        # # ===== cuSOLVER (Linear Algebra Solvers) =====
        # "cusolver": [
        #     "cusolverDnSgeqrf", "cusolverDnSgetrf", "cusolverDnSpotrf",
        # ],

        # # ===== cuRAND (Random Number Generation) =====
        # "curand": [
        #     "curandGenerateUniform", "curandGenerateNormal",
        #     "curandGenerateLogNormal", "curandGeneratePoisson",
        # ],

        # # ===== CUTLASS (CUDA Templates for Linear Algebra Subroutines) =====
        # "cutlass::": [
        #     "cutlass::gemm", "cutlass::conv",
        # ],

        # # ===== NCCL (NVIDIA Collective Communications Library) =====
        # "nccl": [
        #     "ncclAllReduce", "ncclBroadcast", "ncclReduce",
        #     "ncclAllGather", "ncclReduceScatter",
        # ],

        # ===== TensorRT Core Operations =====
        "tensorrt": [
            "tensorrt::ILayer", "tensorrt::IConvolutionLayer",
        ],
    }

    pytorch_heavy_ops = [
        # Matrix Operations (Level 1)
        'torch.matmul', 'torch.mm', 'torch.bmm', 'torch.einsum',

        # Convolution Operations (Level 1)
        'nn.Conv1d', 'nn.Conv2d', 'nn.Conv3d',
        'nn.ConvTranspose1d', 'nn.ConvTranspose2d', 'nn.ConvTranspose3d',
        'F.conv1d', 'F.conv2d', 'F.conv3d',
        'F.conv_transpose1d', 'F.conv_transpose2d', 'F.conv_transpose3d',

        # Pooling Operations (Level 1)
        'nn.MaxPool1d', 'nn.MaxPool2d', 'nn.MaxPool3d',
        'nn.AvgPool1d', 'nn.AvgPool2d', 'nn.AvgPool3d',
        'nn.AdaptiveAvgPool1d', 'nn.AdaptiveAvgPool2d', 'nn.AdaptiveAvgPool3d',
        'nn.AdaptiveMaxPool1d', 'nn.AdaptiveMaxPool2d', 'nn.AdaptiveMaxPool3d',
        'F.max_pool1d', 'F.max_pool2d', 'F.max_pool3d',
        'F.avg_pool1d', 'F.avg_pool2d', 'F.avg_pool3d',
        'F.adaptive_avg_pool1d', 'F.adaptive_avg_pool2d', 'F.adaptive_avg_pool3d',

        # Activation Functions (Level 1)
        'nn.ReLU', 'nn.LeakyReLU', 'nn.GELU', 'nn.SiLU', 'nn.Mish',
        'nn.Sigmoid', 'nn.Tanh', 'nn.Softmax', 'nn.LogSoftmax',
        'nn.ELU', 'nn.SELU', 'nn.PReLU', 'nn.Softplus', 'nn.Softsign',
        'nn.Hardtanh', 'nn.HardSigmoid', 'nn.Hardswish', 'nn.Swish',
        'F.relu', 'F.leaky_relu', 'F.gelu', 'F.silu', 'F.mish',
        'F.sigmoid', 'F.tanh', 'F.softmax', 'F.log_softmax',
        'torch.relu', 'torch.sigmoid', 'torch.tanh',

        # Normalization Operations (Level 1)
        'nn.BatchNorm1d', 'nn.BatchNorm2d', 'nn.BatchNorm3d',
        'nn.LayerNorm', 'nn.GroupNorm', 'nn.InstanceNorm1d', 'nn.InstanceNorm2d', 'nn.InstanceNorm3d',
        'F.batch_norm', 'F.layer_norm', 'F.group_norm', 'F.instance_norm',

        # Loss Functions (Level 1)
        'nn.CrossEntropyLoss', 'nn.NLLLoss', 'nn.MSELoss', 'nn.L1Loss',
        'nn.SmoothL1Loss', 'nn.BCELoss', 'nn.BCEWithLogitsLoss',
        'nn.KLDivLoss', 'nn.CosineEmbeddingLoss', 'nn.CTCLoss',
        'F.cross_entropy', 'F.nll_loss', 'F.mse_loss', 'F.l1_loss',

        # Reduction Operations (Level 1)
        'torch.sum', 'torch.mean', 'torch.prod', 'torch.max', 'torch.min',
        'torch.argmax', 'torch.argmin', 'torch.median', 'torch.std', 'torch.var',
        'torch.norm', 'torch.dist', 'torch.logsumexp',

        # Linear/Recurrent Operations (Level 1)
        'nn.Linear', 'nn.LSTM', 'nn.GRU', 'nn.RNN',
        'F.linear',

        # Attention Operations (Level 1)
        'nn.MultiheadAttention', 'nn.Transformer',
        'F.scaled_dot_product_attention', 'F.multi_head_attention_forward',

        # Embedding Operations (Level 1)
        'nn.Embedding', 'nn.EmbeddingBag',
        'F.embedding', 'F.embedding_bag',

        # Dropout (Level 1)
        'nn.Dropout', 'nn.Dropout1d', 'nn.Dropout2d', 'nn.Dropout3d',
        'F.dropout',

        # Upsampling/Interpolation (Level 1)
        'F.interpolate', 'F.upsample',
        'F.upsample_nearest', 'F.upsample_bilinear',
        'nn.Upsample', 'nn.UpsamplingNearest2d', 'nn.UpsamplingBilinear2d',
    ]
    results["has_heavy_pytorch_ops"] = False
    for op in pytorch_heavy_ops:
        if op in cuda_source:
            results["has_heavy_pytorch_ops"] = True
            break
    # Check for library shortcuts
    for category, patterns in shortcuts.items():
        for pattern in patterns:
            if pattern in cuda_source:
                results["library_shortcuts"].append(pattern)

    # Determine if it's a valid custom kernel
    results["is_valid_custom_kernel"] = (
        results["has_kernel_function"] and
        results["has_kernel_launch"] and
        results["has_thread_indexing"] and
        len(results["library_shortcuts"]) == 0 and
        not results["has_heavy_pytorch_ops"]
    )

    return results


# ===== ADDITIONAL VALIDATION: Check for Functional Patterns =====
def check_kernel_implementation_patterns(cuda_source: str) -> dict:
    """
    Check if the kernel actually implements the algorithm vs calling libraries.
    """
    patterns = {
        # Good signs - actual implementation
        "implements_algorithm": False,
        "uses_shared_memory": False,
        "uses_thread_cooperation": False,
        "manual_memory_access": False,

        # Bad signs - library delegation
        "delegates_to_library": False,
        "minimal_cuda_code": False,
    }

    # Good patterns (indicates real implementation)
    good_indicators = {
        "uses_shared_memory": ["__shared__", "extern __shared__"],
        "uses_thread_cooperation": ["__syncthreads()", "cooperative_groups"],
        "manual_memory_access": ["data_ptr<", "[idx]", "[threadIdx", "[blockIdx"],
        "implements_algorithm": [
            "for", "while",  # loops for computation
            "float sum", "double sum", "int sum",  # accumulation variables
            "atomicAdd", "atomicMax", "atomicMin",  # atomic operations
        ]
    }

    for pattern_name, indicators in good_indicators.items():
        if any(ind in cuda_source for ind in indicators):
            patterns[pattern_name] = True

    # Check if it's mostly just a wrapper
    lines = cuda_source.split('\n')
    code_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith('//')]

    if len(code_lines) < 10:
        patterns["minimal_cuda_code"] = True

    # Check if it delegates to library (torch::, at::, cublas, etc)
    library_patterns = ["torch::", "at::", "cublas", "cudnn", "thrust::"]
    if any(lib in cuda_source for lib in library_patterns):
        patterns["delegates_to_library"] = True

    return patterns


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--code", type=str, required=True)
    argparser.add_argument("--code_type", type=str, required=False, default="triton")
    args = argparser.parse_args()

    # read code from file
    with open(args.code, "r") as f:
        code = f.read()

    if args.code_type == "triton":
        print(resolve_triton_code(code))
    else:
        print(resolve_custom_cuda_kernel(code))

    # Test the hash function
    test_code = """
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hinge_loss_kernel(const float* predictions, const float* targets, float* loss, int n) {
    extern __shared__ float sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute clamped values
    float value = 0.0f;
    if (idx < n) {
        float temp = 1.0f - predictions[idx] * targets[idx];
        value = fmaxf(0.0f, temp);
    }

    // Parallel reduction for sum
    sdata[threadIdx.x] = value;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(loss, sdata[0]);
    }
}

torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    int n = predictions.numel();
    auto loss = torch::zeros({}, predictions.options());

    if (n > 0) {
        int threads = 1024;
        int blocks = (n + threads - 1) / threads;
        hinge_loss_kernel<<<blocks, threads, threads * sizeof(float)>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            loss.data_ptr<float>(),
            n
        );
    }

    return loss / n;
}
'''

cpp_source = "torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"

hinge_loss_ext = load_inline(
    name='hinge_loss_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['hinge_loss_cuda'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return hinge_loss_ext.hinge_loss_cuda(predictions, targets)

    """
    test_path = "/test/path/file.py"

    print_cache_info(test_code, test_path)
