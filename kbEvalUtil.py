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
    model_custom_src: str, context: dict, build_directory: str = None, filename: str = "<string>", code_type: str = "triton"
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
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.device(device), torch.cuda.stream(stream):
        # Warm ups
        for _ in range(num_warmups):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            # use exact same device as the actual trials
            start_event.record(stream=stream)
            kernel_fn(*args)
            end_event.record(stream=stream)
            # Synchronize to ensure the events have completed
            torch.cuda.synchronize(device=device)

        # Actual trials
        for trial in range(num_trials):
            # create event marker default is not interprocess
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            # start_event.device = device
            # end_event.device = device
            start_event.record(stream=stream)
            kernel_fn(*args)
            end_event.record(stream=stream)
            # Synchronize to ensure the events have completed
            torch.cuda.synchronize(device=device)

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
        'hostname': get_hostname(),
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


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--code", type=str, required=True)
    args = argparser.parse_args()

    # read code from file
    with open(args.code, "r") as f:
        code = f.read()

    print(resolve_triton_code(code))


    # Test the hash function
    test_code = """
    #include <torch/extension.h>
    __global__ void test_kernel() { }
    """
    test_path = "/test/path/file.py"

    print_cache_info(test_code, test_path)
