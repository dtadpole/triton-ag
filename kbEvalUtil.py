import socket
import traceback
import yaml
import torch
import torch.nn as nn
from pydantic import BaseModel
import os
import fcntl
import time
import psutil
from logger import logger
import numpy as np
import sys
import ast
import signal
import argparse
from collections import defaultdict

MAX_LOCK_AGE = 30 # seconds

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
    return "\n".join(traceback.format_exception(type(e), e, e.__traceback__))

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
    metadata: dict = {}
    runtime: float = -1.0  # in us, only recorded if we decide to measure performance
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


class FileLock:
    def __init__(self, lock_file):
        self.lock_file = lock_file
        self.lock_fd = None
        self.pid = os.getpid()

    def __enter__(self):
        try:
            # if open file with 'w', it will change modified timestamp even without writing to the file
            self.lock_fd = open(self.lock_file, 'r+')
        except FileNotFoundError:
             # if file does not exist, open file with 'w'
            self.lock_fd = open(self.lock_file, 'w')
        try:
            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            # write my pid to lock file
            self.lock_fd.truncate(0)
            self.lock_fd.write(str(self.pid) + "\n")
            self.lock_fd.flush()
            # os.fsync(self.lock_fd.fileno())
        except BlockingIOError as e:
            self.lock_fd.close()
            raise TimeoutError("Could not acquire lock")
        logger.warning(f"Lock [{self.lock_file}] acquired.")
        return self

    def __exit__(self, type, value, traceback):
        if self.lock_fd:
            self.lock_fd.write('\n[done]\n')
            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
            self.lock_fd.close()
            logger.warning(f"Lock [{self.lock_file}] released.")

def cleanup_lockfile(lock_file: str):
    if os.path.exists(lock_file):
        my_pid = os.getpid()
        lock_modified_time = os.path.getmtime(lock_file)
        with open(lock_file, 'r') as file:
            try:
                fcntl.flock(file.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                first_line = file.readline().strip()
                digits_only = ""
                for c in first_line:
                    if c.isdigit():
                        digits_only += c
                if digits_only:
                    file_pid = int(digits_only)
                    if my_pid != file_pid:
                        if psutil.pid_exists(file_pid):
                            logger.warning(f"[{my_pid}] Lock file [{lock_file}] for [pid={digits_only}] is running...")
                        else:
                            # if process is not running
                            logger.error(f"[{my_pid}] Lock file [{lock_file}] [pid={digits_only}] is not running, deleting...")
                            os.remove(lock_file)
            except BlockingIOError:
                if lock_modified_time < time.time() - MAX_LOCK_AGE:
                    # safety net: if modified time is more than MAX_LOCK_AGE, delete lock file
                    logger.error(f"[{my_pid}] Lock file [{lock_file}] older than [{MAX_LOCK_AGE}s], deleting...")
                    try:
                        os.remove(lock_file)
                    except Exception as e:
                        pass

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
        print(f"{"  "*len(self.scope_stack)} Inside class: [{".".join(self.scope_stack + [node.name])}]")
        self.scope_stack.append(node.name)
        if node.name == "ModelNew":
            self.model_new_class_count += 1
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node):
        print(f"{"  "*len(self.scope_stack)} Inside function: [{".".join(self.scope_stack + [node.name])}]")
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
                print(f"{"  "*len(self.scope_stack)} Function [{full_name}] is [@triton.jit]")
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_AsyncFunctionDef(self, node):
        print(f"{"  "*len(self.scope_stack)} Inside async function: [{".".join(self.scope_stack + [node.name])}]")
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
            print(f"{"  "*len(self.scope_stack)} Found function call: [{qualified_callee}]")
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

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--code", type=str, required=True)
    args = argparser.parse_args()

    # read code from file
    with open(args.code, "r") as f:
        code = f.read()

    print(resolve_triton_code(code))
