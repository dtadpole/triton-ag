# KB Eval Client

This directory contains the evaluation client (`sequential_eval.py`) that interfaces with the kbEvalRemoteServer to evaluate CUDA kernels.

## Overview

The evaluation client takes input files containing evaluation parameters, calls the kbEvalRemoteServer, and writes results as `kbEval.json` in the same folder as the input file.

## Features

- **Flexible Input Formats**: Supports both JSON and simple text file formats
- **Load Balancing**: Automatically selects the best available server based on load
- **Error Handling**: Comprehensive error handling with detailed error reporting
- **Reference-Only Mode**: Option to evaluate only reference code for baseline measurements
- **Configurable**: Uses YAML configuration for server endpoints

## Usage

### Basic Usage

```bash
# Using JSON input format
python sequential_eval.py sample_input.json

# Using simple text input format
python sequential_eval.py simple_input.txt

# Reference-only evaluation
python sequential_eval.py sample_input.json --reference-only

# Verbose output
python sequential_eval.py sample_input.json --verbose
```

### Input File Formats

#### JSON Format (Recommended)

```json
{
  "model_tag": "claude-3-5-sonnet",
  "task_tag": "elemwise_add_test",
  "eval_tag": "test_run_001",
  "time_tag": "20250629_120000",
  "reference_code_path": "elemAddRef.py",
  "generated_code_path": "elemAddCuda.py",
  "metadata": {
    "description": "Testing element-wise addition with custom CUDA kernel",
    "test_type": "performance_comparison",
    "author": "test_user"
  }
}
```

**Required Fields:**
- `model_tag`: Identifier for the model/method used
- `task_tag`: Identifier for the task being evaluated
- `reference_code_path`: Path to reference PyTorch code
- `generated_code_path`: Path to generated CUDA kernel code

**Optional Fields:**
- `eval_tag`: Evaluation identifier (default: "eval")
- `time_tag`: Timestamp identifier (default: auto-generated)
- `metadata`: Additional metadata to include in results

#### Simple Text Format

```
elemAddRef.py
elemAddCuda.py
```

Simple format with one file path per line:
1. First line: Reference code file
2. Second line: Generated code file

### Code File Requirements

#### Reference Code (`elemAddRef.py`)

Must contain:
- `Model` class: PyTorch nn.Module with forward method
- `get_inputs()` function: Returns list of input tensors
- `get_init_inputs()` function: Returns list of initialization parameters

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b

def get_inputs():
    x = torch.randn(64, 16384)
    y = torch.randn(64, 16384)
    return [x, y]

def get_init_inputs():
    return []
```

#### Generated Code (`elemAddCuda.py`)

Must contain:
- `ModelNew` class: PyTorch nn.Module with CUDA kernel implementation
- Should have the same interface as the reference model

```python
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel code...
class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Initialize CUDA kernel

    def forward(self, a, b):
        # Call CUDA kernel
        return result
```

## Configuration

The client uses `kbEval.yaml` for server configuration:

```yaml
kbEvalClient:
  servers:
    - url: http://localhost:5678
    - url: http://server2:5678
```

## Command Line Options

```
usage: sequential_eval.py [-h] [--config CONFIG] [--reference-only] 
                         [--model-tag MODEL_TAG] [--task-tag TASK_TAG] 
                         [--eval-tag EVAL_TAG] [--verbose] input_file

positional arguments:
  input_file           Input file containing evaluation parameters

optional arguments:
  -h, --help           show this help message and exit
  --config CONFIG      Configuration file for server endpoints (default: kbEval.yaml)
  --reference-only     Evaluate reference code only (skip generated code)
  --model-tag MODEL_TAG    Override model tag from input file
  --task-tag TASK_TAG      Override task tag from input file  
  --eval-tag EVAL_TAG      Override eval tag from input file
  --verbose, -v        Enable verbose output
```

## Output

The client writes evaluation results to `kbEval.json` in the same directory as the input file.

### Result Format

```json
{
  "compiled": true,
  "correctness": true,
  "metadata": {
    "model_tag": "claude-3-5-sonnet",
    "task_tag": "elemwise_add_test",
    "eval_tag": "test_run_001",
    "time_tag": "20250629_120000",
    "server_url": "http://localhost:5678",
    "evaluation_timestamp": "2025-06-29T12:00:00.000000",
    "hardware": "NVIDIA GeForce RTX 4090",
    "device": "cuda:0"
  },
  "runtime": 123.45,
  "runtime_stats": {
    "mean": 123.45,
    "std": 5.67,
    "min": 115.2,
    "max": 135.8,
    "num_trials": 100
  }
}
```

### Result Fields

- `compiled`: Whether the code compiled successfully
- `correctness`: Whether the output matches the reference
- `runtime`: Average execution time in microseconds
- `runtime_stats`: Detailed timing statistics
- `metadata`: Additional information about the evaluation

## Example Workflow

1. **Prepare Code Files**
   ```bash
   # Create reference implementation
   cat > elemAddRef.py << EOF
   # Reference PyTorch code...
   EOF
   
   # Create CUDA kernel implementation  
   cat > elemAddCuda.py << EOF
   # Generated CUDA code...
   EOF
   ```

2. **Create Input File**
   ```bash
   cat > eval_input.json << EOF
   {
     "model_tag": "my-model",
     "task_tag": "elementwise-add",
     "reference_code_path": "elemAddRef.py",
     "generated_code_path": "elemAddCuda.py"
   }
   EOF
   ```

3. **Run Evaluation**
   ```bash
   python ../../sequential_eval.py eval_input.json --verbose
   ```

4. **Check Results**
   ```bash
   cat kbEval.json
   ```

## Error Handling

The client provides comprehensive error handling:

- **File Not Found**: Clear error messages for missing files
- **Server Connection**: Automatic retry and fallback servers
- **Compilation Errors**: Captured in result metadata  
- **Runtime Errors**: Logged with stack traces
- **Network Timeouts**: 5-minute timeout with clear error messages

Even when errors occur, a result file is still written with error details in the metadata.

## Requirements

- Python 3.8+
- PyYAML
- requests
- pathlib (built-in)

## Server Requirements

The evaluation requires a running kbEvalRemoteServer. See the main project documentation for server setup instructions. 