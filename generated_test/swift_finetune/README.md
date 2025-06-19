# Swift Qwen3 Fine-tuning Test Suite

This directory contains comprehensive tests for the Swift Qwen3 fine-tuning implementation. The test suite validates all aspects of the fine-tuning pipeline including configuration loading, dataset preparation, training setup, and execution.

## Overview

The Swift fine-tuning implementation (`finetune_swift.py`) provides a wrapper around the **ms-swift** framework (ModelScope Swift) for fine-tuning Qwen3 models with LoRA (Low-Rank Adaptation).

### Key Features Tested

- ✅ Configuration loading and validation
- ✅ Model type mapping (Qwen3 variants)
- ✅ Dataset preparation (local and remote)
- ✅ LoRA configuration setup
- ✅ Training pipeline execution
- ✅ Evaluation and testing workflows
- ✅ Environment setup and dependency management

## Test Structure

### 1. Unit Tests (`test_swift_finetune.py`)

Comprehensive unit tests that validate individual components:

- **Configuration Tests**: YAML loading, defaults, model type mapping
- **Dataset Tests**: Local vs remote dataset preparation
- **Component Tests**: SwiftFineTuningConfig dataclass validation
- **Mock Training Tests**: Training pipeline with mocked Swift functions

### 2. Integration Tests (`test_integration.py`)

End-to-end integration tests that validate the complete pipeline:

- **Environment Setup**: Dependency checking and validation
- **Configuration Creation**: Dynamic test configuration generation
- **Pipeline Execution**: Full training pipeline testing
- **Output Validation**: Results verification and artifact checking

### 3. Test Runner (`run_tests.sh`)

Automated test execution script with:

- **Environment Setup**: Virtual environment and CUDA configuration
- **Dependency Management**: Package installation using `uv` or `pip`
- **Test Orchestration**: Sequential execution of all test suites
- **Result Reporting**: Colored output and comprehensive summaries

## Usage

### Prerequisites

1. **Environment Setup**:
   ```bash
   # Using uv (recommended)
   uv venv
   source .venv/bin/activate
   
   # Or using venv
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   # Install ms-swift and dependencies
   uv pip install ms-swift>=3.0.0
   # Or: pip install ms-swift>=3.0.0
   
   # Install other requirements
   uv pip install -r requirements.txt
   ```

### Running Tests

#### Option 1: Automated Test Runner (Recommended)
```bash
cd generated_test/swift_finetune
./run_tests.sh
```

This will:
- Set up the environment automatically
- Install missing dependencies
- Run all test suites
- Provide a comprehensive summary

#### Option 2: Manual Test Execution

**Unit Tests**:
```bash
cd generated_test/swift_finetune
export PYTHONPATH="../../:$PYTHONPATH"
python3 test_swift_finetune.py
```

**Integration Tests**:
```bash
cd generated_test/swift_finetune
export PYTHONPATH="../../:$PYTHONPATH"
python3 test_integration.py
```

### Environment Configuration

The tests automatically configure the environment according to the repository rules:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
source .venv/bin/activate
export PYTHONPATH="../../:$PYTHONPATH"
```

## Test Configuration

### Mock Data Generation

Tests automatically generate mock training data in JSONL format:

```json
{
  "instruction": "What is machine learning?",
  "input": "",
  "output": "Machine learning is a subset of artificial intelligence..."
}
```

### Test Configuration

Dynamic YAML configuration for testing:

```yaml
model:
  name: "Qwen/Qwen3-8B-Instruct"
  max_seq_length: 16384  # Preserved from original config
  load_in_4bit: true

training:
  num_train_epochs: 1
  learning_rate: 3e-5
  per_device_batch_size: 1
  # ... other training parameters

lora:
  r: 64
  alpha: 64
  dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

## Supported Model Variants

The test suite validates support for multiple Qwen3 model variants:

- `Qwen/Qwen3-8B-Instruct` → `qwen3-8b-instruct`
- `Qwen/Qwen3-7B-Instruct` → `qwen3-7b-instruct`
- `Qwen/Qwen3-14B-Instruct` → `qwen3-14b-instruct`
- `Qwen/Qwen3-8B` → `qwen3-8b`

## Test Results

### Expected Outputs

After successful test execution, you should see:

```
🚀 Swift Qwen3 Fine-tuning Test Runner
========================================
✅ Unit Tests: PASSED
✅ Integration Tests: PASSED
✅ Configuration Tests: PASSED
✅ Script Execution Tests: PASSED

Tests passed: 4/4
✅ All tests passed! 🎉
```

### Test Artifacts

Tests generate temporary artifacts in `/tmp/swift_integration_test_*`:

- `integration_config.yaml`: Test configuration
- `data/train.jsonl`: Mock training data
- `output/test_results.json`: Test execution results

## Troubleshooting

### Common Issues

1. **ImportError: ms-swift not found**
   ```bash
   pip install ms-swift>=3.0.0
   ```

2. **CUDA_VISIBLE_DEVICES not set**
   ```bash
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   ```

3. **Virtual environment not activated**
   ```bash
   source .venv/bin/activate
   ```

4. **Python path issues**
   ```bash
   export PYTHONPATH="../../:$PYTHONPATH"
   ```

### Debugging

Enable verbose output:
```bash
cd generated_test/swift_finetune
python3 -v test_swift_finetune.py
```

Check dependency status:
```bash
python3 -c "
try:
    from swift.llm import SftArguments
    print('✅ ms-swift available')
except ImportError as e:
    print(f'❌ ms-swift not available: {e}')
"
```

## Integration with Main Script

The tests validate the main Swift fine-tuning script (`finetune_swift.py`):

```bash
# After tests pass, you can run the actual fine-tuning:
cd ../../
export CUDA_VISIBLE_DEVICES=0,1,2,3
source .venv/bin/activate
python3 finetune_swift.py --mode full
```

## Contributing

When adding new features to the Swift fine-tuning implementation:

1. Add corresponding unit tests in `test_swift_finetune.py`
2. Update integration tests in `test_integration.py` if needed
3. Run the full test suite: `./run_tests.sh`
4. Ensure all tests pass before committing

## Notes

- Tests use mocked Swift functions for safety and speed
- Actual model downloading and training are not performed in tests
- Test configurations use small parameters for fast execution
- Original sequence length and model name are preserved per repository rules 