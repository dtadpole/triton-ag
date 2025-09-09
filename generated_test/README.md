# TDC Checkpoint Testing Suite

This directory contains test scripts to verify Torch Distributed Checkpoint (TDC) functionality with FSDP training.

## Test Files

### 1. `test_tdc_simple.py`
**Purpose**: Basic TDC functionality test with simple models
- Tests TDC imports and basic functionality
- Creates a simple linear model and tests save/load operations
- Verifies `get_state_dict()`, `tdc.save()`, and `tdc.load()` work correctly
- **Usage**: `python -m torch.distributed.launch --nproc_per_node=2 test_tdc_simple.py`

### 2. `test_tdc_checkpoint.py`
**Purpose**: Full FSDP training engine test with TDC checkpointing
- Tests the complete EngineFSDP with TDC checkpoint saving/loading
- Uses actual model loading and training configuration
- Verifies checkpoint creation and loading in a realistic training scenario
- **Usage**: `python -m torch.distributed.launch --nproc_per_node=2 test_tdc_checkpoint.py`

### 3. `test_checkpoint_state_verification.py`
**Purpose**: Verifies checkpoint state consistency
- Tests that saved checkpoints contain expected data
- Validates model state, optimizer state, and metadata
- **Usage**: `python test_checkpoint_state_verification.py`

### 4. `test_simple_checkpoint_verification.py`
**Purpose**: Simple checkpoint verification without distributed training
- Basic checkpoint save/load verification
- Single GPU test for quick validation
- **Usage**: `python test_simple_checkpoint_verification.py`

### 5. `test_tdc_checkpoint_verification.py`
**Purpose**: TDC-specific checkpoint verification
- Tests TDC checkpoint format and structure
- Verifies distributed checkpoint compatibility
- **Usage**: `python test_tdc_checkpoint_verification.py`

## Test Results

✅ **TDC Checkpoint Functionality Verified**: All tests confirm that TDC checkpoint saving and loading works correctly with the FSDP training setup.

### Key Findings:
- TDC `get_state_dict()` successfully collects model and optimizer state
- TDC `save()` successfully saves checkpoints without hanging
- TDC `load()` successfully loads checkpoints for resuming training
- Distributed synchronization works properly across all ranks
- Checkpoint files are created with all necessary components (tokenizer, config, model state)

## Running Tests

### Prerequisites:
```bash
export CUDA_VISIBLE_DEVICES=2,3  # Use available GPUs
export HF_TOKEN=`cat ${HOME}/.keys/huggingface.api.key`
source .venv/bin/activate
```

### Distributed Tests:
```bash
# Basic TDC test
python -m torch.distributed.launch --nproc_per_node=2 test_tdc_simple.py

# Full FSDP training test
python -m torch.distributed.launch --nproc_per_node=2 test_tdc_checkpoint.py
```

### Single GPU Tests:
```bash
# Simple verification
python test_simple_checkpoint_verification.py

# Checkpoint state verification
python test_checkpoint_state_verification.py

# TDC verification
python test_tdc_checkpoint_verification.py
```

## Notes

- All tests have been successfully run and verified
- The original TDC implementation in `engineFSDP.py` is working correctly
- No hanging issues were found in the checkpoint saving/loading process
- Training successfully progresses through checkpoint points without problems
