# FSDP State Dict Types: LOCAL_STATE_DICT vs SHARDED_STATE_DICT

## Overview

FSDP provides two different state dict types that serve different purposes when working with sharded models. Understanding the difference is crucial for proper parameter handling and hashing.

## LOCAL_STATE_DICT

### What it contains:
- **Only the local shard** of parameters that belong to the current rank
- **Regular torch.Tensor objects** - no ShardedTensor metadata
- **Direct access** to the actual parameter data

### Characteristics:
```python
with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
    state_dict = model.state_dict()
    
    # Each parameter is a regular torch.Tensor
    for name, param in state_dict.items():
        print(f"Type: {type(param)}")  # <class 'torch.Tensor'>
        print(f"Shape: {param.shape}")  # Local shard shape
        print(f"Numel: {param.numel()}")  # Works directly
        print(f"Device: {param.device}")  # Current rank's device
```

### Advantages:
- ✅ **Simple to work with** - regular tensors, no special handling needed
- ✅ **Direct access** - no need to extract local shards
- ✅ **Efficient** - no metadata overhead
- ✅ **Standard operations** - all torch.Tensor methods work

### Use Cases:
- Parameter hashing (like in your test)
- Local parameter manipulation
- When you only need the local shard data

## SHARDED_STATE_DICT

### What it contains:
- **ShardedTensor objects** with metadata about the full tensor
- **Information about all shards** across all ranks
- **Local shard data** accessible via `local_shards()[0].tensor`

### Characteristics:
```python
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    state_dict = model.state_dict()
    
    for name, param in state_dict.items():
        if hasattr(param, 'local_shards'):  # ShardedTensor
            print(f"Type: {type(param)}")  # <class 'torch.distributed._shard.sharded_tensor.ShardedTensor'>
            print(f"Full shape: {param.size()}")  # Full tensor shape across all ranks
            print(f"Full numel: {param.size().numel()}")  # Total parameters across all ranks
            
            # Access local shard
            local_shard = param.local_shards()[0].tensor
            print(f"Local shape: {local_shard.shape}")  # Local shard shape
            print(f"Local numel: {local_shard.numel()}")  # Local parameters
```

### Advantages:
- ✅ **Full tensor metadata** - know the complete tensor structure
- ✅ **Shard information** - understand how parameters are distributed
- ✅ **Compatible with checkpointing** - TDC expects ShardedTensor format

### Disadvantages:
- ❌ **More complex** - requires special handling for ShardedTensor
- ❌ **Metadata overhead** - stores information about all shards
- ❌ **Limited operations** - many torch functions don't work on ShardedTensor

### Use Cases:
- Checkpointing with TDC (Torch Distributed Checkpoint)
- When you need full tensor metadata
- Distributed training state management

## Practical Differences

### Parameter Counting:
```python
# LOCAL_STATE_DICT - Simple
with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
    state_dict = model.state_dict()
    total_params = sum(p.numel() for p in state_dict.values())

# SHARDED_STATE_DICT - Requires special handling
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    state_dict = model.state_dict()
    total_params = 0
    for p in state_dict.values():
        if hasattr(p, 'local_shards') and p.local_shards():
            total_params += p.local_shards()[0].tensor.numel()
        elif isinstance(p, torch.Tensor):
            total_params += p.numel()
```

### Parameter Hashing:
```python
# LOCAL_STATE_DICT - Simple
with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
    state_dict = model.state_dict()
    hasher = hashlib.md5()
    for param in state_dict.values():
        hasher.update(param.cpu().numpy().tobytes())

# SHARDED_STATE_DICT - Requires ShardedTensor handling
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    state_dict = model.state_dict()
    hasher = hashlib.md5()
    for param in state_dict.values():
        if hasattr(param, 'local_shards') and param.local_shards():
            local_shard = param.local_shards()[0].tensor
            hasher.update(local_shard.cpu().numpy().tobytes())
        elif isinstance(param, torch.Tensor):
            hasher.update(param.cpu().numpy().tobytes())
```

## When to Use Which?

### Use LOCAL_STATE_DICT when:
- ✅ **Parameter hashing** - Much simpler and more efficient
- ✅ **Local parameter manipulation** - Direct tensor access
- ✅ **Debugging** - Easier to work with regular tensors
- ✅ **Performance critical** - No metadata overhead

### Use SHARDED_STATE_DICT when:
- ✅ **Checkpointing with TDC** - Required format for distributed checkpointing
- ✅ **Need full tensor metadata** - Understanding complete tensor structure
- ✅ **Distributed state management** - Working with shard information

## Your Test Case

In your test, you switched from `SHARDED_STATE_DICT` to `LOCAL_STATE_DICT`, which is actually a **good choice** because:

1. **Simpler hashing** - No need to handle ShardedTensor objects
2. **Better performance** - No metadata overhead
3. **Easier debugging** - Regular tensors are easier to work with
4. **Same result** - Both approaches hash the same local parameter data

The warning you see about ShardedTensor deprecation is just PyTorch's way of saying they're moving to DTensor, but it doesn't affect functionality.

## Summary

| Aspect | LOCAL_STATE_DICT | SHARDED_STATE_DICT |
|--------|------------------|-------------------|
| **Data Type** | Regular torch.Tensor | ShardedTensor objects |
| **Complexity** | Simple | Complex (requires special handling) |
| **Performance** | Fast | Slower (metadata overhead) |
| **Use Case** | Parameter hashing, local ops | Checkpointing, full metadata |
| **Compatibility** | Standard torch operations | Limited torch operations |
| **Memory** | Lower overhead | Higher overhead |

For your parameter hashing use case, **LOCAL_STATE_DICT is the better choice** - it's simpler, faster, and gives you the same result!
