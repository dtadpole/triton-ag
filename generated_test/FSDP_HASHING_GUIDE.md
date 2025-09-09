# Proper Model Parameter Hashing in FSDP

## The Problem

When working with FSDP (Fully Sharded Data Parallel) models, creating a hash of model parameters is not straightforward because:

1. **Parameters are sharded across ranks** - Each rank only has access to a subset of the model parameters
2. **Direct parameter iteration fails** - `model.parameters()` returns different parameters on each rank
3. **Inconsistent hashing** - Different ranks will produce different hashes for the same model state
4. **ShardedTensor objects** - FSDP state dicts contain `ShardedTensor` objects instead of regular `torch.Tensor` objects, which don't support all tensor operations like `numel()`

## The Solutions

### 1. ✅ CORRECT: Sharded Parameter Hashing (Recommended)

Use `model.state_dict()` to get the sharded parameters for each rank:

```python
def get_model_state_hash(model):
    """Get hash of model parameters for consistency checking in FSDP"""
    hasher = hashlib.md5()
    rank = dist.get_rank()
    
    # Get the model state dict - this gives us the sharded parameters for this rank
    model_state_dict = model.state_dict()
    
    # Count parameters for logging - handle both regular tensors and ShardedTensors
    num_params = 0
    for p in model_state_dict.values():
        # Check for ShardedTensor first (it's a subclass of torch.Tensor)
        if hasattr(p, 'local_shards'):  # This is a ShardedTensor
            num_params += p.size().numel()
        elif isinstance(p, torch.Tensor):  # This is a regular tensor
            num_params += p.numel()
    logger.info(f"🔍 Rank {rank}: Model has [{num_params:,}] parameters in state dict")
    
    # Sort by parameter names for consistent ordering across ranks
    # Each rank will hash its own sharded parameters
    for param_name in sorted(model_state_dict.keys()):
        param_tensor = model_state_dict[param_name]
        
        # Skip non-tensor entries (like metadata)
        if not isinstance(param_tensor, torch.Tensor) and not hasattr(param_tensor, 'size'):
            continue
        
        # For ShardedTensor, we need to get the local shard
        if hasattr(param_tensor, 'local_shards') and param_tensor.local_shards():
            # This is a ShardedTensor - get the local shard data
            local_shard = param_tensor.local_shards()[0].tensor
            tensor_to_hash = local_shard
        elif isinstance(param_tensor, torch.Tensor):
            # This is a regular tensor
            tensor_to_hash = param_tensor
        else:
            continue
            
        # Convert to float32 if needed for consistent hashing
        if tensor_to_hash.dtype == torch.bfloat16:
            tensor_to_hash = tensor_to_hash.float()
        elif tensor_to_hash.dtype == torch.float16:
            tensor_to_hash = tensor_to_hash.float()
            
        # Move to CPU and hash the tensor data
        hasher.update(tensor_to_hash.cpu().numpy().tobytes())
    
    return hasher.hexdigest()
```

**Advantages:**
- ✅ Efficient - no communication between ranks
- ✅ Works well for distributed training
- ✅ Each rank can verify its own parameter consistency
- ✅ Consistent ordering by parameter names

**Use Cases:**
- Parameter consistency checking during training
- Debugging parameter updates
- Verifying checkpoint loading

### 2. 🔍 FULL MODEL: Complete Model Hashing (Expensive)

Gather all parameters to rank 0 for a complete model hash:

```python
def get_full_model_state_hash(model):
    """Get hash of the FULL model parameters (gathers all shards to rank 0)
    
    WARNING: This is expensive and should only be used for debugging/verification.
    It gathers all parameter shards to rank 0, which can cause OOM for large models.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        logger.info("🔍 Rank 0: Computing full model hash (expensive operation)")
    
    # Get the full state dict on rank 0 only
    if rank == 0:
        # Use FSDP's full state dict functionality
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig
        
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            full_state_dict = model.state_dict()
        
        hasher = hashlib.md5()
        num_params = 0
        
        # Sort by parameter names for consistent ordering
        for param_name in sorted(full_state_dict.keys()):
            param_tensor = full_state_dict[param_name]
            
            # Skip non-tensor entries
            if not isinstance(param_tensor, torch.Tensor):
                continue
                
            num_params += param_tensor.numel()
            
            # Convert to float32 if needed for consistent hashing
            if param_tensor.dtype == torch.bfloat16:
                param_tensor = param_tensor.float()
            elif param_tensor.dtype == torch.float16:
                param_tensor = param_tensor.float()
                
            # Hash the tensor data
            hasher.update(param_tensor.cpu().numpy().tobytes())
        
        logger.info(f"🔍 Rank 0: Full model has [{num_params:,}] parameters")
        full_hash = hasher.hexdigest()
        
        # Broadcast the hash to all ranks
        hash_bytes = full_hash.encode()
        for other_rank in range(1, world_size):
            dist.send(torch.tensor([ord(c) for c in hash_bytes], dtype=torch.uint8), dst=other_rank)
        
        return full_hash
    else:
        # Receive the hash from rank 0
        hash_tensor = torch.zeros(32, dtype=torch.uint8)  # MD5 hash is 32 hex chars
        dist.recv(hash_tensor, src=0)
        return ''.join(chr(c.item()) for c in hash_tensor)
```

**Advantages:**
- ✅ All ranks get the same hash
- ✅ Complete model verification
- ✅ Useful for debugging

**Disadvantages:**
- ❌ Expensive - requires gathering all parameters to rank 0
- ❌ Can cause OOM for large models
- ❌ Requires communication between ranks

**Use Cases:**
- Final model verification
- Debugging parameter inconsistencies
- One-time validation

### 3. ❌ INCORRECT: Direct Parameter Iteration

**DON'T DO THIS:**

```python
def incorrect_model_hash(model):
    """INCORRECT: This won't work properly in FSDP"""
    hasher = hashlib.md5()
    
    # This is WRONG for FSDP because:
    # 1. Each rank only sees its sharded parameters
    # 2. Different ranks will have different hashes
    # 3. You can't sort parameters directly
    model_params = model.parameters()
    
    for param in sorted(model_params):  # This sorting won't work properly
        tensor = param.data
        hasher.update(tensor.cpu().numpy().tobytes())
    
    return hasher.hexdigest()
```

**Problems:**
- ❌ Each rank sees different parameters
- ❌ Cannot sort parameters directly
- ❌ Inconsistent results across ranks
- ❌ May cause errors or unexpected behavior

## ShardedTensor Handling

When using `SHARDED_STATE_DICT` with FSDP, the state dict contains `ShardedTensor` objects instead of regular `torch.Tensor` objects. These require special handling:

```python
# For ShardedTensor, we need to get the local shard
if hasattr(param_tensor, 'local_shards') and param_tensor.local_shards():
    # This is a ShardedTensor - get the local shard data
    local_shard = param_tensor.local_shards()[0].tensor
    tensor_to_hash = local_shard
elif isinstance(param_tensor, torch.Tensor):
    # This is a regular tensor
    tensor_to_hash = param_tensor
```

**Key points:**
- `ShardedTensor` objects don't support `numel()` directly - use `size().numel()` instead
- Access the actual tensor data via `local_shards()[0].tensor`
- Each rank only has access to its local shard of the parameter
- **IMPORTANT**: `ShardedTensor` is a subclass of `torch.Tensor`, so check for `hasattr(p, 'local_shards')` BEFORE checking `isinstance(p, torch.Tensor)`

## Best Practices

1. **Use sharded hashing for regular operations** - It's efficient and sufficient for most use cases
2. **Use full model hashing sparingly** - Only for debugging or final verification
3. **Always sort by parameter names** - Ensures consistent ordering
4. **Handle different data types** - Convert bfloat16/float16 to float32 for consistent hashing
5. **Skip non-tensor entries** - State dicts may contain metadata
6. **Move tensors to CPU** - Avoid GPU memory issues during hashing
7. **Handle ShardedTensor objects properly** - Extract local shards for hashing

## Example Usage

```python
# For regular parameter consistency checking
sharded_hash = get_model_state_hash(model)
logger.info(f"Sharded parameter hash: {sharded_hash}")

# For complete model verification (use sparingly)
full_hash = get_full_model_state_hash(model)
logger.info(f"Full model hash: {full_hash}")
```

## Key Takeaways

- **FSDP shards parameters across ranks** - each rank only sees a subset
- **Use `model.state_dict()`** - not `model.parameters()` for FSDP models
- **Sort by parameter names** - ensures consistent ordering
- **Sharded hashing is usually sufficient** - full model hashing is expensive
- **Handle data types properly** - convert to float32 for consistent hashing
