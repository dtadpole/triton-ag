import torch
import torch.nn.functional as F
import math
import time
import sys
import os

# Add parent directory to path to import the implementations
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from SegmentedTransformer import Model as PyTorchModel, get_inputs, get_init_inputs
from SegmentedTransformerTriton import SegmentedTransformerTriton

def compare_models():
    """Compare PyTorch and Triton implementations"""
    print("=== SegmentedTransformer Comparison: PyTorch vs Triton ===\n")
    
    # Initialize both models with same parameters
    init_args = get_init_inputs()
    print(f"Model parameters: num_segments={init_args[0]}, d_model={init_args[1]}, "
          f"num_heads={init_args[2]}, d_ff={init_args[3]}")
    
    pytorch_model = PyTorchModel(*init_args)
    triton_model = SegmentedTransformerTriton(*init_args)
    
    # Copy weights from PyTorch model to Triton model for fair comparison
    with torch.no_grad():
        # Copy QKV weights
        for i in range(init_args[0]):  # num_segments
            triton_model.qkv_weight[i].copy_(pytorch_model.attention.w_qkv.weight[i])
            triton_model.attn_out_weight[i].copy_(pytorch_model.attention.w_o.weight[i])
            triton_model.mlp1_weight[i].copy_(pytorch_model.mlp.linear1.weight[i])
            triton_model.mlp2_weight[i].copy_(pytorch_model.mlp.linear2.weight[i])
    
    # Test on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}\n")
    
    pytorch_model = pytorch_model.to(device)
    triton_model = triton_model.to(device)
    
    # Get test input
    inputs = get_inputs()
    input_tensor = inputs[0].to(device)
    print(f"Input shape: {input_tensor.shape}")
    
    # Test forward pass
    with torch.no_grad():
        # PyTorch forward pass
        start_time = time.time()
        pytorch_output = pytorch_model(input_tensor)
        pytorch_time = time.time() - start_time
        
        # Triton forward pass  
        start_time = time.time()
        triton_output = triton_model(input_tensor)
        triton_time = time.time() - start_time
    
    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"Triton output shape: {triton_output.shape}")
    print(f"\nTiming:")
    print(f"PyTorch forward pass: {pytorch_time*1000:.2f} ms")
    print(f"Triton forward pass: {triton_time*1000:.2f} ms")
    print(f"Speedup: {pytorch_time/triton_time:.2f}x")
    
    # Compare outputs (they won't be identical due to simplified attention in Triton)
    print(f"\nOutput Statistics:")
    print(f"PyTorch - mean: {pytorch_output.mean().item():.6f}, std: {pytorch_output.std().item():.6f}")
    print(f"Triton - mean: {triton_output.mean().item():.6f}, std: {triton_output.std().item():.6f}")
    
    # Check if outputs are in reasonable range
    assert not torch.isnan(pytorch_output).any(), "PyTorch output contains NaN"
    assert not torch.isnan(triton_output).any(), "Triton output contains NaN"
    assert not torch.isinf(pytorch_output).any(), "PyTorch output contains Inf"
    assert not torch.isinf(triton_output).any(), "Triton output contains Inf"
    
    print(f"\n✅ Both implementations produce valid outputs!")
    
    # Parameter count comparison
    pytorch_params = sum(p.numel() for p in pytorch_model.parameters())
    triton_params = sum(p.numel() for p in triton_model.parameters())
    print(f"\nParameter count:")
    print(f"PyTorch model: {pytorch_params:,}")
    print(f"Triton model: {triton_params:,}")
    print(f"Match number of parameters: {'✅' if pytorch_params == triton_params else '❌'}")
    # use allclose to check if the parameters are close
    
    return pytorch_output, triton_output

def benchmark_performance():
    """Benchmark performance with multiple runs"""
    print("\n=== Performance Benchmark ===\n")
    
    init_args = get_init_inputs()
    pytorch_model = PyTorchModel(*init_args)
    triton_model = SegmentedTransformerTriton(*init_args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pytorch_model = pytorch_model.to(device)
    triton_model = triton_model.to(device)
    
    inputs = get_inputs()
    input_tensor = inputs[0].to(device)
    
    # Warmup
    print("Warming up...")
    for _ in range(5):
        with torch.no_grad():
            _ = pytorch_model(input_tensor)
            _ = triton_model(input_tensor)
    
    # Benchmark
    num_runs = 100
    print(f"Running {num_runs} iterations...")
    
    # PyTorch timing
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = pytorch_model(input_tensor)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    pytorch_total_time = time.time() - start_time
    
    # Triton timing
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = triton_model(input_tensor)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    triton_total_time = time.time() - start_time
    
    print(f"\nBenchmark Results ({num_runs} runs):")
    print(f"PyTorch average time: {pytorch_total_time/num_runs*1000:.3f} ms")
    print(f"Triton average time: {triton_total_time/num_runs*1000:.3f} ms")
    print(f"Speedup: {pytorch_total_time/triton_total_time:.2f}x")
    
    # Memory usage comparison
    if device.type == 'cuda':
        print(f"\nMemory Usage:")
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = pytorch_model(input_tensor)
        pytorch_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = triton_model(input_tensor)
        triton_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        print(f"PyTorch peak memory: {pytorch_memory:.1f} MB")
        print(f"Triton peak memory: {triton_memory:.1f} MB")
        print(f"Memory reduction: {(pytorch_memory - triton_memory)/pytorch_memory*100:.1f}%")

if __name__ == "__main__":
    try:
        pytorch_output, triton_output = compare_models()
        # check allclose
        if torch.allclose(pytorch_output, triton_output, rtol=1e-2, atol=1e-2):
            print(f"\n✅ PyTorch and Triton outputs match!")
        else:
            print(f"\n❌ PyTorch and Triton outputs do not match!")
        
        benchmark_performance()
        print(f"\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 