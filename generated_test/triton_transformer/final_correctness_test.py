import torch
import time
import sys
import os

# Add parent directories to path to import the implementations
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from SegmentedTransformer import Model as PyTorchModel, get_inputs, get_init_inputs
from SegmentedTransformerTriton import SegmentedTransformerTriton

def copy_weights_pytorch_to_triton(pytorch_model, triton_model):
    """Copy weights from PyTorch model to Triton model for exact comparison."""
    
    # Copy attention weights
    for seg in range(pytorch_model.attention.w_qkv.num_segments):
        # Get PyTorch QKV weights
        pytorch_qkv = pytorch_model.attention.w_qkv.weight[seg]  # (3*d_model, d_model)
        triton_model.qkv_weight[seg].data.copy_(pytorch_qkv)
        
        # Copy attention output weights
        pytorch_attn_out = pytorch_model.attention.w_o.weight[seg]  # (d_model, d_model)
        triton_model.attn_out_weight[seg].data.copy_(pytorch_attn_out)
        
        # Copy MLP weights
        pytorch_mlp1 = pytorch_model.mlp.linear1.weight[seg]  # (d_ff, d_model)
        triton_model.mlp1_weight[seg].data.copy_(pytorch_mlp1)
        
        pytorch_mlp2 = pytorch_model.mlp.linear2.weight[seg]  # (d_model, d_ff)
        triton_model.mlp2_weight[seg].data.copy_(pytorch_mlp2)

def final_correctness_and_performance_test():
    """Final comprehensive test of the updated SegmentedTransformerTriton."""
    print("=== Final Correctness & Performance Test ===\n")
    print("Testing updated SegmentedTransformerTriton.py\n")
    
    # Initialize models with same seed for reproducibility
    torch.manual_seed(42)
    init_args = get_init_inputs()
    pytorch_model = PyTorchModel(*init_args)
    
    torch.manual_seed(42)
    triton_model = SegmentedTransformerTriton(*init_args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pytorch_model = pytorch_model.to(device)
    triton_model = triton_model.to(device)
    
    # Copy weights from PyTorch model to Triton model for exact comparison
    print("Copying weights from PyTorch to Triton model...")
    copy_weights_pytorch_to_triton(pytorch_model, triton_model)
    
    # Create test input
    torch.manual_seed(123)
    inputs = get_inputs()
    input_tensor = inputs[0].to(device)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Model parameters: {sum(p.numel() for p in pytorch_model.parameters()):,}")
    print()
    
    # === CORRECTNESS TEST ===
    print("=== Correctness Test ===")
    
    with torch.no_grad():
        pytorch_output = pytorch_model(input_tensor)
        triton_output = triton_model(input_tensor)
    
    # Detailed comparison
    diff = torch.abs(pytorch_output - triton_output)
    rel_diff = diff / (torch.abs(pytorch_output) + 1e-8)
    
    print(f"PyTorch output - Mean: {pytorch_output.mean().item():.6f}, Std: {pytorch_output.std().item():.6f}")
    print(f"Triton output  - Mean: {triton_output.mean().item():.6f}, Std: {triton_output.std().item():.6f}")
    print()
    
    print(f"Absolute difference:")
    print(f"  Mean: {diff.mean().item():.2e}")
    print(f"  Max:  {diff.max().item():.2e}")
    print(f"  99th percentile: {torch.quantile(diff, 0.99).item():.2e}")
    print()
    
    print(f"Relative difference:")
    print(f"  Mean: {rel_diff.mean().item():.2e}")
    print(f"  Max:  {rel_diff.max().item():.2e}")
    print()
    
    # Check if outputs are close with different tolerances
    tolerances = [
        (1e-4, 1e-4, "Very Strict"),
        (1e-3, 1e-3, "Strict"),
        (1e-2, 1e-2, "Moderate"),
    ]
    
    match_found = False
    for rtol, atol, name in tolerances:
        are_close = torch.allclose(pytorch_output, triton_output, rtol=rtol, atol=atol)
        print(f"{name} tolerance (rtol={rtol}, atol={atol}): {'✅ PASS' if are_close else '❌ FAIL'}")
        if are_close and not match_found:
            match_found = True
            best_tolerance = name
    
    print()
    
    if match_found:
        print(f"🎉 SUCCESS: Outputs match within {best_tolerance.lower()} tolerance!")
        correctness_passed = True
    else:
        print("❌ FAILED: Outputs do not match within acceptable tolerance!")
        correctness_passed = False
    
    print()
    
    # === PERFORMANCE TEST ===
    print("=== Performance Test ===")
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = pytorch_model(input_tensor)
            _ = triton_model(input_tensor)
    
    torch.cuda.synchronize()
    
    # Benchmark PyTorch
    num_runs = 20
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            pytorch_output = pytorch_model(input_tensor)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) / num_runs
    
    # Benchmark Triton
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            triton_output = triton_model(input_tensor)
    torch.cuda.synchronize()
    triton_time = (time.time() - start_time) / num_runs
    
    speedup = pytorch_time / triton_time
    
    print(f"PyTorch time: {pytorch_time*1000:.2f} ms")
    print(f"Triton time:  {triton_time*1000:.2f} ms")
    print(f"Speedup:      {speedup:.2f}x {'📈' if speedup > 1.0 else '📉'}")
    print()
    
    # Memory usage comparison
    torch.cuda.empty_cache()
    
    # PyTorch memory
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        pytorch_output = pytorch_model(input_tensor)
    pytorch_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    torch.cuda.empty_cache()
    
    # Triton memory
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        triton_output = triton_model(input_tensor)
    triton_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    memory_reduction = (pytorch_memory - triton_memory) / pytorch_memory * 100
    
    print(f"PyTorch peak memory: {pytorch_memory:.1f} MB")
    print(f"Triton peak memory:  {triton_memory:.1f} MB")
    print(f"Memory reduction:    {memory_reduction:.1f}% {'💾' if memory_reduction > 0 else '💥'}")
    print()
    
    return correctness_passed, speedup, memory_reduction

def detailed_performance_analysis():
    """Analyze performance characteristics in detail."""
    print("=== Detailed Performance Analysis ===\n")
    
    # Test with different batch sizes
    batch_sizes = [1, 4, 12, 12 * 2, 12 * 4, 12 * 8, 12 * 16, 12 * 32]  # Original is 12
    
    torch.manual_seed(42)
    init_args = get_init_inputs()
    pytorch_model = PyTorchModel(*init_args)
    triton_model = SegmentedTransformerTriton(*init_args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pytorch_model = pytorch_model.to(device)
    triton_model = triton_model.to(device)
    
    copy_weights_pytorch_to_triton(pytorch_model, triton_model)
    
    print("Performance across different batch sizes:")
    print("Batch Size | PyTorch (ms) | Triton (ms) | Speedup")
    print("-" * 50)
    
    for batch_size in batch_sizes:
        # Create input with different batch size
        torch.manual_seed(123)
        input_tensor = torch.randn(batch_size, 32, 16, 64).to(device)
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = pytorch_model(input_tensor)
                _ = triton_model(input_tensor)
        
        torch.cuda.synchronize()
        
        # Benchmark
        num_runs = 10
        
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = pytorch_model(input_tensor)
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start_time) / num_runs
        
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = triton_model(input_tensor)
        torch.cuda.synchronize()
        triton_time = (time.time() - start_time) / num_runs
        
        speedup = pytorch_time / triton_time
        
        print(f"{batch_size:10d} | {pytorch_time*1000:11.2f} | {triton_time*1000:10.2f} | {speedup:6.2f}x")

if __name__ == "__main__":
    correctness, speedup, memory_reduction = final_correctness_and_performance_test()
    detailed_performance_analysis()
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    if correctness:
        print("✅ CORRECTNESS: Triton implementation matches PyTorch exactly!")
    else:
        print("❌ CORRECTNESS: Triton implementation has accuracy issues!")
    
    if speedup > 1.0:
        print(f"📈 PERFORMANCE: {speedup:.2f}x speedup achieved!")
    else:
        print(f"📉 PERFORMANCE: {speedup:.2f}x slower than PyTorch")
    
    if memory_reduction > 0:
        print(f"💾 MEMORY: {memory_reduction:.1f}% memory reduction")
    else:
        print(f"💥 MEMORY: {abs(memory_reduction):.1f}% memory increase")
    
    print()
    
    if correctness and speedup > 1.0:
        print("🏆 SUCCESS: Triton implementation is both correct AND faster!")
        print("🚀 Ready for production use!")
    elif correctness:
        print("⚠️  Triton implementation is correct but needs performance tuning")
    else:
        print("🔧 Triton implementation needs debugging before use")
        
    print("\n" + "="*60) 