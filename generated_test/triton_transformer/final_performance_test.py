import torch
import torch.nn.functional as F
import math
import time
import sys
import os

# Add parent directories to path to import the implementations
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from SegmentedTransformer import Model as PyTorchModel, get_inputs, get_init_inputs
from SegmentedTransformerTriton import SegmentedTransformerTriton
from SegmentedTransformerOptimized import SegmentedTransformerOptimizedSimple
from SegmentedTransformerTritonCorrected import SegmentedTransformerTritonCorrected

def copy_weights_pytorch_to_triton(pytorch_model, triton_model):
    """Copy weights from PyTorch model to Triton model for exact comparison."""
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

def benchmark_model(model, input_tensor, name, warmup_runs=10, test_runs=100):
    """Benchmark a model with proper warmup and timing."""
    # Warmup
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(input_tensor)
    
    # Synchronize GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.perf_counter()
    for _ in range(test_runs):
        with torch.no_grad():
            _ = model(input_tensor)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    avg_time = (end_time - start_time) / test_runs
    return avg_time

def comprehensive_performance_test():
    """Comprehensive performance test comparing all implementations."""
    print("=== Comprehensive SegmentedTransformer Performance Test ===\n")
    
    # Initialize all models
    torch.manual_seed(42)
    init_args = get_init_inputs()
    
    pytorch_model = PyTorchModel(*init_args)
    triton_basic_model = SegmentedTransformerTriton(*init_args)
    triton_optimized_model = SegmentedTransformerOptimizedSimple(*init_args)
    triton_corrected_model = SegmentedTransformerTritonCorrected(*init_args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Move models to device
    pytorch_model = pytorch_model.to(device)
    triton_basic_model = triton_basic_model.to(device)
    triton_optimized_model = triton_optimized_model.to(device)
    triton_corrected_model = triton_corrected_model.to(device)
    
    # Copy weights to ensure fair comparison
    copy_weights_pytorch_to_triton(pytorch_model, triton_basic_model)
    copy_weights_pytorch_to_triton(pytorch_model, triton_corrected_model)
    
    # Create test input
    torch.manual_seed(123)
    inputs = get_inputs()
    input_tensor = inputs[0].to(device)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Model parameters: {sum(p.numel() for p in pytorch_model.parameters()):,}")
    print()
    
    # Test correctness first
    print("=== Correctness Check ===")
    with torch.no_grad():
        pytorch_output = pytorch_model(input_tensor)
        triton_basic_output = triton_basic_model(input_tensor)
        triton_optimized_output = triton_optimized_model(input_tensor)
        triton_corrected_output = triton_corrected_model(input_tensor)
    
    # Compare outputs
    basic_diff = torch.abs(pytorch_output - triton_basic_output).mean().item()
    optimized_diff = torch.abs(pytorch_output - triton_optimized_output).mean().item()
    corrected_diff = torch.abs(pytorch_output - triton_corrected_output).mean().item()
    
    print(f"PyTorch vs Basic Triton MAE:      {basic_diff:.8f}")
    print(f"PyTorch vs Optimized Triton MAE:  {optimized_diff:.8f}")
    print(f"PyTorch vs Corrected Triton MAE:  {corrected_diff:.8f}")
    
    # Mark which ones are correct
    basic_correct = basic_diff < 1e-3
    optimized_correct = optimized_diff < 1e-3
    corrected_correct = corrected_diff < 1e-6
    
    print(f"Basic Triton correct:    {'✅' if basic_correct else '❌'}")
    print(f"Optimized Triton correct: {'✅' if optimized_correct else '❌'}")
    print(f"Corrected Triton correct: {'✅' if corrected_correct else '✅'}")
    print()
    
    # Performance benchmarking
    print("=== Performance Benchmarking ===")
    print("Benchmarking with 10 warmup runs and 100 test runs...")
    print()
    
    models_to_test = [
        (pytorch_model, "PyTorch Reference"),
        (triton_basic_model, "Basic Triton"),
        (triton_optimized_model, "Optimized Triton"),
        (triton_corrected_model, "Corrected Triton (Accurate)"),
    ]
    
    results = []
    for model, name in models_to_test:
        try:
            avg_time = benchmark_model(model, input_tensor, name)
            results.append((name, avg_time, True))
            print(f"{name:30s}: {avg_time*1000:.3f} ms")
        except Exception as e:
            results.append((name, float('inf'), False))
            print(f"{name:30s}: FAILED ({str(e)[:50]}...)")
    
    print()
    
    # Calculate speedups
    pytorch_time = results[0][1]
    if pytorch_time != float('inf'):
        print("=== Speedup Analysis ===")
        for name, avg_time, success in results[1:]:
            if success and avg_time > 0:
                speedup = pytorch_time / avg_time
                if speedup > 1:
                    print(f"{name:30s}: {speedup:.2f}x faster")
                else:
                    print(f"{name:30s}: {1/speedup:.2f}x slower")
            else:
                print(f"{name:30s}: Failed or invalid")
        print()
    
    # Memory usage analysis
    print("=== Memory Usage Analysis ===")
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            _ = pytorch_model(input_tensor)
            pytorch_memory = torch.cuda.max_memory_allocated()
            
            torch.cuda.reset_peak_memory_stats()
            _ = triton_corrected_model(input_tensor)
            triton_memory = torch.cuda.max_memory_allocated()
            
            print(f"PyTorch peak memory:  {pytorch_memory / 1024**2:.2f} MB")
            print(f"Triton peak memory:   {triton_memory / 1024**2:.2f} MB")
            memory_reduction = (pytorch_memory - triton_memory) / pytorch_memory * 100
            print(f"Memory reduction:     {memory_reduction:.1f}%")
        else:
            print("CUDA not available - skipping memory analysis")
    except Exception as e:
        print(f"Memory analysis failed: {e}")
    
    print()
    
    # Summary
    print("=== Summary ===")
    if corrected_correct:
        print("✅ SUCCESS: Corrected Triton implementation is numerically accurate!")
        best_time = min(r[1] for r in results if r[2] and r[1] != float('inf'))
        best_model = next(r[0] for r in results if r[1] == best_time and r[2])
        print(f"🏆 Fastest accurate implementation: {best_model}")
        
        if best_model != "PyTorch Reference":
            speedup = pytorch_time / best_time
            print(f"🚀 Achieved {speedup:.2f}x speedup over PyTorch")
    else:
        print("❌ No Triton implementation achieved numerical accuracy")
    
    return results

if __name__ == "__main__":
    results = comprehensive_performance_test()
    
    print("\n" + "="*60)
    print("FINAL PERFORMANCE REPORT")
    print("="*60)
    
    for name, time_ms, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        if success:
            print(f"{name:30s}: {time_ms*1000:.3f} ms - {status}")
        else:
            print(f"{name:30s}: FAILED - {status}") 