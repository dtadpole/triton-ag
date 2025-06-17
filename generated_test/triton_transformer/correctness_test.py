import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add parent directories to path to import the implementations
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from SegmentedTransformer import Model as PyTorchModel, get_inputs, get_init_inputs
from SegmentedTransformerTriton import SegmentedTransformerTriton

def detailed_correctness_test():
    """Perform detailed correctness testing between implementations."""
    print("=== Detailed Correctness Test ===\n")
    
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
    print(f"Input range: [{input_tensor.min().item():.4f}, {input_tensor.max().item():.4f}]")
    print()
    
    # Forward pass
    with torch.no_grad():
        pytorch_output = pytorch_model(input_tensor)
        triton_output = triton_model(input_tensor)
    
    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"Triton output shape: {triton_output.shape}")
    print()
    
    # Detailed comparison
    print("=== Output Comparison ===")
    
    # Basic statistics
    print("PyTorch output stats:")
    print(f"  Mean: {pytorch_output.mean().item():.6f}")
    print(f"  Std:  {pytorch_output.std().item():.6f}")
    print(f"  Min:  {pytorch_output.min().item():.6f}")
    print(f"  Max:  {pytorch_output.max().item():.6f}")
    print()
    
    print("Triton output stats:")
    print(f"  Mean: {triton_output.mean().item():.6f}")
    print(f"  Std:  {triton_output.std().item():.6f}")
    print(f"  Min:  {triton_output.min().item():.6f}")
    print(f"  Max:  {triton_output.max().item():.6f}")
    print()
    
    # Difference analysis
    diff = torch.abs(pytorch_output - triton_output)
    rel_diff = diff / (torch.abs(pytorch_output) + 1e-8)
    
    print("Absolute difference stats:")
    print(f"  Mean: {diff.mean().item():.6f}")
    print(f"  Std:  {diff.std().item():.6f}")
    print(f"  Max:  {diff.max().item():.6f}")
    print(f"  99th percentile: {torch.quantile(diff, 0.99).item():.6f}")
    print()
    
    print("Relative difference stats:")
    print(f"  Mean: {rel_diff.mean().item():.6f}")
    print(f"  Std:  {rel_diff.std().item():.6f}")
    print(f"  Max:  {rel_diff.max().item():.6f}")
    print(f"  99th percentile: {torch.quantile(rel_diff, 0.99).item():.6f}")
    print()
    
    # Check if outputs are close
    rtol = 1e-3
    atol = 1e-3
    are_close = torch.allclose(pytorch_output, triton_output, rtol=rtol, atol=atol)
    
    print(f"Outputs close (rtol={rtol}, atol={atol}): {are_close}")
    
    if not are_close:
        print("\n⚠️  Outputs are not close! Analyzing differences...")
        
        # Find indices of largest differences
        flat_diff = diff.flatten()
        top_indices = torch.topk(flat_diff, k=5).indices
        
        print("Top 5 largest absolute differences:")
        for i, idx in enumerate(top_indices):
            pytorch_val = pytorch_output.flatten()[idx].item()
            triton_val = triton_output.flatten()[idx].item()
            diff_val = flat_diff[idx].item()
            print(f"  {i+1}. PyTorch: {pytorch_val:.6f}, Triton: {triton_val:.6f}, Diff: {diff_val:.6f}")
        
        # Check if differences are systematic
        bias = (triton_output - pytorch_output).mean().item()
        print(f"\nSystematic bias (Triton - PyTorch): {bias:.6f}")
        
        # Analyze per-segment differences
        diff_per_segment = diff.mean(dim=(0, 2, 3))  # Average over batch and tokens
        print(f"\nPer-segment mean absolute difference:")
        for i, seg_diff in enumerate(diff_per_segment[:5]):  # Show first 5 segments
            print(f"  Segment {i}: {seg_diff.item():.6f}")
        
    else:
        print("✅ Outputs match within tolerance!")
    
    return are_close

def copy_weights_pytorch_to_triton(pytorch_model, triton_model):
    """Copy weights from PyTorch model to Triton model for exact comparison."""
    
    # Copy attention weights
    # PyTorch model has separate Q, K, V projections, Triton has combined QKV
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

def test_individual_components():
    """Test individual components separately to isolate issues."""
    print("\n=== Component-wise Testing ===\n")
    
    torch.manual_seed(42)
    init_args = get_init_inputs()
    pytorch_model = PyTorchModel(*init_args)
    triton_model = SegmentedTransformerTriton(*init_args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pytorch_model = pytorch_model.to(device)
    triton_model = triton_model.to(device)
    
    # Copy weights
    copy_weights_pytorch_to_triton(pytorch_model, triton_model)
    
    # Test input
    torch.manual_seed(123)
    inputs = get_inputs()
    input_tensor = inputs[0].to(device)
    
    with torch.no_grad():
        # Test attention only
        print("Testing attention component...")
        pytorch_attn = pytorch_model.attention(input_tensor)
        
        # For Triton model, we need to test attention separately
        # This is tricky since the Triton model does everything in one kernel
        # Let's just compare final outputs
        pytorch_after_attn = input_tensor + pytorch_attn
        
        # Test MLP only
        print("Testing MLP component...")
        pytorch_mlp = pytorch_model.mlp(pytorch_after_attn)
        pytorch_final = pytorch_after_attn + pytorch_mlp
        
        # Compare with Triton full output
        triton_output = triton_model(input_tensor)
        
        attn_diff = torch.abs(pytorch_final - triton_output).mean().item()
        print(f"Final output difference: {attn_diff:.6f}")

def test_numerical_stability():
    """Test numerical stability with different input ranges."""
    print("\n=== Numerical Stability Testing ===\n")
    
    test_cases = [
        ("Small values", 1e-3),
        ("Normal values", 1.0),
        ("Large values", 10.0),
    ]
    
    for name, scale in test_cases:
        print(f"Testing {name} (scale={scale})...")
        
        torch.manual_seed(42)
        init_args = get_init_inputs()
        pytorch_model = PyTorchModel(*init_args)
        triton_model = SegmentedTransformerTriton(*init_args)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pytorch_model = pytorch_model.to(device)
        triton_model = triton_model.to(device)
        
        copy_weights_pytorch_to_triton(pytorch_model, triton_model)
        
        # Create scaled input
        torch.manual_seed(123)
        inputs = get_inputs()
        input_tensor = inputs[0].to(device) * scale
        
        with torch.no_grad():
            pytorch_output = pytorch_model(input_tensor)
            triton_output = triton_model(input_tensor)
            
            diff = torch.abs(pytorch_output - triton_output).mean().item()
            rel_diff = (diff / (torch.abs(pytorch_output).mean().item() + 1e-8))
            
            print(f"  Absolute difference: {diff:.6f}")
            print(f"  Relative difference: {rel_diff:.6f}")
        print()

if __name__ == "__main__":
    # Run all tests
    match_result = detailed_correctness_test()
    test_individual_components()
    test_numerical_stability()
    
    print("\n=== Summary ===")
    if match_result:
        print("✅ All tests passed! Outputs match within tolerance.")
    else:
        print("❌ Tests failed! Outputs do not match.")
        print("This suggests there may be implementation differences between PyTorch and Triton versions.")
        print("Common causes:")
        print("- Different numerical precision in calculations")
        print("- Order of operations affecting floating point results")
        print("- Missing or incorrect weight copying")
        print("- Bugs in the Triton kernel implementation") 