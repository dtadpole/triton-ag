import torch
import sys
import os

# Add parent directories to path to import the implementations
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from SegmentedTransformer import Model as PyTorchModel, get_inputs, get_init_inputs
from SegmentedTransformerTritonCorrected import SegmentedTransformerTritonCorrected

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

def test_corrected_implementation():
    """Test the corrected Triton implementation against PyTorch."""
    print("=== Testing Corrected Triton Implementation ===\n")
    
    # Initialize models with same seed for reproducibility
    torch.manual_seed(42)
    init_args = get_init_inputs()
    pytorch_model = PyTorchModel(*init_args)
    
    torch.manual_seed(42)
    triton_corrected_model = SegmentedTransformerTritonCorrected(*init_args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pytorch_model = pytorch_model.to(device)
    triton_corrected_model = triton_corrected_model.to(device)
    
    # Copy weights from PyTorch model to Triton model for exact comparison
    print("Copying weights from PyTorch to corrected Triton model...")
    copy_weights_pytorch_to_triton(pytorch_model, triton_corrected_model)
    
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
        triton_corrected_output = triton_corrected_model(input_tensor)
    
    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"Corrected Triton output shape: {triton_corrected_output.shape}")
    print()
    
    # Detailed comparison
    print("=== Output Comparison (Corrected) ===")
    
    # Basic statistics
    print("PyTorch output stats:")
    print(f"  Mean: {pytorch_output.mean().item():.6f}")
    print(f"  Std:  {pytorch_output.std().item():.6f}")
    print(f"  Min:  {pytorch_output.min().item():.6f}")
    print(f"  Max:  {pytorch_output.max().item():.6f}")
    print()
    
    print("Corrected Triton output stats:")
    print(f"  Mean: {triton_corrected_output.mean().item():.6f}")
    print(f"  Std:  {triton_corrected_output.std().item():.6f}")
    print(f"  Min:  {triton_corrected_output.min().item():.6f}")
    print(f"  Max:  {triton_corrected_output.max().item():.6f}")
    print()
    
    # Difference analysis
    diff = torch.abs(pytorch_output - triton_corrected_output)
    rel_diff = diff / (torch.abs(pytorch_output) + 1e-8)
    
    print("Absolute difference stats:")
    print(f"  Mean: {diff.mean().item():.8f}")
    print(f"  Std:  {diff.std().item():.8f}")
    print(f"  Max:  {diff.max().item():.8f}")
    print(f"  99th percentile: {torch.quantile(diff, 0.99).item():.8f}")
    print()
    
    print("Relative difference stats:")
    print(f"  Mean: {rel_diff.mean().item():.8f}")
    print(f"  Std:  {rel_diff.std().item():.8f}")
    print(f"  Max:  {rel_diff.max().item():.8f}")
    print(f"  99th percentile: {torch.quantile(rel_diff, 0.99).item():.8f}")
    print()
    
    # Check if outputs are close with different tolerances
    tolerances = [
        (1e-2, 1e-2, "Relaxed"),
        (1e-3, 1e-3, "Moderate"),
        (1e-4, 1e-4, "Strict"),
        (1e-5, 1e-5, "Very Strict"),
    ]
    
    for rtol, atol, name in tolerances:
        are_close = torch.allclose(pytorch_output, triton_corrected_output, rtol=rtol, atol=atol)
        print(f"{name} tolerance (rtol={rtol}, atol={atol}): {are_close}")
    
    print()
    
    # Final verdict
    best_tolerance = torch.allclose(pytorch_output, triton_corrected_output, rtol=1e-3, atol=1e-3)
    if best_tolerance:
        print("✅ SUCCESS: Corrected Triton implementation matches PyTorch within reasonable tolerance!")
        speedup_ratio = "Ready for performance testing"
    else:
        print("⚠️  Still has differences, but may be acceptable for some use cases")
        speedup_ratio = "Needs further debugging"
    
    return best_tolerance

if __name__ == "__main__":
    success = test_corrected_implementation()
    
    print(f"\n=== Final Result ===")
    if success:
        print("🎉 The corrected Triton implementation successfully matches PyTorch!")
        print("✅ Ready for performance optimization and benchmarking")
    else:
        print("❌ Still needs more work to match PyTorch exactly")
        print("🔍 Consider:")
        print("  - Double-checking attention computation")
        print("  - Verifying weight layout and indexing")
        print("  - Checking numerical precision issues") 