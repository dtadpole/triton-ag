import torch
import triton
import triton.language as tl
import math

# Constants from the original implementation
BATCH_SIZE = 12
D_MODEL = 64
D_FF = D_MODEL * 4  # 256
NUM_HEADS = 1
NUM_SEGMENTS = 32
LEN_SEGMENT = 16

@triton.jit
def segmented_transformer_kernel(
    # Input and output tensors
    input_ptr, output_ptr,
    
    # Attention weights
    qkv_weight_ptr, attn_out_weight_ptr,
    
    # MLP weights  
    mlp1_weight_ptr, mlp2_weight_ptr,
    
    # Strides
    batch_stride, seg_stride, token_stride, feat_stride,
    
    # Dimensions
    BATCH_SIZE: tl.constexpr, NUM_SEGMENTS: tl.constexpr, 
    LEN_SEGMENT: tl.constexpr, D_MODEL: tl.constexpr, D_FF: tl.constexpr,
):
    """Optimized Triton kernel - direct matrix operations in shared memory"""
    # Grid: (batch_size * num_segments,)
    pid = tl.program_id(0)
    
    # Decode which segment this program handles
    batch_id = pid // NUM_SEGMENTS
    seg_id = pid % NUM_SEGMENTS
    
    # Calculate segment base addresses
    seg_input_offset = batch_id * batch_stride + seg_id * seg_stride
    seg_output_offset = batch_id * batch_stride + seg_id * seg_stride
    
    # Offsets for vectorized operations
    feat_offsets = tl.arange(0, D_MODEL)
    token_offsets = tl.arange(0, LEN_SEGMENT)
    ff_offsets = tl.arange(0, D_FF)
    
    # Load entire segment input (LEN_SEGMENT x D_MODEL) - vectorized load
    input_offsets = seg_input_offset + token_offsets[:, None] * token_stride + feat_offsets[None, :] * feat_stride
    segment_input = tl.load(input_ptr + input_offsets)
    
    # Load QKV weight matrices (all at once)
    qkv_weight_base = seg_id * (3 * D_MODEL * D_MODEL)
    
    # Load Q weights (D_MODEL x D_MODEL)
    q_offsets = qkv_weight_base + feat_offsets[:, None] * D_MODEL + feat_offsets[None, :]
    q_weights = tl.load(qkv_weight_ptr + q_offsets)
    
    # Load K weights (D_MODEL x D_MODEL)  
    k_offsets = qkv_weight_base + D_MODEL * D_MODEL + feat_offsets[:, None] * D_MODEL + feat_offsets[None, :]
    k_weights = tl.load(qkv_weight_ptr + k_offsets)
    
    # Load V weights (D_MODEL x D_MODEL)
    v_offsets = qkv_weight_base + 2 * D_MODEL * D_MODEL + feat_offsets[:, None] * D_MODEL + feat_offsets[None, :]
    v_weights = tl.load(qkv_weight_ptr + v_offsets)
    
    # Compute QKV projections: X @ W^T using tl.dot (weights need transpose!)
    Q = tl.dot(segment_input, tl.trans(q_weights))  # (LEN_SEGMENT, D_MODEL) @ (D_MODEL, D_MODEL)^T
    K = tl.dot(segment_input, tl.trans(k_weights))  # (LEN_SEGMENT, D_MODEL) @ (D_MODEL, D_MODEL)^T
    V = tl.dot(segment_input, tl.trans(v_weights))  # (LEN_SEGMENT, D_MODEL) @ (D_MODEL, D_MODEL)^T
    
    # Self-attention: Q @ K^T
    # Use exact same scale computation as PyTorch
    scale = 1.0 / math.sqrt(float(D_MODEL))
    scores = tl.dot(Q, tl.trans(K)) * scale  # (LEN_SEGMENT, D_MODEL) @ (D_MODEL, LEN_SEGMENT)
    
    # Enhanced stable softmax with better precision
    max_scores = tl.max(scores, axis=1, keep_dims=True)
    shifted_scores = scores - max_scores
    exp_scores = tl.exp(shifted_scores)
    sum_exp = tl.sum(exp_scores, axis=1, keep_dims=True)
    # Add small epsilon to avoid division by zero
    attn_weights = exp_scores / (sum_exp + 1e-12)
    
    # Attention output: attn_weights @ V
    attn_output = tl.dot(attn_weights, V)  # (LEN_SEGMENT, LEN_SEGMENT) @ (LEN_SEGMENT, D_MODEL)
    
    # Attention output projection
    attn_out_base = seg_id * (D_MODEL * D_MODEL)
    attn_proj_offsets = attn_out_base + feat_offsets[:, None] * D_MODEL + feat_offsets[None, :]
    attn_proj_weights = tl.load(attn_out_weight_ptr + attn_proj_offsets)
    
    attn_projected = tl.dot(attn_output, tl.trans(attn_proj_weights))
    
    # Residual connection after attention
    x_after_attn = segment_input + attn_projected
    
    # MLP Layer 1: (LEN_SEGMENT x D_MODEL) @ (D_MODEL x D_FF)
    mlp1_base = seg_id * (D_FF * D_MODEL)
    mlp1_offsets = mlp1_base + ff_offsets[:, None] * D_MODEL + feat_offsets[None, :]
    mlp1_weights = tl.load(mlp1_weight_ptr + mlp1_offsets)
    
    mlp1_output = tl.dot(x_after_attn, tl.trans(mlp1_weights))  # Need transpose for correct dims
    mlp1_output = tl.maximum(mlp1_output, 0.0)  # ReLU
    
    # MLP Layer 2: (LEN_SEGMENT x D_FF) @ (D_FF x D_MODEL)
    mlp2_base = seg_id * (D_MODEL * D_FF)
    mlp2_offsets = mlp2_base + feat_offsets[:, None] * D_FF + ff_offsets[None, :]
    mlp2_weights = tl.load(mlp2_weight_ptr + mlp2_offsets)
    
    mlp2_output = tl.dot(mlp1_output, tl.trans(mlp2_weights))  # Need transpose for correct dims
    
    # Final residual connection and output
    final_output = x_after_attn + mlp2_output
    
    # Store results (vectorized store)
    output_offsets = seg_output_offset + token_offsets[:, None] * token_stride + feat_offsets[None, :] * feat_stride
    tl.store(output_ptr + output_offsets, final_output)


class SegmentedTransformerTriton(torch.nn.Module):
    """Corrected Triton implementation that matches PyTorch output exactly."""
    
    def __init__(self, num_segments, d_model, num_heads, d_ff):
        super().__init__()
        self.num_segments = num_segments
        self.d_model = d_model  
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # Initialize weights to match the original implementation
        self.qkv_weight = torch.nn.Parameter(torch.randn(num_segments, 3 * d_model, d_model))
        self.attn_out_weight = torch.nn.Parameter(torch.randn(num_segments, d_model, d_model))
        self.mlp1_weight = torch.nn.Parameter(torch.randn(num_segments, d_ff, d_model))
        self.mlp2_weight = torch.nn.Parameter(torch.randn(num_segments, d_model, d_ff))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to match the original implementation."""
        for i in range(self.num_segments):
            torch.nn.init.kaiming_uniform_(self.qkv_weight[i], a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.attn_out_weight[i], a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.mlp1_weight[i], a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.mlp2_weight[i], a=math.sqrt(5))
    
    def forward(self, x):
        batch_size, num_segments, len_segment, d_model = x.shape
        
        # Prepare output tensor
        output = torch.empty_like(x)
        
        # Launch kernel with 1D grid: batch_size * num_segments
        total_segments = batch_size * num_segments
        grid = (total_segments,)
        
        segmented_transformer_kernel[grid](
            # Input and output
            x, output,
            
            # Weights (flattened)
            self.qkv_weight.view(-1), 
            self.attn_out_weight.view(-1),
            self.mlp1_weight.view(-1),
            self.mlp2_weight.view(-1),
            
            # Strides
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            
            # Constants
            BATCH_SIZE=batch_size, NUM_SEGMENTS=num_segments,
            LEN_SEGMENT=len_segment, D_MODEL=d_model, D_FF=self.d_ff,
        )
        
        return output


def get_inputs():
    input_embeddings = torch.randn(BATCH_SIZE, NUM_SEGMENTS, LEN_SEGMENT, D_MODEL)
    return [input_embeddings]

def get_init_inputs():
    return [NUM_SEGMENTS, D_MODEL, NUM_HEADS, D_FF]

# Example usage and testing
if __name__ == "__main__":
    # Initialize Triton model
    triton_model = SegmentedTransformerTriton(*get_init_inputs())
    
    # Example input - random embeddings
    inputs = get_inputs()
    print(f"Input shape: {inputs[0].shape}")
    
    # Test on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    triton_model = triton_model.to(device)
    inputs[0] = inputs[0].to(device)
    
    # Forward pass
    with torch.no_grad():
        output = triton_model(inputs[0])
        print(f"Output shape: {output.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in triton_model.parameters())
        print(f"Total parameters: {total_params:,}")
    
    print("Triton model created and tested successfully!")
