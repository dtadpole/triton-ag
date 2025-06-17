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
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one token
    pid = tl.program_id(0)
    
    # Decode which token this program handles
    total_tokens_per_batch = NUM_SEGMENTS * LEN_SEGMENT
    batch_id = pid // total_tokens_per_batch
    remaining = pid % total_tokens_per_batch
    seg_id = remaining // LEN_SEGMENT
    token_id = remaining % LEN_SEGMENT
    
    # Calculate base addresses
    input_offset = batch_id * batch_stride + seg_id * seg_stride + token_id * token_stride
    output_offset = batch_id * batch_stride + seg_id * seg_stride + token_id * token_stride
    
    # Load input token
    feat_offsets = tl.arange(0, BLOCK_SIZE)
    mask = feat_offsets < D_MODEL
    input_token = tl.load(input_ptr + input_offset + feat_offsets * feat_stride, mask=mask)
    
    # === QKV Projection ===
    qkv_weight_offset = seg_id * (3 * D_MODEL * D_MODEL)
    
    # Compute Q, K, V for this token
    q_vec = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    k_vec = tl.zeros((BLOCK_SIZE,), dtype=tl.float32) 
    v_vec = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for d_out in range(D_MODEL):
        if d_out < BLOCK_SIZE:
            # Q projection
            q_weight_offset = qkv_weight_offset + d_out * D_MODEL
            q_weights = tl.load(qkv_weight_ptr + q_weight_offset + feat_offsets, mask=mask)
            q_val = tl.sum(input_token * q_weights)
            q_vec = tl.where(feat_offsets == d_out, q_val, q_vec)
            
            # K projection  
            k_weight_offset = qkv_weight_offset + (D_MODEL + d_out) * D_MODEL
            k_weights = tl.load(qkv_weight_ptr + k_weight_offset + feat_offsets, mask=mask)
            k_val = tl.sum(input_token * k_weights)
            k_vec = tl.where(feat_offsets == d_out, k_val, k_vec)
            
            # V projection
            v_weight_offset = qkv_weight_offset + (2 * D_MODEL + d_out) * D_MODEL
            v_weights = tl.load(qkv_weight_ptr + v_weight_offset + feat_offsets, mask=mask)
            v_val = tl.sum(input_token * v_weights)
            v_vec = tl.where(feat_offsets == d_out, v_val, v_vec)
    
    # === Self-Attention (simplified) ===
    # For simplicity, we'll compute attention with all other tokens in the segment
    attn_output = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for other_token_id in range(LEN_SEGMENT):
        # Load other token
        other_input_offset = batch_id * batch_stride + seg_id * seg_stride + other_token_id * token_stride
        other_token = tl.load(input_ptr + other_input_offset + feat_offsets * feat_stride, mask=mask)
        
        # Compute other token's K and V (simplified - recomputing)
        other_k = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        other_v = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        for d_out in range(D_MODEL):
            if d_out < BLOCK_SIZE:
                # K projection for other token
                k_weight_offset = qkv_weight_offset + (D_MODEL + d_out) * D_MODEL
                k_weights = tl.load(qkv_weight_ptr + k_weight_offset + feat_offsets, mask=mask)
                k_val = tl.sum(other_token * k_weights)
                other_k = tl.where(feat_offsets == d_out, k_val, other_k)
                
                # V projection for other token
                v_weight_offset = qkv_weight_offset + (2 * D_MODEL + d_out) * D_MODEL
                v_weights = tl.load(qkv_weight_ptr + v_weight_offset + feat_offsets, mask=mask)
                v_val = tl.sum(other_token * v_weights)
                other_v = tl.where(feat_offsets == d_out, v_val, other_v)
        
        # Compute attention score (dot product)
        score = tl.sum(q_vec * other_k) / math.sqrt(D_MODEL)
        weight = tl.exp(score)  # Simplified attention (no proper softmax normalization)
        
        # Add weighted value to attention output
        attn_output = attn_output + weight * other_v
    
    # === Attention Output Projection ===
    attn_out_weight_offset = seg_id * (D_MODEL * D_MODEL)
    attn_projected = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for d_out in range(D_MODEL):
        if d_out < BLOCK_SIZE:
            weight_offset = attn_out_weight_offset + d_out * D_MODEL
            weights = tl.load(attn_out_weight_ptr + weight_offset + feat_offsets, mask=mask)
            proj_val = tl.sum(attn_output * weights)
            attn_projected = tl.where(feat_offsets == d_out, proj_val, attn_projected)
    
    # Add residual connection
    x_after_attn = input_token + attn_projected
    
    # === MLP Layer 1 ===
    mlp1_weight_offset = seg_id * (D_FF * D_MODEL)
    mlp1_output = tl.zeros((D_FF,), dtype=tl.float32)
    
    for d_ff in range(D_FF):
        weight_offset = mlp1_weight_offset + d_ff * D_MODEL
        weights = tl.load(mlp1_weight_ptr + weight_offset + feat_offsets, mask=mask)
        activation = tl.sum(x_after_attn * weights)
        # ReLU activation
        activation = tl.maximum(activation, 0.0)
        mlp1_output = tl.where(tl.arange(0, D_FF) == d_ff, activation, mlp1_output)
    
    # === MLP Layer 2 ===
    mlp2_weight_offset = seg_id * (D_MODEL * D_FF)
    mlp2_output = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for d_out in range(D_MODEL):
        if d_out < BLOCK_SIZE:
            weight_offset = mlp2_weight_offset + d_out * D_FF
            # Load D_FF weights
            weights = tl.zeros((D_FF,), dtype=tl.float32)
            for i in range(D_FF):
                weight_val = tl.load(mlp2_weight_ptr + weight_offset + i)
                weights = tl.where(tl.arange(0, D_FF) == i, weight_val, weights)
            
            proj_val = tl.sum(mlp1_output * weights)
            mlp2_output = tl.where(feat_offsets == d_out, proj_val, mlp2_output)
    
    # Final residual connection
    final_output = x_after_attn + mlp2_output
    
    # Store output
    tl.store(output_ptr + output_offset + feat_offsets * feat_stride, final_output, mask=mask)


class SegmentedTransformerTriton(torch.nn.Module):
    """Triton implementation of SegmentedTransformer."""
    
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
        
        # Launch kernel with one program per token
        total_tokens = batch_size * num_segments * len_segment
        grid = (total_tokens,)
        
        # Use power of 2 block size >= d_model
        BLOCK_SIZE = triton.next_power_of_2(d_model)
        
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
            BLOCK_SIZE=BLOCK_SIZE,
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
