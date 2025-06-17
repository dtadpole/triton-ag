import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn


class SegmentedLinear(nn.Module):
    """
    Linear layer where each segment has its own weight matrix and bias.
    
    Args:
        num_segments: Number of separate linear layers (one per segment)
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to include bias terms
        
    Input shape: (..., num_segments, len_segment, in_features)
    Output shape: (..., num_segments, len_segment, out_features)
    """
    
    def __init__(self, num_segments, in_features, out_features, bias=True):
        super().__init__()
        self.num_segments = num_segments
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight tensor: (num_segments, out_features, in_features)
        # Each segment gets its own weight matrix
        self.weight = nn.Parameter(torch.randn(num_segments, out_features, in_features))
        
        if bias:
            # Bias tensor: (num_segments, out_features)
            self.bias = nn.Parameter(torch.randn(num_segments, out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize each segment's weights separately
        for i in range(self.num_segments):
            nn.init.kaiming_uniform_(self.weight[i], a=5**0.5)
        
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        """
        Forward pass using einsum - no weight expansion needed.
        
        Args:
            x: Input tensor of shape (..., num_segments, len_segment, in_features)
        Returns:
            Output tensor of shape (..., num_segments, len_segment, out_features)
        """
        # Einsum: '...si,soi->...so'
        # ... = batch dimensions (flexible)
        # s = num_segments
        # i = in_features  
        # o = out_features
        output = torch.einsum('...sli,soi->...slo', x, self.weight)
        
        if self.bias is not None:
            output = output + self.bias
            
        return output

class SegmentedMultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, num_segments: int, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Combined QKV projection
        self.w_qkv = SegmentedLinear(num_segments, d_model, 3 * d_model, bias=False)
        self.w_o = SegmentedLinear(num_segments, d_model, d_model, bias=False)
        
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_segments, len_segment, d_model = x.size()
        
        # Combined QKV projection and split
        qkv = self.w_qkv(x)  # (batch_size, len_segment, num_segments, 3 * d_model)
        Q, K, V = qkv.chunk(3, dim=-1)  # Each: (batch_size, len_segment, num_segments, d_model)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, num_segments, len_segment, self.num_heads, self.d_k).transpose(-2, -3)  # (batch_size, num_segments, num_heads, len_segment, d_k)
        K = K.view(batch_size, num_segments, len_segment, self.num_heads, self.d_k).transpose(-2, -3)  # (batch_size, num_segments, num_heads, len_segment, d_k)
        V = V.view(batch_size, num_segments, len_segment, self.num_heads, self.d_k).transpose(-2, -3)  # (batch_size, num_segments, num_heads, len_segment, d_k)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch_size, num_segments, num_heads, len_segment, len_segment)
        
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)  # (batch_size, num_segments, num_heads, len_segment, d_k)
        
        # Reshape and concatenate heads
        attention_output = attention_output.transpose(-2, -3).contiguous().view(batch_size, num_segments, len_segment, d_model)
        
        # Final linear projection
        output = self.w_o(attention_output)
        
        return output


class SegmentedMLP(nn.Module):
    """Multi-layer perceptron (feed-forward network)."""
    
    def __init__(self, num_segments: int, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = SegmentedLinear(num_segments, d_model, d_ff, bias=False)
        self.linear2 = SegmentedLinear(num_segments, d_ff, d_model, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.relu(self.linear1(x)))


class Model(nn.Module):
    """Single transformer layer with attention and MLP."""
    
    def __init__(self, num_segments: int, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.attention = SegmentedMultiHeadAttention(num_segments, d_model, num_heads)
        self.mlp = SegmentedMLP(num_segments, d_model, d_ff)

        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, SegmentedLinear):
                # SegmentedLinear handles its own initialization in reset_parameters()
                pass
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-head attention with residual connection
        attn_output = self.attention(x)
        x = x + attn_output
        
        # MLP with residual connection
        mlp_output = self.mlp(x)
        x = x + mlp_output
        
        return x

BATCH_SIZE = 12 * 10
D_MODEL = 64
D_FF = D_MODEL * 4
NUM_HEADS = 1
NUM_SEGMENTS = 32
LEN_SEGMENT = 16

def get_inputs():
    input_embeddings = torch.randn(BATCH_SIZE, NUM_SEGMENTS, LEN_SEGMENT, D_MODEL)
    return [input_embeddings]

def get_init_inputs():
    return [NUM_SEGMENTS, D_MODEL, NUM_HEADS, D_FF]

# Example usage and testing
if __name__ == "__main__":

    # Initialize model
    transformer = Model(*get_init_inputs())
    
    # Example input - random embeddings
    inputs = get_inputs()
    print(f"Input shape: {inputs[0].shape}")
    
    # Forward pass
    with torch.no_grad():
        # Apply transformer layer directly to input embeddings
        output = transformer(inputs[0])
        print(f"Output shape: {output.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in transformer.parameters())
        print(f"Total parameters: {total_params:,}")
    
    print("Model created and tested successfully!")
