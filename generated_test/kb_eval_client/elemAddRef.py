import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b

batch_size = 16 * 2 * 2
dim = 16384 * 2 * 2

def get_inputs():
    x = torch.randn(batch_size, dim)
    y = torch.randn(batch_size, dim)
    return [x, y]

def get_init_inputs():
    return []  # No special initialization inputs needed 