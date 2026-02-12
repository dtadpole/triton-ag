# other patterns
<!-- Updated: 2026-02-12 | Top 1 by speedup -->

# Reflection: 73_Conv2d_BatchNorm_Scaling

## Task
Conv2d -> BatchNorm2d -> Scale (multiply by 2.0)
Input: (128, 8, 128, 128), out_channels=64, kernel_size=3

## Result
Best speedup: 0.39x (iteration 6)
Reference runtime: 2.08ms

## Key Findings
