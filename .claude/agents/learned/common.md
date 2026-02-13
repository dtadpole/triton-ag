# Common Patterns
<!-- Updated: 2026-02-13 | Source: 0212_v8_l2, merged with 0212_v3_l3+0212_l2 -->

## Environment Constraints

- **tl.math.tanh and tl.libdevice.tanh do not exist**: The eval server Triton version lacks these functions. Workaround: use `tanh(x) = 2*sigmoid(2*x) - 1` or `(exp(2x)-1)/(exp(2x)+1)`. The sigmoid-based form is preferred (uses built-in sigmoid). Caution: approximation accumulates error over deep sequential computation (e.g., 512 timesteps x 6 RNN layers -> max_diff=1.03).
  (Source: L2: 11_ConvTranspose2d, 22_Matmul_Scale, 82_Conv2d_Tanh, 95_Matmul_Add_Swish; L3: 33_VanillaRNN)

- **Triton "cpu tensor" pointer error on non-cuda:0 devices**: Triton kernels fail with "Pointer argument cannot be accessed from Triton (cpu tensor?)" when eval server assigns cuda:1/2/3. Fix: wrap forward method with `with torch.cuda.device(x.device):` before any Triton kernel launch. Critical -- 50-80% of eval attempts land on non-cuda:0.
  (Source: universal across all tasks)

- **Eval server blocks nn.* module strings AND F.*/torch.* operations**: Blocks nn.Conv2d, nn.Linear, etc. via source code string matching AND blocks F.conv2d, F.linear, F.batch_norm, torch.matmul, torch.mm, torch.bmm, etc. via both string matching and runtime detection (monkey-patching). Import aliasing will not bypass runtime detection. Workaround: use nn.Parameter + nn.init + write ALL computation in Triton kernels. (Do NOT use `getattr(nn, ...)` bypass -- reward hacking, banned by rule 7.)
  (Source: L2: 1_Conv2D_ReLU, 59_Matmul_Swish, 69_Conv2d_HardSwish, 92_Conv2d_GroupNorm)

- **Eval server runs models in training mode**: Does NOT call `.eval()`. For BatchNorm, must compute running stats manually with Bessel correction in Triton.
  (Source: L2: 11_ConvTranspose2d, 33_Gemm_Scale_BatchNorm, 77_ConvTranspose3d_Scale_BatchNorm)

- **F.batch_norm, F.conv2d, F.conv_transpose2d/3d are now BANNED**: These functional API operations are blocked by the eval server (rule 4). All computation must be written in Triton. Historical performance data about F.* vs nn.* gaps is no longer relevant since neither is allowed in forward().

- **Eval harness does NOT copy weights via load_state_dict**: It sets same random seed and instantiates both Model and ModelNew. Custom model must replicate nn.Linear random state consumption (torch.empty + kaiming_uniform_ + uniform_).
  (Source: L2: 29_Matmul_Mish -- 9 failed iterations from weight mismatch)

- **einops package not installed on eval server**: Tasks importing einops cannot be evaluated.
  (Source: L3: 48_Mamba2ReturnY)

## Anti-Patterns (Never Do This)

- **Triton kernel for 1-2 cheap post-conv activations**: cuDNN already fuses simple activations internally. A separate Triton kernel forces materialization + extra memory round-trip (0.6-0.85x). For conv tasks, focus on algebraic elimination or write the entire conv+activation pipeline in Triton.
  (Source: L2: 1_Conv2D_ReLU at 0.76x, 69_Conv2d_HardSwish, 71_Conv2d_Divide at 1.17x ceiling)

- **In-place Triton kernel writes to input/output tensors**: Causes non-deterministic correctness failures (2/3 trials pass, 1/3 fails). Always allocate fresh output tensor with `torch.empty_like()`.
  (Source: L2: 12_Gemm, 20_ConvTranspose3d, 31_Conv2d_Min, 5_ConvTranspose2d, 7_Conv3d, 91_ConvTranspose2d, 92_Conv2d_GroupNorm)

- **[Historical] F.conv* WITH bias was slower than WITHOUT**: When F.conv* was allowed, passing bias=None and fusing bias in Triton was faster. Now F.conv* is banned entirely (rule 4) — write conv computation in Triton and fuse bias into the matmul epilogue.
  (Source: L2: 1_Conv2D_ReLU, 10_ConvTranspose2d, 49_ConvTranspose3d, 52_Conv2d, 67_Conv2d_GELU, 87_Conv2d_Subtract, 93_ConvTranspose2d)

- **channels_last memory format for conv + Triton**: Format conversion overhead (0.5-1ms) always exceeds cuDNN benefit. Stick with contiguous NCHW/NCDHW.
  (Source: L2: 2_ConvTranspose2d at 0.611x)

- **Custom Triton matmul for very large square GEMMs (>8192x8192)**: cuBLAS is unbeatable for these shapes. However, since torch.mm is now banned (rule 4), you must write Triton matmul for ALL matmul tasks. For very large square GEMMs, accept that achieving 1.3x may not be feasible and focus on epilogue fusion for any speedup possible.
  (Source: L2: 55_Matmul at 0.68x Triton vs 1.025x cuBLAS, 56_Matmul at 0.81x Triton)

- **fp16 for small conv inputs (<=8 input channels, 3x3 kernel)**: Dtype conversion overhead exceeds tensor core benefit. Use TF32 instead (no conversion needed).
  (Source: L2: 4_Conv2d_Mish (fp16 ~1.0x, TF32 1.17x), 67_Conv2d_GELU (fp16 0.58-0.78x), 92_Conv2d_GroupNorm (fp16 0.50-0.63x))

- **Manual transformer/LSTM reimplementation with different parameter names**: Eval harness copies weights by state_dict keys. Weight loading silently fails.
  (Source: L3: 28_VisionTransformer, 40_GRUHidden)

## Universal Techniques

- **fp16 for tensor cores on large GEMMs/convolutions**: Enables tensor cores giving ~2-10x speedup. Cache fp16 weights in __init__ to avoid per-forward conversion overhead (L2: 31_Conv2d_Min: 1.01x uncached vs 1.33x cached). For 3x3 conv with small input channels, prefer TF32 mode instead.
  (Source: L2: 25_Conv2d at 1.74x, 46_Conv2d at 1.68x, 56_Matmul at 3.75x)

- **Matmul epilogue fusion**: For Gemm + pointwise ops, fuse bias/activation/scaling into Triton matmul epilogue. Eliminates N memory round-trips. ~60% first-try success rate. Consistently gives 3-7x.
  (Source: L2: 22_Matmul at 6.06x, 30_Gemm at 6.9x, 63_Gemm at 6.29x, 81_Gemm at 6.42x, and 20+ tasks)

- **Algebraic simplification before writing any kernel**: Check for: (1) sum/mean after matmul distributes into weights (14_Gemm 66x, 18_Matmul 47x, 51_Gemm 27x, 80_Gemm 77x), (2) dead code (min(x,0)+clamp(0,1)=0, 83_Conv3d 10.7x), (3) reorder linear ops (AvgPool before Conv1x1), (4) scale+residual = single multiply, (5) additive constants eliminated by LayerNorm centering (3_ConvTranspose3d), (6) AvgPool commutes with BN affine (72_ConvTranspose3d 4.16x), (7) mean(GN(x)) from per-channel sums (23_Conv3d, 27_Conv3d), (8) BN(x)-mean(BN(x)) = gamma/sigma*(x-mean(x)) (15_ConvTranspose3d).

- **torch.cuda.device(device) context manager**: Essential wrapper around any forward() using Triton kernels.
  (Source: universal across all tasks)

- **torch.backends.cudnn.benchmark = True**: Set in __init__ for conv-heavy tasks. Critical: 48_Conv3d without it 0.91x, with it 1.50x.
  (Source: L2: 48_Conv3d_Scaling, 54_Conv2d, 89_ConvTranspose3d)

- **Fast Mish identity**: mish(x) = x * e(e+2)/(e(e+2)+2) where e=exp(x). ONE exp() instead of exp+log+tanh. Gave 15% kernel speedup (4_Conv2d_Mish: 1.17x to 1.34x).
  (Source: L2: 4_Conv2d_Mish_Mish, 87_Conv2d_Subtract_Mish)

- **Online softmax (2-pass)**: For large spatial softmax (>10K elements), 2-pass algorithm (running max+sum pass 1, normalize pass 2) beats naive 3-pass. Use BLOCK_SIZE=8192, num_warps=16, num_stages=2 for >100K elements per program.
  (Source: L2: 38_ConvTranspose3d at 1.49x, 49_ConvTranspose3d at 1.66x, 89_ConvTranspose3d at 3.03x)

- **Recompute vs store intermediates in Triton**: Reading fp16 input and recomputing in both passes is faster than storing fp32 intermediates to global memory (input stays in L2 cache).
  (Source: L2: 38_ConvTranspose3d -- storing 1.15x, recomputing 1.49x)
