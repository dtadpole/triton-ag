# Common Patterns
<!-- Updated: 2026-02-12 | Source: 0212_v3_l3, merged with 0212_l2 -->

## Environment Constraints
<!-- Things that DON'T work due to server/Triton/PyTorch environment -->

- **tl.math.tanh and tl.libdevice.tanh do not exist**: The eval server's Triton version lacks these functions. Workaround: use `tanh(x) = 2*sigmoid(2*x) - 1` or `(exp(2x)-1)/(exp(2x)+1)`. The sigmoid-based form is preferred (uses Triton's built-in sigmoid). Caution: the approximation accumulates error over deep sequential computation (e.g., 512 timesteps x 6 RNN layers -> max_diff=1.03).
  (Source: L2: 11_ConvTranspose2d, 22_Matmul_Scale, 46_Conv2d_Subtract_Tanh; L3: 33_VanillaRNN, 42_GRUBidirectionalHidden)

- **Triton "cpu tensor" pointer error on non-cuda:0 devices**: Triton kernels fail with "Pointer argument cannot be accessed from Triton (cpu tensor?)" when the eval server assigns cuda:1/2/3. Fix: wrap the forward method with `with torch.cuda.device(x.device):` or call `torch.cuda.set_device(x.device)` before any Triton kernel launch. This is critical -- 50-80% of eval attempts land on non-cuda:0.
  (Source: nearly every task; most severely L2: 84_Gemm_BatchNorm)

- **Eval server blocks nn.* module strings via source code string matching**: Blocks nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.ReLU6, nn.Dropout, nn.LayerNorm, nn.GELU, nn.Softmax, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool1d, nn.Sequential, nn.Identity, nn.LSTM, nn.GRU, nn.TransformerEncoderLayer, nn.TransformerEncoder, nn.MultiheadAttention -- even in comments or unused variable names. Workaround: use nn.Parameter + nn.init + functional API instead. (Do NOT use `getattr(nn, ...)` string concatenation to bypass this check — that is reward hacking, banned by rule 7.)
  (Source: L3: every task; the list grows with each new task)

- **Eval server runs models in training mode**: The server does NOT call `.eval()`. For BatchNorm, `F.batch_norm(training=False)` causes correctness failures (max_diff ~10.0). Must use `F.batch_norm(training=self.training)` to match reference behavior with batch statistics.
  (Source: L2: 11_ConvTranspose2d, 73_Conv2d_BatchNorm; L3: 22_EfficientNetB0, 21_EfficientNetMBConv)

- **F.batch_norm(training=True) is slower than nn.BatchNorm2d**: The functional API lacks cuDNN's optimized fused path. Since nn.BatchNorm* is string-blocked, this creates an unrecoverable performance gap (~0.5ms per BN layer) for BN-dominated tasks. For models with 17+ BN layers (EfficientNet), this creates a ceiling of ~1.2x.
  (Source: L2: 73_Conv2d_BatchNorm at 0.39x; L3: 14_DenseNet121DenseBlock, 24_EfficientNetB2)

- **F.conv_transpose2d/3d is slower than nn.ConvTranspose***: The functional API lacks cuDNN algorithm caching. Gap is ~1.6x for large inputs. Cannot be worked around when nn module is blocked.
  (Source: L2: 2_ConvTranspose2d at 0.626x, 42_ConvTranspose2d, 49_ConvTranspose3d)

- **einops package not installed on eval server**: Tasks that import einops (Mamba2) cannot be evaluated at all because the reference model fails to load.
  (Source: L3: 48_Mamba2ReturnY, 49_Mamba2ReturnFinalState)

- **Correctness can be device-dependent**: Same code may pass on cuda:0 but fail on cuda:2 due to cuDNN algorithm differences on multi-GPU servers.
  (Source: L3: 16_DenseNet201)

- **torch.compile crashes with BatchNorm in training mode**: Crashes with "CUDAGraphs overwritten" error. Additionally, torch.compile is reward hacking (banned, rule 8) — it delegates to PyTorch's compiler instead of writing Triton kernels.
  (Source: L3: 25_ShuffleNetUnit, 6_GoogleNetInceptionModule)

## Anti-Patterns (Never Do This)

- **Triton kernel for 1-2 cheap post-conv activations**: cuDNN already fuses simple activations (ReLU, HardSwish, LeakyReLU) into the conv kernel internally. A separate Triton kernel forces materialization + extra memory round-trip. Proven slower (0.6-0.85x). For L3 full-network tasks with many conv+BN+ReLU blocks, per-block Triton kernels accumulate launch overhead and make things worse.
  (Source: L2: 1_Conv2D_ReLU, 69_Conv2d_HardSwish_ReLU at 0.646x; L3: 10_ResNet101, 20_MobileNetV2)

- **In-place Triton kernel writes to input/output tensors**: Causes non-deterministic correctness failures (typically 2/3 trials pass, 1/3 fails). Always allocate a fresh output tensor with `torch.empty_like()`.
  (Source: L2: 12_Gemm, 31_Conv2d_Min; L3: 32_ConvolutionalVisionTransformer)

- **channels_last memory format for conv + Triton**: Format conversion overhead (0.5-1ms) always exceeds cuDNN benefit. Stick with contiguous NCHW/NCDHW.
  (Source: L2: 2_ConvTranspose2d at 0.611x; L3: 10_ResNet101, 20_MobileNetV2)

- **Manual transformer/LSTM reimplementation with different parameter names**: The eval harness copies weights by state_dict keys. If your module structure differs from the reference, weight loading silently fails or mismatches, causing correctness failures.
  (Source: L3: 28_VisionTransformer, 40_GRUHidden, 38_LSTMBidirectional)

- **Manual GRU/LSTM implementation with Python loops over timesteps**: 20-30x slower than cuDNN which fuses all layers and timesteps into one optimized kernel. Use nn.Parameter + functional API to extract and use RNN weights directly.
  (Source: L3: 39_GRU at 0.05x, 41_GRUBidirectional at 0.034x)

- **fp16 autocast for small-batch RNNs (batch<=10, hidden<=256)**: cuDNN RNN is already optimized for fp32 and the dtype conversion overhead exceeds any tensor core benefit at these sizes.
  (Source: L3: 38_LSTMBidirectional, 39_GRU, 40_GRUHidden)

- **torch.amp.autocast context manager for short-runtime tasks (<5ms)**: The context manager itself has ~0.5ms overhead. Use explicit `.half()` casting instead.
  (Source: L3: 46_NetVladWithGhostClusters at 1.16x autocast vs 1.41x explicit)

## Universal Techniques

- **fp16 for tensor cores on large GEMMs/convolutions** (only when reference model outputs fp16): Enables tensor cores giving ~2-10x speedup. Two approaches: (a) `model.half()` + `x.half()` for full pipeline fp16 (better for deep models -- avoids per-op dtype management), (b) `torch.amp.autocast` for mixed precision (simpler but adds context manager overhead). Model.half() >> autocast for transformer models with many linear layers. **Output dtype must match reference** — using fp16 when the reference outputs fp32 is reward hacking (banned, rule 11).
  (Source: L2: 64_Gemm at 10.25x; L3: 1_MLP at 3.6x, 11_VGG16 at 1.58x, 29_SwinMLP at 3.19x)

- **Matmul epilogue fusion**: For Gemm + pointwise ops, fuse bias/activation/scaling into the Triton matmul epilogue. Consistently gives 2-10x for medium-to-large GEMMs.
  (Source: L2: 22_Matmul at 6.56x, 30_Gemm at 7.12x, and 13+ other tasks)

- **Algebraic simplification before writing any kernel**: Check for: (1) sum/mean after matmul distributes into weights, (2) dead code elimination (unused return values), (3) reorder linear ops (AvgPool before Conv1x1 when both are linear), (4) scale+residual = single multiply.
  (Source: L2: 14_Gemm at 40x, 51_Gemm at 49.5x; L3: 13_DenseNet121TransitionLayer at 1.75x, 36_LSTMHn at 4.08x)

- **torch.cuda.device(device) context manager**: Essential wrapper around any forward() that uses Triton kernels. Ensures Triton pointers resolve to the correct GPU.
  (Source: universal across all tasks)

- **torch.backends.cudnn.benchmark = True**: Set in __init__ for conv-heavy tasks. Auto-selects fastest cuDNN algorithm. Consistent ~2-5% benefit.
  (Source: L2: 54_Conv2d, 85_Conv2d; L3: 23_EfficientNetB1)
