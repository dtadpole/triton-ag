# bench_inference_mode_vs_no_grad.py
import time, torch, torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    f"Torch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | Device: {device}"
)


# small MLP; keep eval() so Dropout/BN are deterministic across runs
class TinyMLP(nn.Module):
    def __init__(self, d=2048, h=2048 * 4, depth=2):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers += [nn.Linear(d, h), nn.GELU()]
            d = h
        layers.append(nn.Linear(h, d))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model = TinyMLP().to(device).eval()
x = torch.randn(32, 2048, device=device)


def bench(ctx_name, ctx_mgr, iters=100, warmup=20):
    with ctx_mgr():
        for _ in range(warmup):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with ctx_mgr():
        ge = torch.is_grad_enabled()
        for _ in range(iters):
            y = model(x)
            _ = float(y.mean())  # touch result
        if device.type == "cuda":
            torch.cuda.synchronize()
    t1 = time.perf_counter()
    return ctx_name, (t1 - t0) * 1000 / iters, ge, y


# correctness check: outputs equal in eval() (no dropout)
with torch.no_grad():
    y_nd = model(x).detach()
with torch.inference_mode():
    y_inf = model(x)
max_abs_diff = (y_nd - y_inf).abs().max().item()

rows = []
for name, mgr in [("no_grad", torch.no_grad), ("inference_mode", torch.inference_mode)]:
    rows.append(bench(name, mgr))

print("\nResults (lower is better):")
for name, ms, grad_enabled, _ in rows:
    print(f"{name:>15}: {ms:.3f} ms/forward | grad_enabled={grad_enabled}")
print(f"\nOutput consistency (eval mode): max |Δ| = {max_abs_diff:.3e}")
