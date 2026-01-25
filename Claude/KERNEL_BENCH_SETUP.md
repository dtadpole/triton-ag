# Claude Code Kernel Bench Setup Guide

Complete setup guide for Claude Code kernel bench integration with direct GPU evaluation.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LOCAL MAC                                            │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Claude Code + MCP Server                                             │   │
│  │  ┌────────────────────────┐    ┌────────────────────────────────┐    │   │
│  │  │ claudeCodeKernelBench  │───▶│ kbEvalClient                   │    │   │
│  │  │ Server.py              │    │ provider: "local"              │    │   │
│  │  │ eval_kernel()          │    │ base_url: localhost:5676       │    │   │
│  │  └────────────────────────┘    └────────────────────────────────┘    │   │
│  └───────────────────────────────────────────│──────────────────────────┘   │
│                                              │                               │
│                               SSH Tunnel (port 5676)                         │
└──────────────────────────────────────────────│───────────────────────────────┘
                                               │
┌──────────────────────────────────────────────│───────────────────────────────┐
│                                              ▼                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  kbEvalServer (port 5676)                                              │  │
│  │  ┌─────────────────────────┐                                           │  │
│  │  │  GPU Kernel Evaluation  │                                           │  │
│  │  │  Compile + Benchmark    │                                           │  │
│  │  │  Returns: speedup,      │                                           │  │
│  │  │    correctness, runtime │                                           │  │
│  │  └─────────────────────────┘                                           │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                         REMOTE GPU DEV SERVER                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Flow:**
1. Claude Code calls `eval_kernel()` MCP tool
2. MCP server uses kbEvalClient to make HTTP request to localhost:5676
3. SSH tunnel forwards request to remote kbEvalServer
4. kbEvalServer compiles and benchmarks kernel on GPU
5. Results returned: compiled, correctness, speedup, runtime

---

## Part 1: Remote GPU Dev Server Setup

Set up the kbEvalServer on a machine with NVIDIA GPU.

### Step 1.1: SSH into Remote Server

```bash
ssh devgpu001.example.com
```

### Step 1.2: Clone Repository

```bash
cd /data/users/$USER

# Get your GitHub token (create one at https://github.com/settings/tokens if needed)
export GITHUB_TOKEN=$(cat ~/.keys/github.api.key)

# Clone with token authentication on the correct branch
git clone -b claude-code-kernel-bench-plan https://${GITHUB_TOKEN}@github.com/dtadpole/triton-ag.git
cd triton-ag
```

**If you don't have a token file yet:**
```bash
mkdir -p ~/.keys
echo "ghp_xxxxxxxxxxxx" > ~/.keys/github.api.key  # Your actual token, no $ prefix
chmod 600 ~/.keys/github.api.key
```

**Note:** The token value should be the raw token (e.g., `ghp_abc123`), not prefixed with `$`. The `${GITHUB_TOKEN}` syntax is shell variable expansion - it substitutes the variable's value into the URL.

### Step 1.3: Configure Proxy (Meta devservers only)

```bash
cat >> ~/.bashrc << 'EOF'
export https_proxy=http://fwdproxy:8080
export http_proxy=http://fwdproxy:8080
export HTTPS_PROXY=http://fwdproxy:8080
export HTTP_PROXY=http://fwdproxy:8080
export no_proxy=".facebook.com,.tfbnw.net,.fb.com,localhost,127.0.0.1"
export NO_PROXY="$no_proxy"
EOF

source ~/.bashrc
```

```bash
mkdir -p ~/.config/pip

cat > ~/.config/pip/pip.conf << 'EOF'
[global]
proxy = http://fwdproxy:8080
trusted-host = pypi.org
               pypi.python.org
               files.pythonhosted.org
EOF
```

### Step 1.4: Create Virtual Environment

```bash
cd /data/users/$USER/triton-ag
python3 -m venv .venv
source .venv/bin/activate
```

### Step 1.5: Install Dependencies

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install fastapi uvicorn pydantic pyyaml ninja loguru psutil triton numpy httpx wandb
```

**Verify:**
```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import kbEvalServer; print('kbEvalServer OK')"
```

### Step 1.6: Create Required Directories

```bash
mkdir -p ~/.kbeval ~/.inference/claude_code_output ~/.keys
```

### Step 1.7: Create API Key File

```bash
mkdir -p ~/.keys
echo "937376ce-bd49-45da-9375-767cebcfcb1f" > ~/.keys/kbeval.api.key
chmod 600 ~/.keys/kbeval.api.key
```

**Verify:**
```bash
cat ~/.keys/kbeval.api.key
```

### Step 1.8: Verify kbEval.yaml Configuration

```bash
grep -A3 "^servers:" kbEval.yaml
```

**Expected output:**
```yaml
servers:
  common:
    api_key_path: ~/.keys/kbeval.api.key
```

### Step 1.9: Check Available GPUs

```bash
nvidia-smi --query-gpu=index,name,memory.free --format=csv
```

Choose a GPU with 8GB+ free memory. Note the index (0, 1, 2, etc.).

### Step 1.10: Start kbEvalServer (Interactive)

```bash
cd /data/users/$USER/triton-ag
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python3 kbEvalServer.py --local_host --port 5676 --device 0
```

Replace `CUDA_VISIBLE_DEVICES=0` and `--device 0` with your chosen GPU index.

### Step 1.11: Start kbEvalServer (Persistent with tmux)

```bash
tmux new-session -d -s kbEval "cd /data/users/$USER/triton-ag && source .venv/bin/activate && while true; do CUDA_VISIBLE_DEVICES=0 python3 kbEvalServer.py --local_host --port 5676 --device 0; sleep 5; done"
```

**Check it's running:**
```bash
tmux list-sessions
curl -s http://localhost:5676/health
```

**View logs:**
```bash
tmux attach -t kbEval
# Detach: Ctrl+B D
```

---

## Part 2: Local Mac Setup

Set up Claude Code with MCP server on your local machine.

### Step 2.1: Verify Python Version

```bash
python3 --version
```

**Required:** Python 3.10 or higher

### Step 2.2: Install Core Dependencies

```bash
python3 -m ensurepip --upgrade
python3 -m pip install pyyaml mcp torch httpx
```

**Verify:**
```bash
python3 -c "import yaml; import mcp; import torch; import httpx; print('Dependencies OK')"
```

### Step 2.3: Verify MCP Server

```bash
cd /path/to/triton-ag

python3 -c "
from claudeCodeKernelBenchServer import config, MCP_AVAILABLE, KBEVAL_AVAILABLE
print(f'MCP_AVAILABLE: {MCP_AVAILABLE}')
print(f'KBEVAL_AVAILABLE: {KBEVAL_AVAILABLE}')
"
```

**Expected:**
```
MCP_AVAILABLE: True
KBEVAL_AVAILABLE: True
```

### Step 2.4: Create API Key File

```bash
mkdir -p ~/.keys
echo "937376ce-bd49-45da-9375-767cebcfcb1f" > ~/.keys/kbeval.api.key
chmod 600 ~/.keys/kbeval.api.key
```

**Verify key was saved:**
```bash
cat ~/.keys/kbeval.api.key
```

### Step 2.5: Verify kbEval.yaml Provider Configuration

```bash
cd /path/to/triton-ag
grep -A6 "^  local:" kbEval.yaml
```

**Expected:**
```yaml
  local:
    base_url: http://localhost:5676
    api_key_path: ~/.keys/kbeval.api.key
    retry_count: 4
    initial_retry_interval: 3
    timeout: 300
```

### Step 2.6: Verify MCP Registration

```bash
cat .mcp.json
```

**Expected:**
```json
{
  "mcpServers": {
    "kernel-bench": {
      "command": "python3",
      "args": ["claudeCodeKernelBenchServer.py"],
      "cwd": "."
    }
  }
}
```

### Step 2.7: Create Output Directory

```bash
mkdir -p ~/.inference/claude_code_output
```

---

## Part 3: Connecting Local to Remote (SSH Tunnel)

Create an SSH tunnel to forward local port 5676 to the remote kbEvalServer.

### Step 3.1: Open SSH Tunnel (Single GPU)

Open a **dedicated terminal window** and run:

```bash
ssh -L 5676:localhost:5676 -N devgpu001.example.com
```

Replace `devgpu001.example.com` with your remote server hostname.

**This terminal must stay open** while using Claude Code.

### Step 3.2: Open SSH Tunnel (Multi-GPU)

For Phase 6 multi-GPU support, forward multiple ports for different kbEvalServers:

**Option A: Multiple ports to same server (different GPUs on same machine):**
```bash
ssh -L 5676:localhost:8082 -L 5677:localhost:8081 -N devvm8491.cco0.facebook.com
```

**Option B: Multiple SSH connections to different servers:**
```bash
# Terminal 1: GPU server 1
ssh -L 5676:localhost:8082 -N devvm8491.cco0.facebook.com

# Terminal 2: GPU server 2
ssh -L 5677:localhost:8082 -N devvm8492.cco0.facebook.com
```

**kbEval.yaml configuration for multi-GPU:**
```yaml
providers:
  local:
    base_url: http://localhost:5676  # GPU 1
    api_key_path: ~/.keys/kbeval.api.key
    timeout: 300
  local_2:
    base_url: http://localhost:5677  # GPU 2
    api_key_path: ~/.keys/kbeval.api.key
    timeout: 300
```

**Verify both connections:**
```bash
curl http://localhost:5676/health && echo " (GPU 1 OK)"
curl http://localhost:5677/health && echo " (GPU 2 OK)"
```

### Step 3.3: Verify Tunnel is Active

In a **different terminal**:

```bash
lsof -i :5676 | grep ssh
```

**Expected:** Shows ssh process listening on port 5676

### Step 3.3: Test Connection Through Tunnel

```bash
curl -s http://localhost:5676/health
```

**Expected:** HTTP 200 response (indicates kbEvalServer is reachable)

### Step 3.4: Test kbEvalClient Connection

```bash
cd /path/to/triton-ag

python3 -c "
from claudeCodeKernelBenchServer import get_kbeval_client
client = get_kbeval_client('kbEval.yaml')
print(f'Client type: {type(client).__name__}')
config = client._provider_config_from_yaml('local')
print(f'Base URL: {config[\"base_url\"]}')
print('Connection OK')
"
```

**Expected:**
```
Client type: KbEvalClient
Base URL: http://localhost:5676
Connection OK
```

---

## Part 4: Verification Test

Run a complete end-to-end test to verify everything works.

### Step 4.1: Test Direct Kernel Evaluation

```bash
cd /path/to/triton-ag

python3 << 'EOF'
import asyncio
from claudeCodeKernelBenchServer import eval_kernel

kernel = '''
import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.maximum(x, 0.0), mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        out = torch.empty_like(x)
        n = x.numel()
        relu_kernel[(n // 1024 + 1,)](x.view(-1), out.view(-1), n, BLOCK=1024)
        return out
'''

result = asyncio.run(eval_kernel(
    task_path='level1/19_ReLU.py',
    kernel_code=kernel,
    session_id='setup_test',
    iteration=0,
    provider='local',
    queue_only=False,
    code_type='triton'
))

print('=== Eval Result ===')
print(f'compiled: {result.get("compiled")}')
print(f'correctness: {result.get("correctness")}')
print(f'speedup: {result.get("speedup", "N/A")}')
print(f'runtime: {result.get("runtime", "N/A")}')
if result.get('metadata', {}).get('hardware'):
    print(f'hardware: {result["metadata"]["hardware"]}')
EOF
```

**Expected:**
```
=== Eval Result ===
compiled: True
correctness: True
speedup: 1.XX
runtime: X.XX
hardware: NVIDIA H100 80GB HBM3
```

### Step 4.2: Start Claude Code

```bash
cd /path/to/triton-ag
claude
```

### Step 4.3: Verify MCP Tools in Claude Code

In Claude Code, type:
```
List the MCP tools with "kernel" in the name
```

**Expected tools:**
- `mcp__kernel-bench__list_kernel_bench_tasks`
- `mcp__kernel-bench__get_task_details`
- `mcp__kernel-bench__eval_kernel`
- `mcp__kernel-bench__save_benchmark_result`
- `mcp__kernel-bench__get_session_summary`

### Step 4.4: Test eval_kernel via MCP

In Claude Code, type:
```
Use eval_kernel to evaluate a simple ReLU Triton kernel for level1/19_ReLU.py.
Use session_id="mcp_test" and provider="local".
```

**Expected:** Claude generates a kernel, submits it, and receives compilation/correctness results.

---

## Configuration Reference

### Sync Requirements

All three must use the **same port**:

| Component | Configuration | Value |
|-----------|---------------|-------|
| kbEvalServer | `--port` flag | 5676 |
| kbEval.yaml | `providers.local.base_url` | http://localhost:5676 |
| SSH Tunnel | `-L` flag | 5676:localhost:5676 |

API key must be **identical** on both machines:

| Machine | File Path |
|---------|-----------|
| Remote GPU server | `~/.keys/kbeval.api.key` |
| Local Mac | `~/.keys/kbeval.api.key` |

### Key Configuration Files

| File | Purpose |
|------|---------|
| `kbEval.yaml` | kbEvalServer/Client config (ports, API keys, timeouts) |
| `.mcp.json` | MCP server registration for Claude Code |
| `claudeCodeKernelBench.yaml` | MCP server settings (provider, paths) |

---

## Troubleshooting

### Connection Refused (localhost:5676)

**Check SSH tunnel is running:**
```bash
lsof -i :5676 | grep ssh
```

If no output, restart the tunnel:
```bash
ssh -L 5676:localhost:5676 -N devgpu001.example.com
```

**Check remote kbEvalServer is running:**
```bash
ssh devgpu001.example.com "curl -s http://localhost:5676/health"
```

If no response, restart kbEvalServer on remote:
```bash
ssh devgpu001.example.com "tmux attach -t kbEval"
```

### API Key Authentication Failed

**Verify keys match:**
```bash
# Local key:
cat ~/.keys/kbeval.api.key

# Remote key:
ssh devgpu001.example.com "cat ~/.keys/kbeval.api.key"
```

Both must be identical. If different, copy remote key to local:
```bash
ssh devgpu001.example.com "cat ~/.keys/kbeval.api.key" > ~/.keys/kbeval.api.key
chmod 600 ~/.keys/kbeval.api.key
```

### GPU Out of Memory

**Check GPU memory on remote:**
```bash
ssh devgpu001.example.com "nvidia-smi --query-gpu=index,memory.free --format=csv"
```

**Find processes using GPU:**
```bash
ssh devgpu001.example.com "nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
```

**Switch to different GPU** (e.g., GPU 1):
```bash
ssh devgpu001.example.com "tmux kill-session -t kbEval"
ssh devgpu001.example.com "tmux new-session -d -s kbEval 'cd /data/users/\$USER/triton-ag && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=1 python3 kbEvalServer.py --local_host --port 5676 --device 0'"
```

### Kernel Compilation Timeout

If you see errors like:
```
kbEvalCli.py could not generate the result file in time
```

This usually means Triton kernel compilation is taking longer than the timeout (default: 180 seconds). The first compilation of a new kernel can take 1-2 minutes. Large kernels with big tensors (e.g., >1GB inputs) may need even more time.

**Increase timeout** when starting the server:
```bash
CUDA_VISIBLE_DEVICES=0 python3 kbEvalServer.py --local_host --port 5676 --device 0 --max_critical_time 240
```

**Note:** Subsequent runs with the same kernel will be faster due to Triton's compilation cache.

### MCP Tools Not Available in Claude Code

**Check MCP server syntax:**
```bash
python3 -m py_compile claudeCodeKernelBenchServer.py && echo "Syntax OK"
```

**Check .mcp.json exists:**
```bash
ls -la .mcp.json
```

**Restart Claude Code from project directory:**
```bash
cd /path/to/triton-ag
claude
```

### Stale MCP Server After Code Changes

After modifying `claudeCodeKernelBenchServer.py` or `kbEval.yaml`:

```bash
# Exit Claude Code (Ctrl+C or type 'exit')

# Clear Python cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# Restart Claude Code
cd /path/to/triton-ag
claude
```

---

## Quick Reference Commands

### Daily Startup Sequence

**Terminal 1 - SSH Tunnel (keep open):**
```bash
ssh -L 5676:localhost:5676 -N devgpu001.example.com
```

**Terminal 2 - Claude Code:**
```bash
cd /path/to/triton-ag
claude
```

### Check Status

```bash
# Tunnel active?
lsof -i :5676 | grep ssh && echo "Tunnel OK" || echo "Tunnel NOT running"

# kbEvalServer reachable?
curl -s http://localhost:5676/health && echo "Server OK" || echo "Server NOT reachable"

# API key exists?
test -f ~/.keys/kbeval.api.key && echo "API key OK" || echo "API key MISSING"

# MCP server OK?
python3 -c "from claudeCodeKernelBenchServer import MCP_AVAILABLE; print(f'MCP: {MCP_AVAILABLE}')"
```

### Remote Server Management

**Start kbEvalServer:**
```bash
ssh devgpu001.example.com "cd /data/users/\$USER/triton-ag && tmux new-session -d -s kbEval 'source .venv/bin/activate && CUDA_VISIBLE_DEVICES=0 python3 kbEvalServer.py --local_host --port 5676 --device 0'"
```

**Check kbEvalServer status:**
```bash
ssh devgpu001.example.com "tmux list-sessions | grep kbEval"
```

**View kbEvalServer logs:**
```bash
ssh -t devgpu001.example.com "tmux attach -t kbEval"
# Detach: Ctrl+B D
```

**Stop kbEvalServer:**
```bash
ssh devgpu001.example.com "tmux kill-session -t kbEval"
```
