# Claude Code Kernel Bench Setup Guide

Step-by-step guide to set up a new environment for running Claude Code with the kernel bench integration.

**Prerequisites:** Claude Code installed, triton-ag codebase synced.

---

## Quick Start (5 minutes)

For basic kernel generation without GPU evaluation:

```bash
cd /path/to/triton-ag

# 1. Bootstrap pip (required on fresh Python installs)
python3 -m ensurepip --upgrade

# 2. Install Python dependencies
python3 -m pip install pyyaml mcp

# 3. Verify MCP server
python3 -c "from claudeCodeKernelBenchServer import config; print('MCP server OK')"

# 4. Start Claude Code from project directory
claude
```

Then in Claude Code:
```
List the kernel bench tasks in level1
```

---

## Full Setup

### Step 1: Verify Python Version

```bash
python3 --version
```

**Required:** Python 3.10 or higher

If Python < 3.10, install a newer version before continuing.

### Step 2: Install Core Dependencies

```bash
# Bootstrap pip if needed
python3 -m ensurepip --upgrade

# Install MCP server dependencies
python3 -m pip install pyyaml mcp
```

**Verify:**
```bash
python3 -c "import yaml; import mcp; print('Core dependencies OK')"
```

### Step 3: Verify MCP Server

```bash
cd /path/to/triton-ag

# Check syntax
python3 -m py_compile claudeCodeKernelBenchServer.py && echo "Syntax OK"

# Check imports and config
python3 -c "
from claudeCodeKernelBenchServer import config, MCP_AVAILABLE
print(f'MCP_AVAILABLE: {MCP_AVAILABLE}')
print(f'Config sections: {list(config.keys())}')
print('MCP server ready' if MCP_AVAILABLE else 'FAIL: MCP not available')
"
```

**Expected output:**
```
MCP_AVAILABLE: True
Config sections: ['kernel_bench', 'kbeval', 'workflow', 'output', 'iteration']
MCP server ready
```

### Step 4: Verify MCP Registration

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

### Step 5: Verify Kernel Bench Tasks

```bash
# Check kernel_bench directory exists
ls kernel_bench/level1/*.py | head -5
```

**Expected:** List of Python task files (e.g., `1_Square_matrix_multiplication_.py`)

If missing, the kernel_bench tasks need to be synced from KernelBench repository.

### Step 6: Create Output Directory

```bash
mkdir -p ~/.inference/claude_code_output
```

### Step 7: Start Claude Code

**Important:** Start Claude Code FROM the project directory (where `.mcp.json` is located).

```bash
cd /path/to/triton-ag
claude
```

### Step 8: Verify MCP Tools Available

In Claude Code, ask:
```
What MCP tools do you have available? List any tools with "kernel" in the name.
```

**Expected tools:**
- `list_kernel_bench_tasks`
- `get_task_details`
- `eval_kernel`
- `save_benchmark_result`
- `get_session_summary`

---

## Optional: Workflow Server Setup

Required for queue-based kernel evaluation (Test 3B+). The workflow server manages task queues for asynchronous kernel evaluation.

### Step 1: Install Additional Dependencies

```bash
python3 -m ensurepip --upgrade
python3 -m pip install loguru fastapi uvicorn httpx
```

**Verify:**
```bash
python3 -c "import loguru, fastapi, uvicorn, httpx; print('Workflow dependencies OK')"
```

### Step 2: Verify Workflow Server Imports

```bash
python3 -c "import workflowServer; print('workflowServer import OK')"
```

If this fails, install missing dependencies from Step 1.

### Step 3: Verify Workflow Configuration Files

**Check workflow config exists:**
```bash
cat workflow/claude_code.yaml
```

**Expected output:**
```yaml
# Claude Code Kernel Bench workflow configuration
queues:
  - name: kbEval.pending

global:
  prefix_tag: "claude_code"
  start_epoch: 0
  start_block: 0
  end_epoch: 1
  end_block: 1
```

**Check registry entry:**
```bash
grep -A4 "claude_code:" workflow.yaml
```

**Expected output:**
```yaml
  claude_code:
    short_name: "cc"
    config_path: "workflow/claude_code.yaml"
    data_dir: "~/.workflow"
```

**Check local provider config:**
```bash
grep -A4 "^  local:" workflow.yaml
```

**Expected output:**
```yaml
  local:
    host: "localhost"
    port: 8488
    retries: 3
    timeout: 60
```

### Step 4: Start Workflow Server

In a separate terminal:
```bash
cd /path/to/triton-ag
python3 workflowServer.py --host :: --port 8488
```

Wait for the log message:
```
FastAPI server listening on: [:::8488]
```

### Step 5: Verify Workflow Server is Running

In another terminal:
```bash
# Check queue list
curl -s http://localhost:8488/queue/list/claude_code
```

**Expected:** `{"queues":["kbEval.pending"]}`

```bash
# Check queue size (should be 0 initially)
curl -s http://localhost:8488/queue/qsize/claude_code/kbEval.pending
```

**Expected:** `0`

### Step 6: Run E2E Test with Queue

```bash
python3 test_mcp_e2e.py --with-queue
```

**Expected output (ends with):**
```
SUCCESS: Queued to kbEval.pending
Work item submitted_at: 2026-01-24T...
...
=== Test 3: PASS ===
```

### Step 7: Verify Queue Submission

```bash
# Check queue now has 1 item
curl -s http://localhost:8488/queue/qsize/claude_code/kbEval.pending
```

**Expected:** `1`

```bash
# Peek at the queued item
curl -s http://localhost:8488/queue/peek/claude_code/kbEval.pending | python3 -m json.tool | head -10
```

**Expected structure:**
```json
{
    "task_path": "level1/100_HingeLoss.py",
    "kernel_code": "...",
    "session_id": "test_e2e_flow",
    "iteration": 0,
    "submitted_at": "2026-01-24T...",
    "status": "pending"
}
```

### Workflow Server Pass Criteria

- [ ] `python3 -c "import workflowServer"` succeeds
- [ ] `workflow/claude_code.yaml` exists with `kbEval.pending` queue
- [ ] `workflow.yaml` has `claude_code` registry entry
- [ ] Workflow server starts without errors
- [ ] `curl .../queue/list/claude_code` returns queue list
- [ ] `python3 test_mcp_e2e.py --with-queue` passes
- [ ] Queue contains submitted work item

---

## Optional: Remote Workflow Server Setup (GPU Machine)

For running queue-based evaluation with a remote GPU machine. This setup allows Claude Code on a local machine (e.g., laptop) to submit kernels to a workflow server running on a remote GPU server.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        REMOTE GPU SERVER                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  Workflow Server (port 8488)          kbEvalServer (port 5676)         │ │
│  │  ┌─────────────────────────┐          ┌─────────────────────────┐      │ │
│  │  │  Queue: kbEval.pending  │─────────▶│  GPU Kernel Evaluation  │      │ │
│  │  │  prefix_tag: claude_code│          │  Compile + Benchmark    │      │ │
│  │  └─────────────────────────┘          └─────────────────────────┘      │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                              ▲                                               │
└──────────────────────────────│───────────────────────────────────────────────┘
                               │ HTTP (port 8488)
                               │
┌──────────────────────────────│───────────────────────────────────────────────┐
│                              │                                               │
│  ┌───────────────────────────┴─────────────────────────────────────────────┐ │
│  │  Claude Code + MCP Server                                               │ │
│  │  ┌─────────────────────────┐    ┌─────────────────────────────────────┐ │ │
│  │  │ claudeCodeKernelBench   │───▶│ WorkflowClient                      │ │ │
│  │  │ Server.py               │    │ provider: "remote_gpu"              │ │ │
│  │  │ eval_kernel(queue_only) │    │ host: gpu-server.example.com:8488   │ │ │
│  │  └─────────────────────────┘    └─────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                        LOCAL MACHINE (Claude Code)                           │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Part 1: Remote Server Setup (GPU Machine)

SSH into your GPU server and set up the workflow server and kbEvalServer.

> **Note:** For detailed setup instructions including troubleshooting, see `Claude/KBEVAL_SETUP.md`.

#### One-Time Setup

These steps only need to be done once when first setting up the environment.

##### Step 1: Clone Repository

```bash
# On remote GPU server (Meta devserver example)
cd /data/users/$USER
git clone -b claude-code-kernel-bench-plan git@github.com:dtadpole/triton-ag.git
cd triton-ag
```

##### Step 2: Configure Proxy Settings (Meta devservers only)

Add to `~/.bashrc`:

```bash
# Proxy settings for external network access
export https_proxy=http://fwdproxy:8080
export http_proxy=http://fwdproxy:8080
export HTTPS_PROXY=http://fwdproxy:8080
export HTTP_PROXY=http://fwdproxy:8080
export no_proxy=".facebook.com,.tfbnw.net,.fb.com,localhost,127.0.0.1"
export NO_PROXY="$no_proxy"
```

Then reload:
```bash
source ~/.bashrc
```

##### Step 3: Configure pip Proxy (Meta devservers only)

Create `~/.config/pip/pip.conf`:

```ini
[global]
proxy = http://fwdproxy:8080
trusted-host = pypi.org
               pypi.python.org
               files.pythonhosted.org
```

##### Step 4: Create Virtual Environment

```bash
cd /data/users/$USER/triton-ag
python3 -m venv .venv
source .venv/bin/activate
```

##### Step 5: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install kbEval and workflow dependencies
pip install fastapi uvicorn pydantic pyyaml ninja loguru psutil triton numpy httpx
```

**Verify:**
```bash
python3 -c "import workflowServer; print('workflowServer OK')"
python3 -c "import kbEvalServer; print('kbEvalServer OK')"
```

##### Step 6: Create Required Directories

```bash
mkdir -p ~/.workflow ~/.kbeval ~/.inference/claude_code_output
```

#### Recurring Setup (Every Login)

Each time you log into the devserver:

```bash
cd /data/users/$USER/triton-ag
source .venv/bin/activate
```

#### Starting Services

##### Step 7: Start Workflow Server (in tmux)

```bash
# Create or attach to tmux session
tmux new-session -s workflow

# Inside tmux, activate venv and start workflow server (bind to all interfaces for remote access)
cd /data/users/$USER/triton-ag
source .venv/bin/activate
while true; do python3 workflowServer.py --host :: --port 8488; sleep 5; done
```

Press `Ctrl+B D` to detach from tmux.

**Verify server is listening:**
```bash
curl -s http://localhost:8488/queue/list/claude_code
```

**Expected:** `{"queues":["kbEval.pending"]}`

##### Step 8: Start kbEvalServer (in separate tmux)

```bash
# Create tmux session for kbEval
tmux new-session -s kbEval

# Inside tmux, activate venv and start kbEvalServer (use appropriate GPU device)
cd /data/users/$USER/triton-ag
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python3 kbEvalServer.py --local_host --port 5676 --device 0
```

Press `Ctrl+B D` to detach from tmux.

**Verify kbEval is running:**
```bash
curl -s http://localhost:5676/health
```

##### Step 9: Verify Remote Access

From another machine (or the remote server itself), verify the workflow server is accessible:

```bash
# Replace gpu-server with actual hostname or IP
curl -s http://gpu-server:8488/queue/list/claude_code
```

**Expected:** `{"queues":["kbEval.pending"]}`

If connection fails, check firewall settings:
```bash
# On the remote server (if using firewalld)
sudo firewall-cmd --add-port=8488/tcp --permanent
sudo firewall-cmd --reload

# Or using ufw
sudo ufw allow 8488/tcp
```

### Part 2: Local Environment Setup (Claude Code Machine)

Configure the local machine to connect to the remote workflow server.

There are two connection methods:
- **Option A: Direct Connection** - If your local machine can directly reach the remote server (same network, VPN, etc.)
- **Option B: SSH Tunnel** - If the remote server is on an internal network not directly accessible

---

#### Option A: Direct Connection

Use this if your local machine can directly reach the remote server's IP/hostname.

##### Step A1: Add Remote Provider to workflow.yaml

Edit `workflow.yaml` to add a new provider for the remote GPU server:

```yaml
providers:
  local:
    host: "localhost"
    port: 8488
    retries: 3
    timeout: 60

  # Add this new provider
  remote_gpu:
    host: "gpu-server.example.com"  # Replace with actual hostname/IP
    port: 8488
    retries: 5
    timeout: 120
```

**Verify the provider is configured:**
```bash
grep -A4 "remote_gpu:" workflow.yaml
```

##### Step A2: Update MCP Server Configuration

Edit `claudeCodeKernelBench.yaml` to use the remote provider:

```yaml
workflow:
  prefix_tag: "claude_code"
  eval_queue: "kbEval.pending"
  config_file: "workflow.yaml"
  provider_name: "remote_gpu"  # Changed from "local"
```

#### Step A3: Verify WorkflowClient Can Connect

```bash
python3 -c "
from workflowClient import WorkflowClient
client = WorkflowClient(prefix_tag='claude_code', provider_name='remote_gpu')
import asyncio
result = asyncio.run(client.queue_list())
print(f'Connected! Queues: {result}')
"
```

**Expected:** `Connected! Queues: ['kbEval.pending']`

#### Step A4: Restart Claude Code

Claude Code must be restarted to pick up the configuration changes:

```bash
cd /path/to/triton-ag
claude
```

---

#### Option B: SSH Tunnel

Use this if the remote server is on an internal network that your local machine cannot directly reach. The SSH tunnel creates a secure connection through an SSH bastion/jump host.

##### Step B1: Create SSH Tunnel

On your local machine, open a terminal and create the tunnel:

```bash
# Replace devvm8491.cco0.facebook.com with your remote server hostname
ssh -L 8488:localhost:8488 -N devvm8491.cco0.facebook.com
```

This command:
- `-L 8488:localhost:8488` - Forwards local port 8488 to the remote server's localhost:8488
- `-N` - Don't run a command, just forward ports
- The tunnel stays open as long as this terminal is running

**Keep this terminal open** while using Claude Code.

##### Step B2: Verify Tunnel Connection

In another terminal, test the connection:

```bash
curl -s http://localhost:8488/queue/list/claude_code
```

**Expected:** `{"queues":["kbEval.pending"]}`

If this works, your tunnel is active and forwarding correctly.

##### Step B3: Configure Local Provider

The `workflow.yaml` already has a `local` provider configured for `localhost:8488`. Verify it:

```bash
grep -A4 "^  local:" workflow.yaml
```

**Expected:**
```yaml
  local:
    host: "localhost"
    port: 8488
    retries: 3
    timeout: 60
```

Ensure `claudeCodeKernelBench.yaml` uses the local provider:

```bash
grep "provider_name" claudeCodeKernelBench.yaml
```

**Expected:** `provider_name: "local"`

##### Step B4: Restart Claude Code

Exit any running Claude Code session and restart:

```bash
cd /path/to/triton-ag
claude
```

Claude Code will now use `localhost:8488` which goes through your SSH tunnel to the remote server.

##### SSH Tunnel Tips

- **Keep the tunnel terminal open** - The tunnel closes when you close the terminal or press Ctrl+C
- **Background the tunnel** - Use `ssh -L 8488:localhost:8488 -N -f devvm8491...` (adds `-f` to run in background)
- **Check if tunnel is running** - `lsof -i :8488` should show ssh listening
- **Reconnect after network changes** - If you change networks (WiFi, VPN), you may need to restart the tunnel

### Part 3: Verification Test (Queue Submission Only)

#### Test Remote Queue Submission

Run the E2E test with queue submission to verify the full flow:

```bash
python3 test_mcp_e2e.py --with-queue
```

**Expected output:**
```
=== Test 3: Kernel Evaluation E2E ===
...
SUCCESS: Queued to kbEval.pending
Work item submitted_at: 2026-01-24T...
=== Test 3: PASS ===
```

#### Verify Item Reached Remote Server

On the remote GPU server, check the queue:

```bash
curl -s http://localhost:8488/queue/qsize/claude_code/kbEval.pending
```

**Expected:** `1` (or higher if previous items exist)

```bash
curl -s http://localhost:8488/queue/peek/claude_code/kbEval.pending | python3 -m json.tool | head -10
```

**Expected:** Work item with task_path, kernel_code, session_id, etc.

---

### Part 4: Full End-to-End Test (Queue + GPU Evaluation)

This test validates the complete pipeline: Claude Code submits kernels → Workflow Server queues them → kbEvalServer evaluates on GPU → Results returned.

#### Prerequisites

- Part 1 completed (workflowServer running on remote)
- Part 2 completed (local Claude Code can submit to queue)
- GPU available on remote server

#### Step 1: Stop Workflow Server (if needed to restart fresh)

On the remote server, if you need to restart with a clean queue:

```bash
# Find and stop existing workflowServer
tmux kill-session -t workflow 2>/dev/null || true

# Clear the queue (optional - for clean test)
rm -rf ~/.workflow/claude_code/kbEval.pending 2>/dev/null || true
```

#### Step 2: Start Workflow Server

```bash
# Create tmux session for workflow server
tmux new-session -d -s workflow

# Start workflow server (with venv activation)
tmux send-keys -t workflow 'cd /data/users/$USER/triton-ag && source .venv/bin/activate && while true; do python3 workflowServer.py --host :: --port 8488; sleep 5; done' Enter
```

**Verify:**
```bash
curl -s http://localhost:8488/queue/list/claude_code
```

**Expected:** `{"queues":["kbEval.pending"]}`

#### Step 3: Start kbEvalServer

The kbEvalServer handles kernel compilation and benchmarking.

```bash
# Create tmux session for kbEval
tmux new-session -d -s kbEval

# Start kbEvalServer (with venv activation)
tmux send-keys -t kbEval 'cd /data/users/$USER/triton-ag && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=0 python3 kbEvalServer.py --local_host --port 5676 --device 0' Enter
```

**Verify kbEvalServer is running:**
```bash
curl -s http://localhost:5676/health
```

**Expected:** Health check response (200 OK)

#### Step 4: Start Queue Consumer

The kbEvalServer doesn't consume from the workflow queue directly. Create a queue consumer script that bridges the queue to kbEvalServer.

Create `kbEvalQueueConsumer.py`:

```python
#!/usr/bin/env python3
"""
Queue consumer that pulls from workflow queue and sends to kbEvalServer.
"""
import asyncio
import httpx
from workflowClient import WorkflowClient
from loguru import logger

KBEVAL_URL = "http://localhost:5676"
PREFIX_TAG = "claude_code"
QUEUE_NAME = "kbEval.pending"

async def process_item(item: dict) -> dict:
    """Send item to kbEvalServer for evaluation."""
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(
            f"{KBEVAL_URL}/eval",
            json={
                "task_path": item["task_path"],
                "kernel_code": item["kernel_code"],
                "session_id": item.get("session_id", "unknown"),
                "iteration": item.get("iteration", 0),
            }
        )
        return response.json()

async def main():
    workflow_client = WorkflowClient(prefix_tag=PREFIX_TAG, provider_name="local")

    logger.info(f"Starting queue consumer for {PREFIX_TAG}/{QUEUE_NAME}")
    logger.info(f"Sending evaluations to {KBEVAL_URL}")

    while True:
        try:
            # Try to dequeue an item (blocks until available or timeout)
            item = await workflow_client.dequeue(QUEUE_NAME)

            if item:
                logger.info(f"Processing: {item.get('task_path')} iter={item.get('iteration')}")
                result = await process_item(item)
                logger.info(f"Result: compiled={result.get('compiled')}, "
                           f"correctness={result.get('correctness')}, "
                           f"speedup={result.get('speedup', 0):.2f}x")
            else:
                # No item available, wait before retrying
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error processing item: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
```

Start the queue consumer:

```bash
# Create tmux session for queue consumer
tmux new-session -d -s queueConsumer

# Start queue consumer (with venv activation)
tmux send-keys -t queueConsumer 'cd /data/users/$USER/triton-ag && source .venv/bin/activate && python3 kbEvalQueueConsumer.py' Enter
```

#### Step 5: Verify All Services Running

```bash
# Check tmux sessions
tmux list-sessions
```

**Expected:**
```
kbEval: 1 windows ...
queueConsumer: 1 windows ...
workflow: 1 windows ...
```

```bash
# Check ports are listening
netstat -tlnp 2>/dev/null | grep -E '8488|5676' || lsof -i :8488 -i :5676
```

#### Step 6: Submit Test Kernel from Local Claude Code

On your local machine with Claude Code, submit a kernel for evaluation:

In Claude Code:
```
Generate a simple ReLU Triton kernel for kernel_bench/level1/19_ReLU.py and submit it for evaluation using eval_kernel with queue_only=True. Use session_id="e2e_test".
```

Or run programmatically:
```bash
python3 test_mcp_e2e.py --with-queue
```

#### Step 7: Monitor Queue Processing

On the remote server, watch the queue being consumed:

```bash
# Watch queue size (should go from 1 to 0 as kbEvalServer processes)
watch -n 1 'curl -s http://localhost:8488/queue/qsize/claude_code/kbEval.pending'
```

Or check kbEvalServer logs:
```bash
tmux attach -t kbEval
# (Press Ctrl+B D to detach)
```

**Expected log output:**
```
Processing work item from queue: kbEval.pending
Task: level1/19_ReLU.py
Compiling kernel...
Running benchmark...
Result: compiled=True, correctness=True, speedup=1.23
```

#### Step 8: Verify Evaluation Results

The kbEvalServer will store results. Check the output directory:

```bash
ls -la ~/.inference/claude_code_output/e2e_test/
```

**Expected:** Directory with evaluation results

#### Step 9: Full E2E Test Script

For automated testing, create a test that:
1. Submits a kernel
2. Waits for processing
3. Verifies results

```bash
# Submit kernel and wait for evaluation
python3 -c "
import asyncio
import time
from claudeCodeKernelBenchServer import eval_kernel, get_session_summary

async def test_e2e():
    # Submit kernel
    kernel_code = '''
import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, tl.maximum(x, 0.0), mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        out = torch.empty_like(x.view(-1))
        relu_kernel[(triton.cdiv(x.numel(), 1024),)](x.view(-1), out, x.numel(), BLOCK_SIZE=1024)
        return out.view(x.shape)
'''
    result = await eval_kernel('level1/19_ReLU.py', kernel_code, 'e2e_full_test', 0, queue_only=True)
    print(f'Submitted: {result}')

    # Wait for processing (adjust timeout as needed)
    print('Waiting for kbEvalServer to process...')
    for i in range(30):
        time.sleep(2)
        summary = await get_session_summary('e2e_full_test')
        if summary.get('19_ReLU', {}).get('iterations', []):
            latest = summary['19_ReLU']['iterations'][-1]
            if latest.get('compiled'):
                print(f'SUCCESS: compiled={latest[\"compiled\"]}, correctness={latest[\"correctness\"]}, speedup={latest[\"speedup\"]}')
                return
        print(f'  Waiting... ({i+1}/30)')
    print('TIMEOUT: kbEvalServer did not process in time')

asyncio.run(test_e2e())
"
```

### Part 4 Pass Criteria

| Check | How to Verify | Expected |
|-------|---------------|----------|
| Workflow server running | `curl http://localhost:8488/health` | 200 OK |
| kbEvalServer running | `curl http://localhost:5676/health` | 200 OK |
| Queue accessible | `curl .../queue/list/claude_code` | Shows `kbEval.pending` |
| Submission works | Submit kernel from Claude Code | `status: queued` |
| Queue consumed | Watch queue size | Goes from 1 → 0 |
| Evaluation succeeds | Check kbEvalServer logs | `compiled=True` |
| Results saved | Check output directory | Files present |

### Troubleshooting Part 4

#### Queue Not Being Consumed

1. **Verify all three services are running:**
   ```bash
   tmux list-sessions
   # Should show: workflow, kbEval, queueConsumer
   ```

2. **Check queue consumer is connected:**
   ```bash
   tmux attach -t queueConsumer
   # Should show: "Starting queue consumer for claude_code/kbEval.pending"
   ```

3. **Verify kbEvalServer is accessible from queue consumer:**
   ```bash
   curl -s http://localhost:5676/health
   ```

4. **Check queue consumer logs for errors:**
   ```bash
   tmux attach -t queueConsumer
   # Look for error messages
   ```

#### Kernel Compilation Fails

1. **Check CUDA is available:**
   ```bash
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Check Triton is installed:**
   ```bash
   python3 -c "import triton; print(triton.__version__)"
   ```

3. **Check GPU memory:**
   ```bash
   nvidia-smi
   ```

#### Results Not Appearing

1. **Check output directory permissions:**
   ```bash
   ls -la ~/.inference/claude_code_output/
   ```

2. **Check kbEvalServer output path configuration**

### Remote Server Pass Criteria

| Check | Command | Expected |
|-------|---------|----------|
| Workflow server running | `curl http://gpu-server:8488/health` | `{"status":"ok"}` or 200 |
| Queue exists | `curl http://gpu-server:8488/queue/list/claude_code` | `{"queues":["kbEval.pending"]}` |
| Local client connects | `python3 -c "..."` (Step 3 above) | Shows queue list |
| E2E test passes | `python3 test_mcp_e2e.py --with-queue` | `Test 3: PASS` |
| Item in remote queue | `curl .../qsize/...` on remote | `>= 1` |

### Troubleshooting Remote Setup

#### Connection Refused to Remote Server

1. **Check server is running:**
   ```bash
   # On remote server
   curl http://localhost:8488/health
   ```

2. **Check firewall allows port 8488:**
   ```bash
   # On remote server
   sudo netstat -tlnp | grep 8488
   ```

3. **Check server is bound to all interfaces (not just localhost):**
   - Server must be started with `--host ::` or `--host 0.0.0.0`

4. **Test connectivity:**
   ```bash
   # From local machine
   nc -zv gpu-server 8488
   ```

#### Provider Configuration Errors

1. **Verify provider name matches:**
   - `claudeCodeKernelBench.yaml` → `provider_name: "remote_gpu"`
   - `workflow.yaml` → `providers: remote_gpu: ...`

2. **Check for typos in hostname:**
   ```bash
   ping gpu-server.example.com
   ```

3. **Verify YAML syntax:**
   ```bash
   python3 -c "import yaml; yaml.safe_load(open('workflow.yaml'))" && echo "YAML OK"
   ```

#### Queue Submission Succeeds But Item Not on Remote

1. **Check you're looking at the right server:**
   - Local workflow server may also be running on port 8488
   - Stop local server or use different port

2. **Verify the client is using remote provider:**
   ```bash
   python3 -c "
   from workflowClient import WorkflowClient
   client = WorkflowClient(prefix_tag='claude_code', provider_name='remote_gpu')
   print(f'Host: {client.provider}')
   "
   ```

---

## Optional: kbEvalServer Setup (GPU Required)

Required for actual kernel compilation and benchmarking (Test 4+).

### Prerequisites

- NVIDIA GPU with CUDA
- PyTorch with CUDA support
- Triton installed

### Start kbEvalServer

```bash
# Use appropriate GPU device
CUDA_VISIBLE_DEVICES=0 python3 kbEvalServer.py --local_host --port 5676 --device 0
```

Wait for the server to start and show ready message.

### Verify kbEvalServer

```bash
curl -s http://localhost:5676/health
```

---

## Verification Tests

Run these to verify the setup is working:

### Test 1: MCP Functions (No servers required)

```bash
python3 test_mcp_functions.py
```

**Expected:** Ends with `Test 2 Part A: PASS`

### Test 2: End-to-End Without Queue

```bash
python3 test_mcp_e2e.py
```

**Expected:** Ends with `Test 3: PASS`

### Test 3: End-to-End With Queue (Requires workflow server)

```bash
python3 test_mcp_e2e.py --with-queue
```

**Expected:** Ends with `Test 3: PASS` and shows queue submission success.

### Test 4: End-to-End With Remote Workflow Server

After completing the "Remote Workflow Server Setup" section:

```bash
# Ensure claudeCodeKernelBench.yaml has provider_name: "remote_gpu"
python3 test_mcp_e2e.py --with-queue
```

**Expected:**
- Test passes locally with `Test 3: PASS`
- Queue item appears on remote server (verify with `curl http://gpu-server:8488/queue/qsize/claude_code/kbEval.pending`)

---

## Usage Examples

### List Available Tasks

In Claude Code:
```
Use the list_kernel_bench_tasks tool to list all level1 tasks.
```

### Optimize a Single Kernel

```
Optimize the kernel in kernel_bench/level1/1_Square_matrix_multiplication_.py

Use session_id="my_test_session" and generate a Triton kernel.
```

### Run Batch Benchmark

```
Run a benchmark session on 5 tasks from level1.
Use session_id="batch_test".
For each task, generate a Triton kernel and save the result.
```

---

## Troubleshooting

### "No module named pip" Error

Some Python installations don't include pip by default. Bootstrap it first:

```bash
python3 -m ensurepip --upgrade
```

Then retry the pip install command. If ensurepip fails, you may need to install pip via your system package manager (e.g., `brew install python` on macOS).

### MCP Tools Not Available

1. Verify `.mcp.json` exists in project root
2. Verify `mcp` package is installed: `python3 -m pip install mcp`
3. Restart Claude Code from the project directory (start with `claude` from `/path/to/triton-ag`)
4. Check MCP server can start: `python3 -c "from claudeCodeKernelBenchServer import MCP_AVAILABLE; print(MCP_AVAILABLE)"`

**Note:** After modifying `claudeCodeKernelBenchServer.py`, you must restart Claude Code to reload the MCP server.

### Queue Submission Returns "WorkflowClient not available"

This error occurs if the MCP server can't import the `workflowClient` module. Verify:

1. All workflow dependencies are installed:
   ```bash
   python3 -m pip install loguru fastapi uvicorn httpx pydantic
   ```

2. WorkflowClient can be imported:
   ```bash
   python3 -c "from workflowClient import WorkflowClient; print('OK')"
   ```

3. Claude Code was started from the project directory (not from a parent or different directory)

### Workflow Server Connection Failed

1. Verify server is running: `curl http://localhost:8488/health`
2. Check port is not blocked
3. Verify `workflow/claude_code.yaml` exists
4. Check registry entry in `workflow.yaml`

### Kernel Bench Tasks Not Found

1. Verify path in `claudeCodeKernelBench.yaml`:
   ```yaml
   kernel_bench:
     base_dir: "./kernel_bench"  # or absolute path
   ```
2. Check tasks exist: `ls kernel_bench/level1/*.py | wc -l`

### Import Errors

Bootstrap pip if needed, then install missing dependencies:
```bash
python3 -m ensurepip --upgrade
python3 -m pip install pyyaml mcp loguru fastapi uvicorn httpx
```

---

## File Reference

| File | Purpose |
|------|---------|
| `claudeCodeKernelBenchServer.py` | MCP server with kernel bench tools |
| `claudeCodeKernelBench.yaml` | MCP server configuration (provider settings) |
| `.mcp.json` | MCP registration for Claude Code |
| `workflow.yaml` | Workflow registry and provider configurations |
| `workflow/claude_code.yaml` | Claude Code workflow queue configuration |
| `benchmarkCompare.py` | Compare Claude vs RL results |
| `test_mcp_functions.py` | Verify MCP tool functions |
| `test_mcp_e2e.py` | End-to-end verification |

---

## Next Steps

After setup is complete:

1. **Basic Usage:** Generate kernels for individual tasks
2. **Batch Benchmarks:** Run on multiple tasks with session tracking
3. **Remote GPU:** Set up remote workflow server for GPU-based evaluation
4. **Comparison:** Compare results with RL-trained models using `benchmarkCompare.py`
5. **Full E2E:** With kbEvalServer, get actual compilation and speedup metrics
