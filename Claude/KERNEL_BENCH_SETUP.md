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

#### Step 1: Clone or Sync Repository

```bash
# On remote GPU server
cd /path/to/projects
git clone <triton-ag-repo-url> triton-ag
cd triton-ag
```

#### Step 2: Install Dependencies

```bash
python3 -m ensurepip --upgrade
python3 -m pip install loguru fastapi uvicorn httpx pyyaml pydantic torch triton
```

**Verify:**
```bash
python3 -c "import workflowServer; print('workflowServer OK')"
python3 -c "import kbEvalServer; print('kbEvalServer OK')"
```

#### Step 3: Create Required Directories

```bash
mkdir -p ~/.workflow ~/.kbeval ~/.inference/claude_code_output
```

#### Step 4: Start Workflow Server (in tmux)

```bash
# Create or attach to tmux session
tmux new-session -s workflow

# Start workflow server (bind to all interfaces for remote access)
cd /path/to/triton-ag
while true; do python3 workflowServer.py --host :: --port 8488; sleep 5; done
```

Press `Ctrl+B D` to detach from tmux.

**Verify server is listening:**
```bash
curl -s http://localhost:8488/queue/list/claude_code
```

**Expected:** `{"queues":["kbEval.pending"]}`

#### Step 5: Start kbEvalServer (in separate tmux)

```bash
# Create tmux session for kbEval
tmux new-session -s kbEval

# Start kbEvalServer (use appropriate GPU device)
cd /path/to/triton-ag
CUDA_VISIBLE_DEVICES=0 python3 kbEvalServer.py --local_host --port 5676 --device 0
```

Press `Ctrl+B D` to detach from tmux.

**Verify kbEval is running:**
```bash
curl -s http://localhost:5676/health
```

#### Step 6: Verify Remote Access

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

#### Step 1: Add Remote Provider to workflow.yaml

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

#### Step 2: Update MCP Server Configuration

Edit `claudeCodeKernelBench.yaml` to use the remote provider:

```yaml
workflow:
  prefix_tag: "claude_code"
  eval_queue: "kbEval.pending"
  config_file: "workflow.yaml"
  provider_name: "remote_gpu"  # Changed from "local"
```

#### Step 3: Verify WorkflowClient Can Connect

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

#### Step 4: Restart Claude Code

Claude Code must be restarted to pick up the configuration changes:

```bash
cd /path/to/triton-ag
claude
```

### Part 3: Verification Test

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
