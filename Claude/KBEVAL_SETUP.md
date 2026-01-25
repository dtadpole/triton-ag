# kbEval & Workflow Server Setup Guide

This guide covers setting up and running the kbEval (Kernel Benchmark Evaluation) server and workflowServer on a Meta devserver.

## Prerequisites

- Meta devserver with GPU(s) (H100, A100, etc.)
- Python 3.11+ available
- CUDA toolkit installed

---

## One-Time Setup

These steps only need to be done once when first setting up the environment.

### 1. Clone the Repository

```bash
cd /data/users/$USER
git clone -b claude-code-kernel-bench-plan git@github.com:dtadpole/triton-ag.git
cd triton-ag
```

### 2. Configure Proxy Settings

Add to `~/.bashrc`:

```bash
# Proxy settings for external network access
export https_proxy=http://fwdproxy:8080
export http_proxy=http://fwdproxy:8080
export HTTPS_PROXY=http://fwdproxy:8080
export HTTP_PROXY=http://fwdproxy:8080
export no_proxy=".facebook.com,.tfbnw.net,.fb.com,localhost,127.0.0.1"
export NO_PROXY="$no_proxy"

# Convenience alias
alias with-proxy='HTTPS_PROXY=http://fwdproxy:8080 HTTP_PROXY=http://fwdproxy:8080 https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080'
```

Then reload:
```bash
source ~/.bashrc
```

### 3. Configure pip Proxy

Create `~/.config/pip/pip.conf`:

```ini
[global]
proxy = http://fwdproxy:8080
trusted-host = pypi.org
               pypi.python.org
               files.pythonhosted.org
```

### 4. Create Virtual Environment

```bash
cd /data/users/$USER/triton-ag
python3 -m venv .venv
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install kbEval dependencies
pip install fastapi uvicorn pydantic pyyaml ninja loguru psutil triton numpy

# Optional: Install full dependencies for development
pip install -r requirements_devserver.txt
```

### 6. Create API Key

The kbEval API key is used for client-server authentication.

**For local testing only:** This is optional - kbEvalServer will auto-generate one if missing.

**For remote access (required):** You must create a key and share it with client machines.

```bash
mkdir -p ~/.keys
# Generate a random UUID key
python3 -c "import uuid; print(str(uuid.uuid4()))" > ~/.keys/kbeval.api.key
chmod 600 ~/.keys/kbeval.api.key

# View the generated key
cat ~/.keys/kbeval.api.key
```

**Setting up remote clients:** Copy the same API key to each machine that needs to access kbEval:

```bash
# On each remote/client machine
mkdir -p ~/.keys
echo "YOUR-API-KEY-HERE" > ~/.keys/kbeval.api.key
chmod 600 ~/.keys/kbeval.api.key
```

Replace `YOUR-API-KEY-HERE` with the actual key from the server machine.

### 7. Create Required Directories

```bash
mkdir -p ~/.kbeval ~/.workflow ~/.inference ~/.trainer
```

---

## Recurring Setup (Every Login)

These steps are needed each time you log into the devserver and want to start the services.

### 1. Navigate to Project

```bash
cd /data/users/$USER/triton-ag
```

### 2. Activate Virtual Environment

```bash
source .venv/bin/activate
```

---

## Starting kbEval Server

### Option A: Single GPU (recommended for testing)
```bash
# Use GPU 7 (or any available GPU)
python kbEvalServer.py --local_host --port 5676 --device 7
```

### Option B: Using Makefile (uses config from kbEval.yaml)
```bash
make kbeval_local
```

### Option C: Production mode (auto-restart loop)
```bash
while true; do python kbEvalServer.py --local_host --port 5676 --device 7; sleep 5; done
```

### Option D: Run in tmux (recommended for long-running sessions)
```bash
# Create new tmux session
tmux new-session -s kbeval

# Inside tmux, run kbEval
cd /data/users/$USER/triton-ag
source .venv/bin/activate
python kbEvalServer.py --local_host --port 5676 --device 7

# Detach with: Ctrl+b, then d
# Reattach later with: tmux attach -t kbeval
```

---

## Starting Workflow Server

The workflowServer manages training/inference workflows and task queues.

### Option A: Local only (recommended for testing)
```bash
python workflowServer.py --host localhost --port 8488
```

### Option B: Accept connections from other machines
```bash
# Use :: to bind to all IPv6 interfaces (also accepts IPv4)
python workflowServer.py --host :: --port 8488
```

### Option C: Production mode (auto-restart loop)
```bash
while true; do python workflowServer.py --host :: --port 8488; sleep 5; done
```

### Option D: Run in tmux (recommended for long-running sessions)
```bash
# Create new tmux session
tmux new-session -s workflow

# Inside tmux, run workflowServer
cd /data/users/$USER/triton-ag
source .venv/bin/activate
while true; do python workflowServer.py --host :: --port 8488; sleep 5; done

# Detach with: Ctrl+b, then d
# Reattach later with: tmux attach -t workflow
```

---

## Accessing Services from Another Machine

### Find Your Devserver Hostname

```bash
hostname -f
# Example output: devvm8491.cco0.facebook.com
```

### Remote Access to kbEval Server

If you started kbEval with `--local_host`, it only accepts local connections. To allow remote access:

```bash
# Start kbEval binding to all interfaces (remove --local_host)
python kbEvalServer.py --host :: --port 5676 --device 7
```

From another machine:
```bash
# Health check
curl http://devvm8491.cco0.facebook.com:5676/health

# Or use the configured hostname from kbEval.yaml
```

### Remote Access to Workflow Server

From another machine:
```bash
# Health check
curl http://devvm8491.cco0.facebook.com:8488/health

# List workflows
curl http://devvm8491.cco0.facebook.com:8488/workflow/list
```

### Configuring Clients to Connect

**For kbEval clients**, edit `kbEval.yaml` on the client machine:
```yaml
providers:
  my_remote:
    host: "devvm8491.cco0.facebook.com"
    port: 5676
    api_key_path: "~/.keys/kbeval.api.key"
```

Make sure the client machine has the same API key file (see Step 6 in One-Time Setup).

**For workflow clients**, edit `workflow.yaml`:
```yaml
providers:
  my_remote:
    host: "devvm8491.cco0.facebook.com"
    port: 8488
    retries: 5
    timeout: 300
```

### SSH Tunneling (Alternative)

If direct connection isn't available, use SSH tunneling:

```bash
# From your local machine, create a tunnel to the devserver
ssh -L 5676:localhost:5676 -L 8488:localhost:8488 devvm8491.cco0.facebook.com

# Then access via localhost
curl http://localhost:5676/health
curl http://localhost:8488/health
```

---

## Quick Start Scripts

### Create `~/start_kbeval.sh`:
```bash
#!/bin/bash
cd /data/users/$USER/triton-ag
source .venv/bin/activate
python kbEvalServer.py --local_host --port 5676 --device 7
```

### Create `~/start_workflow.sh`:
```bash
#!/bin/bash
cd /data/users/$USER/triton-ag
source .venv/bin/activate
python workflowServer.py --host :: --port 8488
```

### Create `~/start_all.sh` (starts both in tmux):
```bash
#!/bin/bash
PROJECT_DIR="/data/users/$USER/triton-ag"

# Start kbEval in tmux
tmux new-session -d -s kbeval "cd $PROJECT_DIR && source .venv/bin/activate && python kbEvalServer.py --host :: --port 5676 --device 7"

# Start workflow in tmux
tmux new-session -d -s workflow "cd $PROJECT_DIR && source .venv/bin/activate && while true; do python workflowServer.py --host :: --port 8488; sleep 5; done"

echo "Started kbeval and workflow servers in tmux sessions"
echo "Use 'tmux attach -t kbeval' or 'tmux attach -t workflow' to view"
```

Make scripts executable:
```bash
chmod +x ~/start_kbeval.sh ~/start_workflow.sh ~/start_all.sh
```

---

## Command Reference

### kbEvalServer Options

| Command | Description |
|---------|-------------|
| `--local_host` | Bind to localhost only (no remote access) |
| `--host HOST` | Bind address (use `::` for all interfaces) |
| `--port PORT` | Server port (default: 5676) |
| `--device N` | GPU device index to use |

### workflowServer Options

| Command | Description |
|---------|-------------|
| `--host HOST` | Bind address (`localhost` for local only, `::` for all) |
| `--port PORT` | Server port (default: 8488) |

---

## Verify Services are Running

```bash
# Check kbEval
curl http://localhost:5676/health

# Check workflow
curl http://localhost:8488/health

# Check GPU usage
nvidia-smi

# List tmux sessions
tmux list-sessions
```

---

## Troubleshooting

### "No module named X" error
```bash
source .venv/bin/activate
pip install <missing-module>
```

### CUDA out of memory
Use a different GPU device:
```bash
python kbEvalServer.py --host :: --port 5676 --device 0  # Try GPU 0 instead
```

### Port already in use
```bash
# Find process using the port
lsof -i :5676
lsof -i :8488

# Kill it or use a different port
python kbEvalServer.py --host :: --port 5677 --device 7
python workflowServer.py --host :: --port 8489
```

### Cannot connect from remote machine
1. Ensure you're NOT using `--local_host` for kbEval
2. Ensure you're using `--host ::` or `--host 0.0.0.0` for workflowServer
3. Check firewall rules (typically not an issue on devservers)
4. Verify the hostname: `hostname -f`

---

## Directory Reference

| Path | Purpose |
|------|---------|
| `/data/users/$USER/triton-ag` | Project root |
| `/data/users/$USER/triton-ag/.venv` | Python virtual environment |
| `~/.keys/kbeval.api.key` | API authentication key |
| `~/.kbeval/` | Evaluation cache directory |
| `~/.workflow/` | Workflow registry data, queue persistence |
| `~/.inference/` | Inference outputs |
| `~/.trainer/` | Model checkpoints |
| `~/.config/pip/pip.conf` | pip proxy configuration |

---

## Default Ports Summary

| Service | Default Port | Purpose |
|---------|--------------|---------|
| kbEval | 5676 | Kernel compilation and benchmarking |
| workflowServer | 8488 | Workflow orchestration and task queues |

