#!/usr/bin/env python3
"""
Kernel Bench Server Configuration Tool.

Manage kbEval server providers, API keys, and SSH tunnels.

Usage:
    python kb_server.py list                              # List all providers
    python kb_server.py add NAME URL [--devices=N]        # Add/update provider
    python kb_server.py test [PROVIDER]                   # Test connection
    python kb_server.py key [--set=KEY]                   # View or set API key
    python kb_server.py tunnel USER@HOST:PORT [--local=PORT]  # Setup SSH tunnel
"""

import sys
import os
import json
import subprocess
from pathlib import Path

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


CONFIG_FILE = "kbEval.yaml"
API_KEY_PATH = Path.home() / ".keys" / "kbeval.api.key"


def load_config():
    """Load kbEval.yaml configuration."""
    if not YAML_AVAILABLE:
        print("Error: PyYAML not installed. Run: pip install pyyaml")
        sys.exit(1)

    config_path = Path(CONFIG_FILE)
    if not config_path.exists():
        print(f"Error: {CONFIG_FILE} not found")
        sys.exit(1)

    with open(config_path) as f:
        return yaml.safe_load(f)


def save_config(config):
    """Save configuration to kbEval.yaml."""
    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def cmd_list():
    """List all configured providers."""
    config = load_config()
    providers = config.get("providers", {})

    print("\n=== kbEval Server Configuration ===\n")
    print(f"{'Provider':<15} {'URL':<45} {'Timeout':<10}")
    print("-" * 70)

    for name, prov in providers.items():
        url = prov.get("base_url", "N/A")
        timeout = prov.get("timeout", 300)
        print(f"{name:<15} {url:<45} {timeout}s")

    print()

    # Check API key
    if API_KEY_PATH.exists():
        key = API_KEY_PATH.read_text().strip()
        masked = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
        print(f"API Key: {masked} ({API_KEY_PATH})")
    else:
        print(f"API Key: NOT SET (expected at {API_KEY_PATH})")
        print("  Use: python kb_server.py key --set=YOUR_KEY")
    print()


def cmd_add(name, url, devices=None, timeout=300):
    """Add or update a provider."""
    config = load_config()

    if "providers" not in config:
        config["providers"] = {}

    config["providers"][name] = {
        "base_url": url,
        "api_key_path": "~/.keys/kbeval.api.key",
        "retry_count": 4,
        "initial_retry_interval": 3,
        "timeout": timeout
    }

    save_config(config)

    print(f"\nUpdated provider '{name}':")
    print(f"  URL: {url}")
    print(f"  Timeout: {timeout}s")
    if devices:
        print(f"  Devices: {devices} (info only, server reports actual count)")
    print(f"\nTo use: /kernel-bench level1 --session=test --provider={name}")
    print()

    # Test connection
    cmd_test(name)


def cmd_test(provider=None):
    """Test connection to a provider."""
    config = load_config()
    providers = config.get("providers", {})

    if provider is None:
        provider = "local"

    if provider not in providers:
        print(f"Error: Provider '{provider}' not found")
        return False

    url = providers[provider].get("base_url", "")
    print(f"\nTesting {provider} ({url})...")

    # Try /stats endpoint first, then /health
    for endpoint in ["/stats", "/health", "/"]:
        try:
            result = subprocess.run(
                ["curl", "-s", "--max-time", "5", f"{url}{endpoint}"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0 and result.stdout:
                try:
                    data = json.loads(result.stdout)
                    if "num_devices" in data:
                        print(f"  ✓ Connected - {data.get('num_devices', '?')} GPU(s) available")
                    elif "status" in data:
                        print(f"  ✓ Connected - Status: {data.get('status')}")
                    else:
                        print(f"  ✓ Connected - Response: {result.stdout[:100]}")
                    return True
                except json.JSONDecodeError:
                    if "ok" in result.stdout.lower() or "healthy" in result.stdout.lower():
                        print(f"  ✓ Connected")
                        return True
        except Exception as e:
            pass

    print(f"  ✗ Connection failed")
    print(f"    Check if server is running at {url}")
    return False


def cmd_key(set_key=None):
    """View or set API key."""
    API_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)

    if set_key:
        API_KEY_PATH.write_text(set_key.strip())
        os.chmod(API_KEY_PATH, 0o600)
        print(f"API key saved to {API_KEY_PATH}")
    else:
        if API_KEY_PATH.exists():
            key = API_KEY_PATH.read_text().strip()
            masked = key[:8] + "..." + key[-4:] if len(key) > 12 else key
            print(f"API Key: {masked}")
            print(f"Path: {API_KEY_PATH}")
        else:
            print(f"API key not set.")
            print(f"Expected path: {API_KEY_PATH}")
            print(f"\nTo set: python kb_server.py key --set=YOUR_KEY")


def cmd_tunnel(target, local_port=None):
    """Setup SSH tunnel to remote server.

    Args:
        target: user@host:port format
        local_port: local port to forward (default: same as remote)
    """
    # Parse target
    if "@" not in target:
        print("Error: Use format user@host:port")
        sys.exit(1)

    user_host, remote_port = target.rsplit(":", 1) if ":" in target.split("@")[1] else (target, "8082")

    if local_port is None:
        local_port = remote_port

    print(f"\nSetting up SSH tunnel:")
    print(f"  Local:  localhost:{local_port}")
    print(f"  Remote: {user_host}:{remote_port}")
    print()

    # Build SSH command
    ssh_cmd = [
        "ssh", "-N", "-L", f"{local_port}:localhost:{remote_port}",
        user_host
    ]

    print(f"Command: {' '.join(ssh_cmd)}")
    print()
    print("Starting tunnel (Ctrl+C to stop)...")
    print()

    try:
        subprocess.run(ssh_cmd)
    except KeyboardInterrupt:
        print("\nTunnel closed.")


def print_usage():
    """Print usage information."""
    print(__doc__)


def main():
    args = sys.argv[1:]

    if not args or args[0] in ["-h", "--help"]:
        print_usage()
        sys.exit(0)

    cmd = args[0]

    if cmd == "list":
        cmd_list()

    elif cmd == "add":
        if len(args) < 3:
            print("Usage: python kb_server.py add NAME URL [--devices=N] [--timeout=N]")
            sys.exit(1)
        name = args[1]
        url = args[2]
        devices = None
        timeout = 300
        for arg in args[3:]:
            if arg.startswith("--devices="):
                devices = int(arg.split("=")[1])
            elif arg.startswith("--timeout="):
                timeout = int(arg.split("=")[1])
        cmd_add(name, url, devices, timeout)

    elif cmd == "test":
        provider = args[1] if len(args) > 1 else None
        cmd_test(provider)

    elif cmd == "key":
        set_key = None
        for arg in args[1:]:
            if arg.startswith("--set="):
                set_key = arg.split("=", 1)[1]
        cmd_key(set_key)

    elif cmd == "tunnel":
        if len(args) < 2:
            print("Usage: python kb_server.py tunnel user@host:port [--local=PORT]")
            sys.exit(1)
        target = args[1]
        local_port = None
        for arg in args[2:]:
            if arg.startswith("--local="):
                local_port = arg.split("=")[1]
        cmd_tunnel(target, local_port)

    else:
        print(f"Unknown command: {cmd}")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
