#!/usr/bin/env python3
"""
Evaluation client that takes input file, calls kbEvalRemoteServer and writes result as kbEval.json
"""

import argparse
import json
import os
import requests
import yaml
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import traceback

class KbEvalClient:
    def __init__(self, config_path: str = "kbEval.yaml"):
        """Initialize the client with configuration"""
        self.config_path = config_path
        self.load_config()
        
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, "r") as f:
                self.config = yaml.load(f, Loader=yaml.FullLoader)
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_path} not found. Using default localhost server.")
            self.config = {
                "kbEvalClient": {
                    "servers": [{"url": "http://localhost:5678"}]
                }
            }
    
    def pick_server(self) -> str:
        """Pick the best available server based on load balancing"""
        servers = self.config["kbEvalClient"]["servers"]
        
        if len(servers) == 1:
            return servers[0]["url"]
            
        # Try to get stats from all servers and pick the least loaded one
        best_server = None
        min_load = float("inf")
        
        for server in servers:
            try:
                response = requests.get(f"{server['url']}/stats", timeout=5)
                if response.status_code == 200:
                    stats = response.json()
                    load = stats.get("pending_requests", 0) / max(stats.get("num_devices", 1), 1)
                    if load < min_load:
                        min_load = load
                        best_server = server["url"]
            except Exception as e:
                print(f"Warning: Could not get stats from {server['url']}: {e}")
                
        return best_server or servers[0]["url"]
    
    def read_code_file(self, file_path: str) -> str:
        """Read code from file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise FileNotFoundError(f"Could not read file {file_path}: {e}")
    
    def parse_input_file(self, input_file_path: str) -> Dict[str, Any]:
        """Parse input file and extract evaluation parameters"""
        input_path = Path(input_file_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file {input_file_path} does not exist")
            
        # Try to parse as JSON first
        try:
            with open(input_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Validate required fields for JSON format
            required_fields = ["model_tag", "task_tag", "reference_code_path", "generated_code_path"]
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                raise ValueError(f"Missing required fields in JSON: {missing_fields}")
                
            # Convert relative paths to absolute paths relative to input file directory
            base_dir = input_path.parent
            reference_path = base_dir / data["reference_code_path"]
            generated_path = base_dir / data["generated_code_path"]
            
            return {
                "model_tag": data["model_tag"],
                "task_tag": data["task_tag"],
                "eval_tag": data.get("eval_tag", "eval"),
                "time_tag": data.get("time_tag", datetime.now().strftime("%Y%m%d_%H%M%S")),
                "reference_code": self.read_code_file(str(reference_path)),
                "generated_code": self.read_code_file(str(generated_path)),
                "metadata": data.get("metadata", {})
            }
            
        except (json.JSONDecodeError, ValueError) as e:
            # If JSON parsing fails, try to interpret as a simple text file with file paths
            try:
                with open(input_file_path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                
                if len(lines) < 2:
                    raise ValueError("Input file must contain at least reference and generated code paths")
                
                base_dir = input_path.parent
                reference_path = base_dir / lines[0]
                generated_path = base_dir / lines[1]
                
                # Extract task info from file names/paths
                task_tag = input_path.stem
                model_tag = "unknown"
                
                return {
                    "model_tag": model_tag,
                    "task_tag": task_tag,
                    "eval_tag": "eval",
                    "time_tag": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "reference_code": self.read_code_file(str(reference_path)),
                    "generated_code": self.read_code_file(str(generated_path)),
                    "metadata": {}
                }
                
            except Exception as text_error:
                raise ValueError(f"Could not parse input file as JSON ({e}) or text format ({text_error})")
    
    def call_kb_eval_server(self, eval_params: Dict[str, Any]) -> Dict[str, Any]:
        """Call the kbEvalRemoteServer with evaluation parameters"""
        server_url = self.pick_server()
        
        print(f"Calling server: {server_url}")
        print(f"Evaluating: {eval_params['model_tag']} / {eval_params['task_tag']} / {eval_params['eval_tag']}")
        
        try:
            response = requests.post(
                f"{server_url}/kb_eval",
                json={
                    "model_tag": eval_params["model_tag"],
                    "task_tag": eval_params["task_tag"],
                    "eval_tag": eval_params["eval_tag"],
                    "time_tag": eval_params["time_tag"],
                    "reference_code": eval_params["reference_code"],
                    "generated_code": eval_params["generated_code"],
                },
                headers={"Content-Type": "application/json"},
                timeout=300  # 5 minute timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"Server returned status {response.status_code}: {response.text}")
                
            result = response.json()
            
            # Add metadata from input if available
            if "metadata" not in result:
                result["metadata"] = {}
            result["metadata"].update(eval_params.get("metadata", {}))
            result["metadata"].update({
                "model_tag": eval_params["model_tag"],
                "task_tag": eval_params["task_tag"],
                "eval_tag": eval_params["eval_tag"],
                "time_tag": eval_params["time_tag"],
                "server_url": server_url,
                "evaluation_timestamp": datetime.now().isoformat()
            })
            
            return result
            
        except requests.exceptions.Timeout:
            raise Exception("Server request timed out after 5 minutes")
        except requests.exceptions.ConnectionError:
            raise Exception(f"Could not connect to server at {server_url}")
        except Exception as e:
            raise Exception(f"Error calling server: {e}")
    
    def call_kb_eval_reference(self, eval_params: Dict[str, Any]) -> Dict[str, Any]:
        """Call the kbEvalRemoteServer for reference evaluation only"""
        server_url = self.pick_server()
        
        print(f"Calling server for reference evaluation: {server_url}")
        print(f"Evaluating reference: {eval_params['model_tag']} / {eval_params['task_tag']}")
        
        try:
            response = requests.post(
                f"{server_url}/kb_eval_ref",
                json={
                    "model_tag": eval_params["model_tag"],
                    "task_tag": eval_params["task_tag"],
                    "time_tag": eval_params["time_tag"],
                    "reference_code": eval_params["reference_code"],
                },
                headers={"Content-Type": "application/json"},
                timeout=300  # 5 minute timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"Server returned status {response.status_code}: {response.text}")
                
            result = response.json()
            
            # Add metadata from input if available
            if "metadata" not in result:
                result["metadata"] = {}
            result["metadata"].update(eval_params.get("metadata", {}))
            result["metadata"].update({
                "model_tag": eval_params["model_tag"],
                "task_tag": eval_params["task_tag"],
                "time_tag": eval_params["time_tag"],
                "server_url": server_url,
                "evaluation_timestamp": datetime.now().isoformat(),
                "is_reference_only": True
            })
            
            return result
            
        except requests.exceptions.Timeout:
            raise Exception("Server request timed out after 5 minutes")
        except requests.exceptions.ConnectionError:
            raise Exception(f"Could not connect to server at {server_url}")
        except Exception as e:
            raise Exception(f"Error calling server: {e}")
    
    def write_result(self, input_file_path: str, result: Dict[str, Any]):
        """Write evaluation result as kbEval.json in the same folder as input file"""
        input_path = Path(input_file_path)
        output_path = input_path.parent / "kbEval.json"
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"Result written to: {output_path}")
        except Exception as e:
            raise Exception(f"Could not write result file {output_path}: {e}")
    
    def evaluate(self, input_file_path: str, reference_only: bool = False) -> Dict[str, Any]:
        """Main evaluation function"""
        try:
            # Parse input file
            eval_params = self.parse_input_file(input_file_path)
            
            # Call appropriate server endpoint
            if reference_only:
                result = self.call_kb_eval_reference(eval_params)
            else:
                result = self.call_kb_eval_server(eval_params)
            
            # Write result
            self.write_result(input_file_path, result)
            
            return result
            
        except Exception as e:
            error_result = {
                "compiled": False,
                "correctness": False,
                "metadata": {
                    "client_error": str(e),
                    "client_error_traceback": traceback.format_exc(),
                    "evaluation_timestamp": datetime.now().isoformat()
                },
                "runtime": -1.0,
                "runtime_stats": {}
            }
            
            # Still write the error result
            try:
                self.write_result(input_file_path, error_result)
            except:
                print(f"Could not write error result to file")
            
            raise e


def main():
    parser = argparse.ArgumentParser(
        description="Kernel Bench Evaluation Client - calls kbEvalRemoteServer and writes results"
    )
    parser.add_argument(
        "input_file",
        help="Input file containing evaluation parameters (JSON format) or file paths"
    )
    parser.add_argument(
        "--config",
        default="kbEval.yaml",
        help="Configuration file for server endpoints (default: kbEval.yaml)"
    )
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Evaluate reference code only (skip generated code)"
    )
    parser.add_argument(
        "--model-tag",
        help="Override model tag from input file"
    )
    parser.add_argument(
        "--task-tag", 
        help="Override task tag from input file"
    )
    parser.add_argument(
        "--eval-tag",
        help="Override eval tag from input file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        # Create client
        client = KbEvalClient(config_path=args.config)
        
        # Override parameters if provided
        if args.verbose:
            print(f"Processing input file: {args.input_file}")
        
        # Evaluate
        result = client.evaluate(args.input_file, reference_only=args.reference_only)
        
        # Print summary
        print("\n=== Evaluation Summary ===")
        print(f"Compiled: {result.get('compiled', False)}")
        print(f"Correctness: {result.get('correctness', False)}")
        print(f"Runtime: {result.get('runtime', -1):.2f} microseconds")
        
        if result.get('correctness') and result.get('runtime', -1) > 0:
            print("✅ Evaluation completed successfully!")
        elif result.get('compiled') and not result.get('correctness'):
            print("⚠️  Code compiled but correctness check failed")
        elif not result.get('compiled'):
            print("❌ Code compilation failed")
        else:
            print("❓ Evaluation completed with unknown status")
            
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        if args.verbose:
            print("\nDetailed error:")
            print(traceback.format_exc())
        exit(1)


if __name__ == "__main__":
    main()
