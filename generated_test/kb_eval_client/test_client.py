#!/usr/bin/env python3
"""
Test script for the KB Eval Client
Demonstrates how to use the evaluation client programmatically
"""

import sys
import os
import json
from pathlib import Path

# Add the parent directory to the path to import sequential_eval
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sequential_eval import KbEvalClient

def test_json_input():
    """Test evaluation with JSON input format"""
    print("=" * 50)
    print("Testing JSON Input Format")
    print("=" * 50)
    
    try:
        client = KbEvalClient()
        
        # Test with the sample JSON file
        input_file = "sample_input.json"
        result = client.evaluate(input_file)
        
        print(f"✅ JSON evaluation completed successfully!")
        print(f"Compiled: {result.get('compiled', False)}")
        print(f"Correctness: {result.get('correctness', False)}")
        print(f"Runtime: {result.get('runtime', -1):.2f} microseconds")
        
        return True
        
    except Exception as e:
        print(f"❌ JSON evaluation failed: {e}")
        return False

def test_text_input():
    """Test evaluation with simple text input format"""
    print("\n" + "=" * 50)
    print("Testing Simple Text Input Format")
    print("=" * 50)
    
    try:
        client = KbEvalClient()
        
        # Test with the simple text file
        input_file = "simple_input.txt"
        result = client.evaluate(input_file)
        
        print(f"✅ Text evaluation completed successfully!")
        print(f"Compiled: {result.get('compiled', False)}")
        print(f"Correctness: {result.get('correctness', False)}")
        print(f"Runtime: {result.get('runtime', -1):.2f} microseconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Text evaluation failed: {e}")
        return False

def test_reference_only():
    """Test reference-only evaluation"""
    print("\n" + "=" * 50)
    print("Testing Reference-Only Evaluation")
    print("=" * 50)
    
    try:
        client = KbEvalClient()
        
        # Test reference-only evaluation
        input_file = "sample_input.json"
        result = client.evaluate(input_file, reference_only=True)
        
        print(f"✅ Reference-only evaluation completed successfully!")
        print(f"Compiled: {result.get('compiled', False)}")
        print(f"Runtime: {result.get('runtime', -1):.2f} microseconds")
        print(f"Is Reference Only: {result.get('metadata', {}).get('is_reference_only', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Reference-only evaluation failed: {e}")
        return False

def test_server_connection():
    """Test server connection"""
    print("\n" + "=" * 50)
    print("Testing Server Connection")
    print("=" * 50)
    
    try:
        client = KbEvalClient()
        server_url = client.pick_server()
        print(f"✅ Server available at: {server_url}")
        
        # Try to get server stats
        import requests
        response = requests.get(f"{server_url}/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print(f"Server stats: {stats}")
            return True
        else:
            print(f"❌ Server responded with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Server connection failed: {e}")
        print("Note: Make sure kbEvalRemoteServer is running")
        return False

def main():
    """Run all tests"""
    print("KB Eval Client Test Suite")
    print("=" * 50)
    
    # Change to the directory containing the test files
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Run tests
    tests = [
        ("Server Connection", test_server_connection),
        ("JSON Input", test_json_input),
        ("Text Input", test_text_input), 
        ("Reference Only", test_reference_only),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit(main()) 