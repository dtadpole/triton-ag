#!/bin/bash

echo "=== Testing vLLM Integration ==="

# Test 1: Simple vLLM API test
echo "1. Testing vLLM API directly..."
python generated_test/vllm_test/test_vllm_completions.py

if [ $? -eq 0 ]; then
    echo "✓ vLLM API test passed"
    
    # Test 2: Single file processing
    echo "2. Testing single file processing..."
    python sequential_inference.py --client vllm --input-dir generated_test/simple_test --output-dir ./_output_vllm_test --num-tasks 1
    
    if [ $? -eq 0 ]; then
        echo "✓ Single file processing passed"
        
        # Test 3: Batch processing (small subset)
        echo "3. Testing batch processing..."
        python sequential_inference.py --client vllm --input-dir ./kernel_bench/level1 --output-dir ./_output_vllm_batch --num-tasks 2
        
        if [ $? -eq 0 ]; then
            echo "✓ All vLLM tests passed!"
            echo "🎉 vLLM integration is working correctly!"
        else
            echo "✗ Batch processing failed"
        fi
    else
        echo "✗ Single file processing failed"
    fi
else
    echo "✗ vLLM API test failed - server may be down"
fi 