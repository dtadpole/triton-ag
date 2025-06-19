#!/usr/bin/env python3
"""
Test script for Qwen3 tokenizer from HuggingFace
This script loads the Qwen3/Qwen3-8B tokenizer and tests it against a simple string.
"""

import os
import sys
from transformers import AutoTokenizer

def test_qwen3_tokenizer():
    """Test the Qwen3 tokenizer by loading it and tokenizing a simple string."""
    
    print("Testing Qwen3/Qwen3-8B tokenizer...")
    
    # Load the tokenizer from HuggingFace
    try:
        print("Loading tokenizer from HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        print(f"✓ Tokenizer loaded successfully: {tokenizer.__class__.__name__}")
        
        # Test with a simple string
        test_string = "Hello, world! This is a test for the Qwen3 tokenizer."
        print(f"\nTest string: '{test_string}'")
        
        # Tokenize the string
        tokens = tokenizer.tokenize(test_string)
        print(f"Tokens: {tokens}")
        print(f"Number of tokens: {len(tokens)}")
        
        # Encode the string (get token IDs)
        token_ids = tokenizer.encode(test_string, add_special_tokens=True)
        print(f"Token IDs: {token_ids}")
        print(f"Number of token IDs: {len(token_ids)}")
        
        # Decode back to string
        decoded_string = tokenizer.decode(token_ids)
        print(f"Decoded string: '{decoded_string}'")
        
        # Check if encoding/decoding is consistent
        if test_string.strip() == decoded_string.strip():
            print("✓ Encoding/decoding is consistent!")
        else:
            print("⚠ Warning: Decoded string differs from original")
            
        # Test with special tokens
        print(f"\nSpecial tokens:")
        print(f"BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
        print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        print(f"UNK token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
        
        # Vocab size
        print(f"Vocab size: {tokenizer.vocab_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading or testing tokenizer: {e}")
        return False

def main():
    """Main function to run the tokenizer test."""
    print("=" * 60)
    print("Qwen3 Tokenizer Test")
    print("=" * 60)
    
    success = test_qwen3_tokenizer()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 