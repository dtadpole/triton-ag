#!/usr/bin/env python3
"""
Minimal test script to verify GEPA dependencies are installed correctly.

Usage:
    1. Install dependencies: pip install -r requirements.txt
    2. Set your OpenAI API key: export OPENAI_API_KEY=your_key_here
    3. Run this script: python examples/test_dependencies.py
"""

import sys

# Append paths to sys.path at the END (after site-packages)
# - /workspace: for custom modules in workspace root
# - /workspace/gepa: for importing adapters directly (without gepa. prefix)
# This ensures installed packages (like gepa) take precedence.
if '/workspace' not in sys.path:
    sys.path.append('/workspace')
if '/workspace/gepa' not in sys.path:
    sys.path.append('/workspace/gepa')


def test_gepa_import():
    """Test that GEPA core package can be imported."""
    print("Testing GEPA import...")
    try:
        import gepa
        from gepa import GEPAResult, optimize  # noqa: F401
        from gepa.core.adapter import (  # noqa: F401
            EvaluationBatch, GEPAAdapter
        )

        ver = getattr(gepa, "__version__", "unknown")
        print(f"  ✓ GEPA version: {ver}")
        print("  ✓ Core imports: optimize, GEPAResult, GEPAAdapter")
        return True
    except ImportError as e:
        print(f"  ✗ Failed to import GEPA: {e}")
        return False


def test_litellm_import():
    """Test that litellm is available."""
    print("\nTesting litellm import...")
    try:
        import litellm

        # Handle different litellm versions
        try:
            ver = litellm.__version__
        except AttributeError:
            try:
                from litellm.version import __version__ as ver
            except ImportError:
                ver = "installed"

        print(f"  ✓ litellm version: {ver}")
        return True
    except ImportError as e:
        print(f"  ✗ Failed to import litellm: {e}")
        return False


def test_dspy_import():
    """Test that DSPy is available (optional)."""
    print("\nTesting DSPy import (optional)...")
    try:
        import dspy
        from dspy.teleprompt.gepa import GEPA  # noqa: F401

        ver = dspy.__version__ if hasattr(dspy, "__version__") else "unknown"
        print(f"  ✓ DSPy version: {ver}")
        print("  ✓ DSPy GEPA teleprompter available")
        return True
    except ImportError as e:
        print(f"  ⚠ DSPy not available (optional): {e}")
        return False


def test_utilities():
    """Test utility packages."""
    print("\nTesting utility packages...")
    success = True

    try:
        import tqdm

        print(f"  ✓ tqdm version: {tqdm.__version__}")
    except ImportError as e:
        print(f"  ✗ tqdm not available: {e}")
        success = False

    try:
        import yaml

        print(f"  ✓ PyYAML version: {yaml.__version__}")
    except ImportError as e:
        print(f"  ✗ PyYAML not available: {e}")
        success = False

    try:
        import datasets

        print(f"  ✓ datasets version: {datasets.__version__}")
    except ImportError as e:
        print(f"  ⚠ datasets not available (optional): {e}")

    return success


def test_minimal_gepa_workflow():
    """
    Test a minimal GEPA workflow without actually calling an LLM.
    This verifies the core classes and data structures work.
    """
    print("\nTesting minimal GEPA workflow...")
    try:
        from gepa.core.adapter import EvaluationBatch
        from gepa.core.state import GEPAState  # noqa: F401

        # Test EvaluationBatch creation
        eval_batch = EvaluationBatch(
            outputs=["output1", "output2"],
            scores=[0.8, 0.9],
            trajectories=None,
        )
        n_outputs = len(eval_batch.outputs)
        avg_score = sum(eval_batch.scores) / len(eval_batch.scores)
        print(f"  ✓ EvaluationBatch: {n_outputs} outputs, avg={avg_score:.2f}")

        # Test seed candidate structure
        seed_candidate = {
            "system_prompt": "You are a helpful assistant.",
            "instruction": "Answer the question step by step.",
        }
        print(f"  ✓ Seed candidate valid: {list(seed_candidate.keys())}")

        return True
    except Exception as e:
        print(f"  ✗ Workflow test failed: {e}")
        return False


def test_llm_connection(skip_if_no_key=True):
    """
    Test LLM connection via litellm (requires API key).
    Set skip_if_no_key=False to fail if no API key is set.
    """
    import os

    print("\nTesting LLM connection...")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        if skip_if_no_key:
            print("  ⚠ Skipped: OPENAI_API_KEY not set")
            return True
        else:
            print("  ✗ OPENAI_API_KEY not set")
            return False

    try:
        import litellm

        # Make a minimal test call
        response = litellm.completion(
            model="openai/gpt-4.1-mini",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10,
        )
        result = response.choices[0].message.content
        print(f"  ✓ LLM connection successful. Response: {result}")
        return True
    except Exception as e:
        print(f"  ✗ LLM connection failed: {e}")
        return False


def main():
    """Run all dependency tests."""
    print("=" * 60)
    print("GEPA Dependency Test")
    print("=" * 60)

    results = {
        "GEPA Import": test_gepa_import(),
        "LiteLLM Import": test_litellm_import(),
        "DSPy Import": test_dspy_import(),
        "Utilities": test_utilities(),
        "GEPA Workflow": test_minimal_gepa_workflow(),
        "LLM Connection": test_llm_connection(skip_if_no_key=True),
    }

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
        if not passed and test_name not in ["DSPy Import"]:  # DSPy is optional
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All required dependencies are correctly installed!")
        print("\nNext steps:")
        print("  1. Set OPENAI_API_KEY environment variable")
        print("  2. Try the basic_optimization.py example")
        return 0
    else:
        print("Some dependencies are missing. Please install with:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
