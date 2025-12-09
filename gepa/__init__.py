"""
GEPA Integration for Triton-AG

GEPA (Genetic-Pareto) is a framework for optimizing text components of systems
using LLM-based reflection and evolutionary search.

Usage:
    import gepa
    from triton_ag.gepa.adapters import TritonAdapter

    result = gepa.optimize(
        seed_candidate={"system_prompt": "..."},
        trainset=trainset,
        valset=valset,
        adapter=TritonAdapter(...),
        reflection_lm="openai/gpt-4.1-mini",
        max_metric_calls=150,
    )
"""

__version__ = "0.1.0"
