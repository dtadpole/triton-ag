"""
PKPO Reward Transformations
From: Pass@K Policy Optimization (Walder & Karkhanis, 2025)
Paper: https://arxiv.org/abs/2505.15201

This module implements the reward transformations for optimizing pass@k objectives.
The main function to use is `sloo_minus_one` which transforms a batch of rewards
to optimize for the maximum reward (pass@k) instead of average reward (pass@1).
"""

import numpy as np
from typing import Callable


def _m_normed(N: int, K: int, i: int, j: int) -> float:
    """Helper function for computing normalized matrix elements."""
    if i == j and i >= K - 1:
        return (
            K / (N - K + 1) *
            np.prod(np.arange(i - K + 2, i + 1) / np.arange(N - K + 2, N + 1))
        )
    elif j > i and j >= K - 1 and K >= 2:
        return (
            K / (N - K + 1) * (K - 1) / N *
            np.prod(np.arange(j - K + 2, j) / np.arange(N - K + 2, N))
        )
    return 0


def _m_diagonal(N: int, K: int) -> np.ndarray:
    """Compute diagonal elements of transformation matrix."""
    return np.array([_m_normed(N, K, i, i) for i in range(N)])


def rho(g: np.ndarray, K: int) -> float:
    """
    Unbiased estimator of maxg@k (Equation 12).

    Args:
        g: Raw rewards (will be sorted internally)
        K: Target k value

    Returns:
        Estimated maxg@k value
    """
    return (np.sort(g) * _m_diagonal(len(g), K)).sum()


def _delta(N: int, K: int, i: int) -> float:
    """Helper for computing delta values."""
    return _m_normed(N, K, i, i + 1) - _m_normed(N, K, i + 1, i + 1)


def _deltas(N: int, K: int) -> np.ndarray:
    """Compute array of delta values."""
    return np.array([_delta(N - 1, K, i) for i in range(N - 2)])


def _sorted_apply(func: Callable) -> Callable:
    """
    Decorator that sorts input array, applies function, then un-sorts output.
    This ensures the transformation respects the original sample ordering.
    """
    def inner(x: np.ndarray, *args, **kwargs) -> np.ndarray:
        i_sort = np.argsort(x)
        func_x = np.zeros_like(x)
        func_x[i_sort] = func(x[i_sort], *args, **kwargs)
        return func_x
    return inner


@_sorted_apply
def s(g: np.ndarray, K: int):
    """
    Compute per-sample transformed rewards (Equation 19).

    This is the base transformation without variance reduction baseline.

    Args:
        g: Raw rewards (assumed sorted by decorator)
        K: Target k value

    Returns:
        Transformed rewards
    """
    N = len(g)
    c = g * _m_diagonal(N, K)
    c[:(N - 1)] += g[1:] * _deltas(N + 1, K)
    return np.cumsum(c[::-1])[::-1]


@_sorted_apply
def _b(g: np.ndarray, K: int) -> np.ndarray:
    """
    Compute leave-one-out baseline values.

    Helper function for variance reduction.

    Args:
        g: Raw rewards (assumed sorted by decorator)
        K: Target k value

    Returns:
        Baseline values for each sample
    """
    N = len(g)
    w = (_m_diagonal(N - 1, K) * np.arange(1, N)).astype(float)
    w[1:] += _deltas(N, K) * np.arange(1, N - 1)
    c1 = np.array([(w * g[1:]).sum()])
    c2 = (g[:-1] - g[1:]) * w
    return np.cumsum(np.concatenate((c1, c2)))


def sloo(g: np.ndarray, K: int) -> np.ndarray:
    """
    Transform rewards with leave-one-out baseline (Equation 29).

    This version uses a leave-one-out baseline for variance reduction.

    Args:
        g: Raw rewards array of shape [n]
        K: Target k value (1 <= K <= n)

    Returns:
        Transformed rewards that optimize pass@K
    """
    return s(g, K) - _b(g, K) / (len(g) - 1)


def sloo_minus_one(g: np.ndarray, K: int) -> np.ndarray:
    """
    Transform rewards with (k-1) baseline (Equation 33) - RECOMMENDED.

    This is the paper's recommended transformation. It uses pass@(k-1) as the
    baseline, which provides better variance reduction than sloo.

    Args:
        g: Raw rewards array of shape [n]
        K: Target k value (1 <= K <= n)

    Returns:
        Transformed rewards that optimize pass@K

    Example:
        >>> # You have 8 samples with raw rewards
        >>> raw_rewards = np.array([0.0, 0.5, 0.2, 1.0, 0.3, 0.0, 0.8, 0.6])
        >>>
        >>> # Transform to optimize pass@8
        >>> transformed = sloo_minus_one(raw_rewards, K=8)
        >>>
        >>> # Use transformed rewards in your policy gradient
        >>> advantages = transformed  # Already centered
        >>> loss = -log_probs * advantages
    """
    if K == 1:
        # For K=1, just use standard mean centering
        return g - g.mean()

    return s(g, K) - _b(g, K - 1) * K / (K - 1) / len(g)


# Example usage and testing
if __name__ == "__main__":
    print("PKPO Reward Transformation Demo")
    print("=" * 50)

    # Simulate 8 samples with varying rewards
    np.random.seed(42)
    raw_rewards = np.array([0.0, 0.5, 0.2, 1.0, 0.3, 0.0, 0.8, 0.6])

    print(f"\nRaw rewards: {raw_rewards}")
    print(f"Mean: {raw_rewards.mean():.3f}, Max: {raw_rewards.max():.3f}")

    # Standard GRPO (optimizes average)
    standard_advantages = raw_rewards - raw_rewards.mean()
    print(f"\nStandard GRPO advantages (optimizes pass@1):")
    print(f"{standard_advantages}")

    # PKPO for different k values
    for k in [2, 4, 8]:
        pkpo_advantages = sloo_minus_one(raw_rewards, K=k)
        print(f"\nPKPO advantages (optimizes pass@{k}):")
        print(f"{pkpo_advantages}")
        print(f"Notice: Highest reward gets more weight for larger k")

    print("\n" + "=" * 50)
    print("Key insight: Higher k values put more weight on the")
    print("best samples, encouraging diversity and exploration!")
