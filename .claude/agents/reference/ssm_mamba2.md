# SSM/Mamba2 Reference
<!-- Updated: 2026-02-15 | Source: level3_20260214_235132, level3_20260215_020905, level3_20260215_122506 -->

## Code Templates
<!-- No templates yet. Add proven kernel patterns here as they are discovered. -->

## Tier 1: Algorithm Alternatives

No entries yet. All evaluated Mamba2 tasks were structurally broken (see Anti-Patterns).

## Tier 2: Architecture Variants

No entries yet.

## Tier 3-4: Tuning Guide

No entries yet.

## Anti-Patterns

### level3: 48_Mamba2ReturnY (0x, iter 0) -- einops import blocks evaluation
**Key insight**: Tasks whose reference Model imports `einops` are unevaluable because the package is not installed on the eval server.
**Why it failed**: The reference model uses `from einops import rearrange` at module level. The eval harness must load the reference Model to compare outputs, so a missing import makes the entire task fail before any kernel code is even tested. All 20 iterations produced ModuleNotFoundError.
**Better approach**: Skip these tasks immediately. Check for third-party imports (einops, flash_attn, etc.) in the reference Model before attempting optimization.

### level3: 49_Mamba2ReturnFinalState (0x, iter 0) -- einops import blocks evaluation
**Key insight**: Same root cause as 48_Mamba2ReturnY -- einops dependency in reference model.
**Why it failed**: Identical to above. Manual reshape replacements in ModelNew cannot help because the reference Model itself fails to load.
**Better approach**: Skip. No amount of kernel optimization works around a broken reference implementation.

## Decision Tree
1. **Pre-check**: Verify reference Model has no missing third-party imports (einops, flash_attn). If blocked, skip immediately.
2. Check Tier 1 algebraic shortcuts first (none known yet for Mamba2).
3. If no shortcut: select from Tier 1-2 for explore phase.
4. Exploit phase: apply Tier 3-4 tuning.
