# Claude/ — Design Documents

This folder contains all design documents for the kernel-bench multi-agent system.

## Living Documents (keep up to date)

These two documents must always reflect the current state of the system:

| Document | Purpose | When to update |
|---|---|---|
| **[Architecture.md](Architecture.md)** | Current system architecture: agent roles, MCP server, eval pipeline, session management, learning system | Any change to agent responsibilities, MCP tools, state management, or system behavior |
| **[User Manual.md](User%20Manual.md)** | Setup guide: SSH tunnels, server config, environment setup, common workflows | Any change to setup steps, configuration files, or operational procedures |

## Phase History (append-only)

Each phase documents the design at the time it was implemented. These are historical snapshots — do not modify them after the next phase begins.

| Document | Date | Summary |
|---|---|---|
| [00 - Codebase Introduction](00%20-%20Codebase%20Introduction.md) | 2026-01-24 | Original triton-ag RL codebase overview (pre-Claude Code integration) |
| [01 - Phase 1 - Claude Code Integration](01%20-%20Phase%201%20-%20Claude%20Code%20Integration.md) | 2026-01-25 | Initial MCP server, single-task agent, eval pipeline |
| [02 - Phase 2 - Skill and Multi-Agent](02%20-%20Phase%202%20-%20Skill%20and%20Multi-Agent.md) | 2026-02-05 | Skill command, parallel workers, strategy sub-agents, learning system |
| [03 - Phase 3 - Batch Architecture](03%20-%20Phase%203%20-%20Batch%20Architecture.md) | 2026-02-12 | Batch resilience: atomic claiming, session state, reflection pipeline |
| [04 - Phase 4 - Supervisor Architecture](04%20-%20Phase%204%20-%20Supervisor%20Architecture.md) | 2026-02-14 | Supervisor agent, agent rename (worker/optimizer), retry logic, completion tracking |
| [05 - Phase 5 - Controller Pattern](05%20-%20Phase%205%20-%20Controller%20Pattern.md) | 2026-02-14 | Skill agent as lightweight verification controller with post-execution checklist |

## Analysis & Reference

| Document | Purpose |
|---|---|
| [Reward Hacking Analysis](Reward%20Hacking%20Analysis.md) | Analysis of gaming patterns in kernel benchmark results (22.4% of tasks) |
| [References/](References/) | Supporting materials: Claude Code integration notes, multi-agent tradeoffs, workflow docs |

## Conventions

1. **Architecture.md is the source of truth.** If it contradicts a phase doc, Architecture.md is correct.
2. **Phase docs are immutable after the next phase ships.** They capture the design rationale at that point in time.
3. **New phases get the next number.** Phase 5 would be `05 - Phase 5 - {Name}.md`.
4. **Update Architecture.md and User Manual.md with every significant change.** These are the only docs that should stay current.
