# Multi-Agent Architecture with Claude Code

> **Reference document for `CLAUDE_CODE_AGENT_ADVANCED.md`** - Contains detailed multi-agent patterns and reliability research.

## Overview

This document describes a multi-agent system built on Claude Code using markdown-defined agents. Each agent runs in its own context window with true parallelism, orchestrated by Claude Code without requiring API access or custom infrastructure.

```
CLAUDE.md + Agent Prompts (MD files)
            ↓
      Claude Code (parent)
            ↓
      Task tool spawns subagents
            ↓
    ┌───────┼───────┐
    ↓       ↓       ↓
 Agent A  Agent B  Agent C   ← Separate context windows
```

**Key Properties:**
- Separate context windows per agent
- True parallel execution
- No API key required (uses Claude Code subscription)
- Easy iteration (edit MD files)
- Claude-guided orchestration

---

## Project Structure

```
project/
├── CLAUDE.md                        # Main orchestration & workflows
├── .claude/
│   ├── agents/
│   │   ├── coordinator.md           # Coordinator agent (supervisor)
│   │   ├── coder.md                 # Coder worker agent
│   │   ├── analysis.md              # Analysis worker agent
│   │   └── eval.md                  # Eval worker agent (resource-constrained)
│   ├── memory/
│   │   ├── preferences.md           # User preferences (persistent)
│   │   ├── decisions.md             # Architectural decisions
│   │   └── learnings.md             # Past learnings
│   ├── logs/
│   │   └── workflow-trace.md        # Running log of agent progress
│   └── tools/
│       ├── query-db.sh              # Database helper
│       └── notify-slack.sh          # Notification helper
```

---

## Agent Hierarchy

```
                    ┌─────────────────┐
                    │   Coordinator   │
                    │   (supervisor)  │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            ↓                ↓                ↓
     ┌──────────┐     ┌──────────┐     ┌──────────┐
     │  Coder   │     │ Analysis │     │   Eval   │
     │ (worker) │     │ (worker) │     │ (worker) │
     └──────────┘     └──────────┘     └──────────┘
                                        ↑
                                   HW constrained
```

| Agent | Role | Resource |
|-------|------|----------|
| Coordinator | Supervises workers, reviews progress, monitors logs | Unconstrained |
| Coder | Implementation, code changes | Unconstrained |
| Analysis | Data analysis, research, investigation | Unconstrained |
| Eval | Evaluation, testing, validation | **Hardware-constrained** |

---

## Core Design Elements

### 1. File-Based Memory System

Agents don't persist memory across sessions. Use explicit memory files that agents read and update.

**`.claude/memory/preferences.md`:**
```markdown
# User Preferences

## Code Style
- Use async/await, never callbacks
- Prefer functional programming patterns
- Always add TypeScript types

## Architecture
- Use dependency injection
- Prefer composition over inheritance

## Testing
- Require unit tests for all new functions
- Use vitest, not jest
```

**`.claude/memory/decisions.md`:**
```markdown
# Architectural Decisions

## 2024-01-15: Authentication
- Decision: JWT with refresh tokens
- Reason: Stateless, works with microservices
- Files: src/auth/*

## 2024-01-20: Database
- Decision: PostgreSQL with Prisma ORM
- Reason: Type safety, migration support
- Files: prisma/schema.prisma
```

---

### 2. Workflow Templates

Define explicit workflows to ensure consistent agent orchestration.

**Standard Workflow (Coordinator-led):**

```
1. Parent spawns Coordinator with:
   - Task description
   - Expected milestones
   - Success criteria

2. Coordinator breaks down task and spawns workers:
   - Analysis agents for investigation
   - Coder agents for implementation
   - Eval agents for validation (HW-gated)

3. After each worker:
   - Coordinator reviews output
   - Logs to workflow-trace.md
   - Approves, requests rework, or escalates

4. Coordinator gates phase transitions:
   - Analysis → Coder (when investigation complete)
   - Coder → Eval (when implementation ready)
   - Eval → Done (when validation passes)

5. Coordinator provides final summary
```

**Phase progression:**

| Phase | Workers | Gate Condition |
|-------|---------|----------------|
| Investigation | Analysis | Findings reviewed by Coordinator |
| Implementation | Coder | Code reviewed by Coordinator |
| Validation | Eval | Results reviewed by Coordinator |
| Complete | — | Coordinator final approval |

**Quick Task (No Coordinator):**
- Single file edits
- Typo fixes
- Documentation updates
- User says "quick" or "just do it"

---

### 3. Tool Extensions

Extend agent capabilities with shell scripts and MCP servers.

**Helper scripts (`.claude/tools/`):**

```bash
# .claude/tools/query-db.sh
#!/bin/bash
psql "$DATABASE_URL" -c "$1" --json

# .claude/tools/notify-slack.sh
#!/bin/bash
curl -X POST -H 'Content-type: application/json' \
  --data "{\"text\":\"$1\"}" "$SLACK_WEBHOOK_URL"
```

**MCP configuration (`.claude/mcp.json`):**
```json
{
  "servers": {
    "database": {
      "command": "npx",
      "args": ["@your-org/db-mcp-server"]
    }
  }
}
```

---

### 4. Resource-Constrained Agent Spawning

The Eval agent requires MCP tools only available on specific hardware. Control spawning to match resource availability.

| Agent | MCP Required | Hardware | Max Concurrent |
|-------|--------------|----------|----------------|
| Coordinator | None | Any | 1 (singleton) |
| Coder | None | Any | Unlimited |
| Analysis | None | Any | Unlimited |
| Eval | eval-compute | Eval nodes | Limited (check availability) |

**Spawning rules:**
1. Only one Coordinator active at a time
2. Check Eval hardware availability before spawning Eval agents
3. Prefer Analysis over Eval when possible (for pre-checks)
4. Queue Eval requests if at capacity

---

### 5. Structured Agent Communication

Use JSON for agent outputs to enable reliable handoffs and debugging.

**Worker output format:**
```json
{
  "agent": "coder | analysis | eval",
  "task_id": "task-123",
  "status": "completed | partial | blocked",
  "output": {},
  "blockers": [],
  "ready_for_review": true
}
```

**Coordinator output format:**
```json
{
  "agent": "coordinator",
  "review_type": "progress_check | quality_gate | final_review",
  "workers_reviewed": ["coder", "analysis"],
  "status": {
    "on_track": true,
    "completed_tasks": [],
    "blocked_tasks": [],
    "concerns": []
  },
  "actions_taken": [],
  "next_steps": [],
  "escalate_to_user": false
}
```

**Workflow trace log (`.claude/logs/workflow-trace.md`):**
```markdown
## [2024-01-20 10:30] Task: implement-rate-limiting

### Phase: Investigation
- Worker: Analysis
- Status: completed
- Findings: Found 5 API endpoints, no existing rate limiting
- Coordinator: Approved, proceeding to implementation

### Phase: Implementation
- Worker: Coder
- Status: completed
- Files: src/middleware/rateLimit.ts
- Coordinator: Approved, proceeding to validation

### Phase: Validation
- Worker: Eval
- Status: completed
- Resource: eval-compute node-1
- Results: All tests passed
- Coordinator: Approved, task complete

### Final Summary
- Total workers spawned: 3
- Revisions requested: 0
- Blockers encountered: 0
- Outcome: Success
```

---

## Agent Definitions

### Coordinator (`.claude/agents/coordinator.md`)

```markdown
# Coordinator Agent

You are the supervisor agent responsible for ensuring workers make progress and the overall workload stays on track.

## Responsibilities

1. **Monitor Worker Progress**
   - Review outputs from Coder, Analysis, and Eval agents
   - Detect stalled or blocked workers
   - Identify when workers are going off-track

2. **Review Workflow Logs**
   - Read .claude/logs/workflow-trace.md
   - Track completion status of each task
   - Identify bottlenecks or failures

3. **Ensure Workload Progress**
   - Compare current state to expected milestones
   - Flag tasks that are taking too long
   - Escalate blockers to user

4. **Quality Gate**
   - Verify worker outputs meet requirements
   - Request rework if quality is insufficient
   - Approve progression to next phase

## Input Expected
- Task breakdown with expected milestones
- Worker agent outputs to review
- Workflow logs

## Output Format
{
  "agent": "coordinator",
  "review_type": "progress_check | quality_gate | final_review",
  "workers_reviewed": ["coder", "analysis"],
  "status": {
    "on_track": true,
    "completed_tasks": ["task-1", "task-2"],
    "in_progress_tasks": ["task-3"],
    "blocked_tasks": [],
    "concerns": []
  },
  "actions_taken": [
    {"action": "approved coder output", "reason": "meets requirements"},
    {"action": "requested rework from analysis", "reason": "missing edge case"}
  ],
  "next_steps": ["spawn eval agent for validation"],
  "escalate_to_user": false
}

## Progress Monitoring Protocol

1. After each worker completes, review its output
2. Update .claude/logs/workflow-trace.md with status
3. If worker output is insufficient:
   - Provide specific feedback
   - Request targeted rework
   - Track revision count (max 3)
4. If worker is blocked:
   - Identify blocker cause
   - Attempt resolution or escalate
5. Periodically check overall timeline:
   - Compare progress to milestones
   - Alert if falling behind

## Rules
- Do not perform worker tasks yourself
- Focus on coordination and quality
- Be specific in feedback
- Escalate early rather than late
```

### Coder (`.claude/agents/coder.md`)

```markdown
# Coder Agent

You are a focused implementation agent.

## Initialization
1. Read .claude/memory/preferences.md
2. Apply all stated preferences

## Input Expected
- Task description from Coordinator
- Analysis findings (if available)
- Relevant file paths

## Task
1. Understand requirements and context
2. Follow existing patterns
3. Implement the solution
4. Report progress to Coordinator

## Output Format
{
  "agent": "coder",
  "task_id": "task-123",
  "status": "completed | partial | blocked",
  "inputs_received": {"from_agent": "coordinator", "key_points": ["summary"]},
  "decisions": [{"decision": "choice", "reason": "why"}],
  "files_modified": ["path/to/file"],
  "implementation_summary": "what was done",
  "blockers": [],
  "ready_for_review": true
}

## Rules
- Follow existing code style
- Don't over-engineer
- Flag ambiguities to Coordinator
- Report blockers immediately
```

### Analysis (`.claude/agents/analysis.md`)

```markdown
# Analysis Agent

You are a focused analysis and investigation agent.

## Initialization
1. Read .claude/memory/preferences.md
2. Read .claude/memory/decisions.md

## Input Expected
- Analysis task from Coordinator
- Scope and focus areas
- Relevant data or files

## Task
1. Search codebase or data sources
2. Analyze patterns and issues
3. Synthesize findings
4. Report to Coordinator

## Output Format
{
  "agent": "analysis",
  "task_id": "task-123",
  "status": "completed | partial | blocked",
  "scope_analyzed": ["files", "data sources"],
  "findings": [
    {"finding": "description", "severity": "high|medium|low", "evidence": "..."}
  ],
  "patterns_identified": ["pattern 1", "pattern 2"],
  "recommendations": ["recommendation 1"],
  "blockers": [],
  "ready_for_review": true
}

## Rules
- Do not modify files
- Verify by reading, don't assume
- Be thorough but concise
- Report blockers to Coordinator
```

### Eval (`.claude/agents/eval.md`)

```markdown
# Eval Agent

You are a focused evaluation and validation agent.

## Resource Requirements
- MCP Server: eval-compute
- Hardware: Eval node (limited availability)
- Only spawn when evaluation/testing is required

## Input Expected
- Evaluation task from Coordinator
- Artifacts to evaluate (code, data, outputs)
- Evaluation criteria

## Task
1. Set up evaluation environment
2. Run evaluation/tests
3. Collect results
4. Report to Coordinator

## Output Format
{
  "agent": "eval",
  "task_id": "task-123",
  "status": "completed | partial | blocked",
  "resource_used": "eval-compute MCP on node X",
  "evaluation_type": "unit_test | integration | benchmark | validation",
  "results": {
    "passed": true,
    "score": 0.95,
    "details": {}
  },
  "failures": [],
  "recommendations": [],
  "eval_time_seconds": 120,
  "blockers": [],
  "ready_for_review": true
}

## When to Use This Agent
- Running tests or benchmarks
- Validating outputs
- Performance evaluation
- Quality assessment requiring compute

## When NOT to Use
- Simple code review (use Coordinator)
- Analysis tasks (use Analysis agent)
- Pre-checks that don't need eval hardware

## Rules
- Report resource usage
- Include timing information
- Flag any evaluation failures clearly
- Report blockers to Coordinator
```

---

## Main Orchestrator Template (`CLAUDE.md`)

This is the complete orchestrator configuration to place in your project root.

```markdown
# Multi-Agent Orchestrator

## Session Initialization

At session start:
1. Read .claude/memory/preferences.md
2. Read .claude/memory/decisions.md
3. Apply preferences to all work

## Memory Updates

When user expresses a preference or we make a significant decision:
1. Update the relevant memory file
2. Confirm: "I've noted this in .claude/memory/[file].md"

## Agent Hierarchy

### Coordinator (supervisor)
- Spawned for any multi-step task
- Reviews all worker outputs
- Monitors .claude/logs/workflow-trace.md
- Gates progression between phases

### Workers (report to Coordinator)
| Agent | Use For | Resource |
|-------|---------|----------|
| Coder | Implementation | Unconstrained |
| Analysis | Investigation, research | Unconstrained |
| Eval | Testing, validation | **HW constrained** |

## Workflow Execution

### Standard Workflow (Coordinator-led)

1. **Spawn Coordinator** with task breakdown and milestones
2. Coordinator spawns workers as needed:
   - Analysis agents for investigation (parallel OK)
   - Coder agents for implementation (parallel OK)
   - Eval agents for validation (check HW availability)
3. After each worker completes:
   - Coordinator reviews output
   - Updates .claude/logs/workflow-trace.md
   - Approves, requests rework, or escalates
4. Coordinator gates progression:
   - Analysis complete → proceed to Coder
   - Coder complete → proceed to Eval
   - Eval complete → final review
5. Coordinator provides final summary

### Quick Task (No Coordinator)

For simple tasks, handle directly:
- Single file edits
- Obvious fixes
- User says "quick" or "just do it"

## Resource Management

Before spawning Eval agents:
1. Check eval-compute MCP availability
2. If unavailable, queue request
3. Log resource wait in workflow-trace.md

## Logging Protocol

All agents update .claude/logs/workflow-trace.md:

```
## [timestamp] Task: [task-id]

### Worker: [agent-name]
- Status: completed | partial | blocked
- Output summary: ...
- Blockers: ...

### Coordinator Review
- Approved: yes | no
- Feedback: ...
- Next: [next action]
```

## Progress Monitoring

Coordinator checks progress after each worker:
1. Is worker on track?
2. Are there blockers?
3. Is quality sufficient?
4. Should we escalate?

If falling behind expected milestones:
- Identify cause
- Attempt resolution
- Escalate to user if needed

## Communication Protocol

1. Parent spawns Coordinator with full task context
2. Coordinator spawns workers with specific subtasks
3. Workers report back to Coordinator in JSON
4. Coordinator synthesizes and reports to parent
5. Parent presents final result to user

## After Completion

Provide workflow summary:
- Coordinator's final assessment
- All workers spawned and their outputs
- Decisions made and rationale
- Files changed
- Any concerns or follow-ups

## Custom Tools

Available via Bash:
- .claude/tools/query-db.sh "SQL" — Database queries
- .claude/tools/notify-slack.sh "msg" — Notifications
```

---

## Quick Start

```bash
# 1. Create directory structure
mkdir -p .claude/agents .claude/memory .claude/logs .claude/tools

# 2. Initialize memory and log files
echo "# User Preferences" > .claude/memory/preferences.md
echo "# Decisions" > .claude/memory/decisions.md
echo "# Workflow Trace" > .claude/logs/workflow-trace.md

# 3. Create CLAUDE.md (copy Main Orchestrator Template above)

# 4. Create agent prompts in .claude/agents/
#    - coordinator.md
#    - coder.md
#    - analysis.md
#    - eval.md

# 5. Start Claude Code
claude

# 6. Test with a task
> "Analyze the auth module, implement rate limiting, and validate it works"
```

Claude will:
1. Spawn Coordinator with the task breakdown
2. Coordinator spawns Analysis agent to investigate auth module
3. Coordinator reviews findings, spawns Coder agent
4. Coordinator reviews implementation, spawns Eval agent (checks HW)
5. Coordinator provides final summary with workflow trace

---

## Agent Types Reference

| Agent | Role | Tools | Resource | Reports To |
|-------|------|-------|----------|------------|
| Coordinator | Supervise, review, gate progress | All read tools | Unconstrained | Parent |
| Coder | Implementation | All tools | Unconstrained | Coordinator |
| Analysis | Investigation, research | Glob, Grep, Read | Unconstrained | Coordinator |
| Eval | Testing, validation | eval-compute MCP | **HW constrained** | Coordinator |

---

## Reliability & Determinism Analysis

This section examines when and how the prompt-based multi-agent approach exhibits inconsistent behavior compared to code-based orchestration, backed by research data.

### Fundamental Non-Determinism

Even with temperature=0, LLMs produce inconsistent outputs. Research from [arXiv:2408.04667](https://arxiv.org/html/2408.04667v5) documents:

| Metric | Finding |
|--------|---------|
| Accuracy variance | **Up to 15%** across 10 identical runs |
| Performance gap | **Up to 70%** between best and worst runs on same task |
| Example | Mixtral-8x7b on college math: 75% best run vs 3% worst run |
| Token agreement (TARr@10) | Often **below 50%** for identical prompts |

**Root causes** ([source](https://mbrenndoerfer.com/writing/why-llms-are-not-deterministic)):
- Floating-point precision cascades over long generations
- GPU hardware variability (operation order affects rounding)
- Mixture-of-Experts routing influenced by concurrent requests
- Batching effects altering low-level numerical behavior

**Implication for this design:** The orchestrator (parent Claude) may classify the same task differently across sessions, spawn different numbers of agents, or synthesize findings differently—even with identical inputs.

---

### Multi-Agent Coordination Failures

Research from [arXiv:2503.13657](https://arxiv.org/html/2503.13657v1) identifies **14 distinct failure modes** in multi-agent LLM systems:

**Category 1: Specification & System Design (5 modes)**
- Task disobedience
- Role violation
- Step repetition
- Conversation history loss
- Unaware termination conditions

**Category 2: Inter-Agent Misalignment (6 modes)**
- Conversation resets
- Failure to seek clarification
- Task derailment
- Information withholding
- Ignored agent input
- Reasoning-action mismatch

**Category 3: Task Verification & Termination (3 modes)**
- Premature termination
- Incomplete/no verification
- Incorrect verification

**Critical finding:** "Failures are not isolated events; they have a cascading effect that influences other failure categories."

**Quantified impact:** ChatDev (a multi-agent coding system) achieves **as low as 25% accuracy** on some benchmarks. Enhanced prompting improved results by only +14%.

---

### When Inconsistency Manifests

Based on the research, inconsistency increases with:

| Factor | Low Risk | High Risk |
|--------|----------|-----------|
| **Task complexity** | Single-step, clear output | Multi-step, ambiguous requirements |
| **Agent count** | 1-2 agents | 4+ agents in sequence |
| **Context handoff** | Structured JSON | Natural language summaries |
| **Iteration loops** | None | Review/revise cycles |
| **Output length** | Short (<500 tokens) | Long (>2000 tokens) |
| **Session length** | Fresh session | Long conversation history |

**Specific failure scenarios in this design:**

1. **Coordinator oversight gaps**
   - Coordinator approves insufficient worker output
   - Misses blocker that should have been escalated
   - Scale: Depends on task complexity and Coordinator prompt quality

2. **Worker-Coordinator sync drift**
   - Worker output format deviates from expected JSON
   - Coordinator misinterprets worker status
   - Scale: ~10-15% variance in interpretation

3. **Phase gate inconsistency**
   - Same Analysis output approved Monday, rejected Tuesday
   - Causes unpredictable progression timing
   - Scale: Observed in code generation studies

4. **Context loss in handoffs**
   - Coder agent "forgets" key Analysis finding passed in prompt
   - Implements solution that ignores stated constraint
   - Scale: Increases with prompt length and number of handoffs

5. **Eval resource contention**
   - Multiple tasks compete for limited Eval hardware
   - Coordinator waits indefinitely or times out
   - Scale: Depends on HW availability and queue management

---

### Reliability Under Stress

[ReliabilityBench (arXiv:2601.06112)](https://arxiv.org/abs/2601.06112v1) tested agents under production conditions:

| Condition | Success Rate |
|-----------|--------------|
| Baseline (no stress) | 96.9% |
| With task perturbations (ε=0.2) | 88.1% |
| With API failures (rate limiting) | Significant degradation (most damaging) |

**Perturbations tested:** Semantically equivalent task rephrasing, timeout injection, partial responses, schema drift.

---

### Comparison: Prompt-Based vs Code-Based Orchestration

| Dimension | Prompt-Based (This Design) | Code-Based (Python) |
|-----------|---------------------------|---------------------|
| **Workflow determinism** | Low: Claude decides dynamically | High: Explicit control flow |
| **Agent spawn count** | Variable per run | Fixed by code |
| **Handoff consistency** | Depends on synthesis quality | Explicit data passing |
| **Failure recovery** | Claude's judgment | Your retry logic |
| **Loop termination** | Prompt-specified max | Code-enforced max |
| **Reproducibility** | Low (~50% token agreement) | High (same code = same flow) |
| **Auditability** | Reconstruct from logs | Deterministic trace |

---

### Mitigation Effectiveness

How well do this design's mitigations address the issues:

| Mitigation | Addresses | Residual Risk |
|------------|-----------|---------------|
| **Coordinator supervision** | Worker quality gaps | Coordinator itself may have oversight gaps |
| **Structured JSON Output** | Sync drift | JSON structure consistent, values may vary |
| **Workflow trace logging** | Debugging complexity | Log quality depends on agent compliance |
| **Phase gates** | Premature progression | Gate decisions still non-deterministic |
| **Memory Files** | Context loss across sessions | Within-session handoff loss remains |
| **HW availability checks** | Resource contention | Queue management is best-effort |

---

### Recommendations by Use Case

| Use Case | Recommendation |
|----------|----------------|
| **Prototyping, exploration** | This design is suitable; variance is acceptable |
| **Internal tooling** | Suitable with human review of outputs |
| **Customer-facing features** | Add validation layer or human approval |
| **High-stakes decisions** | Use code-based orchestration for determinism |
| **Regulated industries** | Code-based with audit logging required |
| **Batch processing** | Expect ~10-15% variance in outputs |

---

### Sources

- [Non-Determinism of "Deterministic" LLM Settings (arXiv:2408.04667)](https://arxiv.org/html/2408.04667v5)
- [Why Temperature=0 Doesn't Guarantee Determinism (Brenndoerfer)](https://mbrenndoerfer.com/writing/why-llms-are-not-deterministic)
- [Why Do Multi-Agent LLM Systems Fail? (arXiv:2503.13657)](https://arxiv.org/html/2503.13657v1)
- [ReliabilityBench: LLM Agent Reliability Under Stress (arXiv:2601.06112)](https://arxiv.org/abs/2601.06112v1)
- [Claude AI Code Generation Inconsistencies (WebProNews)](https://www.webpronews.com/claude-ais-code-generation-speed-gains-vs-2025-inconsistencies/)
- [Claude Sonnet 4.5 Benchmarks (Analytics Vidhya)](https://www.analyticsvidhya.com/blog/2025/09/claude-sonnet-4-5/)

---

## Alternative Approaches

### Single-Context Personas

One Claude instance role-plays multiple personas without spawning agents.

**Key Limitation:** All personas share one context window. Context limits and role bleeding occur as conversations grow.

**Use when:** Prototyping, simple workflows.

### Custom Python Orchestration

Build your own orchestration with direct API calls.

**Key Limitation:** Requires API access (pay-per-token). You build and maintain all infrastructure.

**Use when:** Production systems needing persistent memory across sessions, strict determinism, or custom tools beyond MCP.
