"""
Triton Kernel Generation Adapter for GEPA.

This adapter integrates GEPA with the triton-ag InferenceClient
to optimize prompts for Triton kernel generation tasks.

It uses KbEvalClient to evaluate generated Triton kernels for:
- Compilation success
- Correctness (numerical accuracy)
- Performance (runtime/speedup)
"""

import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

# Append /workspace to sys.path at the END (after site-packages)
# This ensures installed packages (like gepa) take precedence,
# while custom modules in /workspace are still importable.
if "/workspace" not in sys.path:
    sys.path.append("/workspace")

from gepa.core.adapter import EvaluationBatch, GEPAAdapter


def calculate_reward(
    compiled: bool,
    correctness: bool,
    speedup: float,
    speedup_threshold: float,
    compile_score: float = 0.15,
    correct_score: float = 0.3,
    speedup_score: float = 0.3,
) -> float:
    """
    Calculate a configurable reward score for Triton kernel evaluation.

    The reward is calculated in a tiered manner:
    1. If compilation fails: return 0
    2. If compilation succeeds but correctness fails: return compile_score
    3. If both compile and correctness pass: return correct_score + speedup bonus

    The speedup bonus is calculated as:
    - If speedup >= speedup_threshold: add speedup_score to the base correct_score

    Args:
        compiled: Whether the code compiled successfully
        correctness: Whether the code passed correctness tests
        speedup: The actual speedup achieved (reference_runtime / runtime)
        speedup_threshold: Minimum speedup required to earn the speedup bonus
        compile_score: Score for code that compiles but fails correctness (default: 0.15)
        correct_score: Base score for correct code (default: 0.3)
        speedup_score: Bonus score when speedup >= threshold (default: 0.3)

    Returns:
        Calculated reward score

    Example:
        >>> # Failed compilation
        >>> calculate_reward(False, False, 0.0, 1.0)
        0

        >>> # Compiled but incorrect
        >>> calculate_reward(True, False, 0.0, 1.0)
        0.15

        >>> # Correct but no speedup bonus (speedup < threshold)
        >>> calculate_reward(True, True, 0.8, 1.0)
        0.3

        >>> # Correct with speedup bonus (speedup >= threshold)
        >>> calculate_reward(True, True, 1.5, 1.0)
        0.6
    """
    if not compiled:
        return 0

    if not correctness:
        return compile_score

    # Base score for correct code
    score = correct_score

    # Add speedup bonus if threshold is met
    if speedup >= speedup_threshold:
        score += speedup_score

    return score


@dataclass
class TritonKernelDataInst:
    """Data instance for Triton kernel generation task."""

    task_id: str
    reference_code: str
    expected_output: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TritonKernelTrajectory:
    """Trajectory for Triton kernel generation."""

    data: TritonKernelDataInst
    system_prompt: str
    user_prompt: str
    generated_code: str
    compilation_result: Optional[str] = None
    correctness_result: Optional[str] = None
    speedup: Optional[float] = None
    eval_metadata: Optional[Dict[str, Any]] = None


@dataclass
class TritonKernelOutput:
    """Output from Triton kernel generation."""

    generated_code: str
    compilation_success: bool
    correctness_success: bool
    speedup: Optional[float] = None
    runtime: Optional[float] = None
    error_message: Optional[str] = None
    eval_metadata: Optional[Dict[str, Any]] = None


class TraceLogger:
    """Logger for tracking LLM responses, KbEval results, and execution traces.

    Directory structure:
    traces/
    ├── code_generation/
    │   ├── train/
    │   │   └── {task_id}/
    │   │       ├── iter_{iter}_candidate_{idx}_request.json
    │   │       ├── iter_{iter}_candidate_{idx}_response.json
    │   │       └── iter_{iter}_candidate_{idx}_code.py
    │   └── val/
    │       └── {task_id}/
    │           └── ...
    ├── reflection/
    │   ├── iter_{iter}_reflection_input.json
    │   ├── iter_{iter}_reflection_response.json
    │   └── iter_{iter}_new_prompt.txt
    ├── kbeval_results/
    │   ├── train/
    │   │   └── {task_id}/
    │   │       └── iter_{iter}_candidate_{idx}_kbeval.json
    │   └── val/
    │       └── {task_id}/
    │           └── iter_{iter}_candidate_{idx}_kbeval.json
    ├── candidates/
    │   └── candidate_{idx}.txt
    └── summary.log
    """

    def __init__(self, run_dir: Optional[str] = None, enabled: bool = True):
        self.enabled = enabled and run_dir is not None
        self.run_dir = Path(run_dir) if run_dir else None
        self.iteration = 0
        self.candidate_idx = 0
        self.eval_mode = "val"  # "train" or "val"

        if self.enabled and self.run_dir:
            # Create log directories
            self.traces_dir = self.run_dir / "traces"

            # Code generation directories
            self.code_gen_dir = self.traces_dir / "code_generation"
            self.code_gen_train_dir = self.code_gen_dir / "train"
            self.code_gen_val_dir = self.code_gen_dir / "val"

            # Reflection directories
            self.reflection_dir = self.traces_dir / "reflection"

            # KbEval results directories
            self.kbeval_dir = self.traces_dir / "kbeval_results"
            self.kbeval_train_dir = self.kbeval_dir / "train"
            self.kbeval_val_dir = self.kbeval_dir / "val"

            # Candidates directory
            self.candidates_dir = self.traces_dir / "candidates"

            for d in [
                self.traces_dir,
                self.code_gen_train_dir,
                self.code_gen_val_dir,
                self.reflection_dir,
                self.kbeval_train_dir,
                self.kbeval_val_dir,
                self.candidates_dir,
            ]:
                d.mkdir(parents=True, exist_ok=True)

            # Create summary log file
            self.summary_log = self.traces_dir / "summary.log"
            with open(self.summary_log, "w") as f:
                f.write(f"GEPA Triton Kernel Optimization Trace Log\n")
                f.write(f"Started: {datetime.now().isoformat()}\n")
                f.write("=" * 60 + "\n\n")

    def _write_summary(self, message: str):
        """Append to summary log."""
        if not self.enabled:
            return
        with open(self.summary_log, "a") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")

    def set_iteration(self, iteration: int):
        """Set current iteration number."""
        self.iteration = iteration

    def set_candidate_idx(self, candidate_idx: int):
        """Set current candidate index."""
        self.candidate_idx = candidate_idx

    def set_eval_mode(self, mode: str):
        """Set evaluation mode: 'train' or 'val'."""
        assert mode in ("train", "val"), f"Invalid eval mode: {mode}"
        self.eval_mode = mode

    def _get_code_gen_task_dir(self, task_id: str) -> Path:
        """Get code generation directory for a task."""
        base_dir = self.code_gen_train_dir if self.eval_mode == "train" else self.code_gen_val_dir
        task_dir = base_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir

    def _get_kbeval_task_dir(self, task_id: str) -> Path:
        """Get kbeval results directory for a task."""
        base_dir = self.kbeval_train_dir if self.eval_mode == "train" else self.kbeval_val_dir
        task_dir = base_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir

    def log_llm_request(
        self,
        task_id: str,
        system_prompt: str,
        user_prompt: str,
        candidate_idx: Optional[int] = None,
    ):
        """Log LLM code generation request (prompts)."""
        if not self.enabled:
            return

        idx = candidate_idx if candidate_idx is not None else self.candidate_idx
        task_dir = self._get_code_gen_task_dir(task_id)
        filename = f"iter_{self.iteration:03d}_candidate_{idx:03d}_request.json"
        filepath = task_dir / filename

        data = {
            "iteration": self.iteration,
            "candidate_idx": idx,
            "task_id": task_id,
            "eval_mode": self.eval_mode,
            "timestamp": datetime.now().isoformat(),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        self._write_summary(f"[{self.eval_mode}] LLM Request: {task_id} (iter={self.iteration}, candidate={idx})")

    def log_llm_response(
        self,
        task_id: str,
        raw_response: str,
        extracted_code: str,
        candidate_idx: Optional[int] = None,
    ):
        """Log LLM code generation response."""
        if not self.enabled:
            return

        idx = candidate_idx if candidate_idx is not None else self.candidate_idx
        task_dir = self._get_code_gen_task_dir(task_id)

        # Save response JSON
        filename = f"iter_{self.iteration:03d}_candidate_{idx:03d}_response.json"
        filepath = task_dir / filename

        data = {
            "iteration": self.iteration,
            "candidate_idx": idx,
            "task_id": task_id,
            "eval_mode": self.eval_mode,
            "timestamp": datetime.now().isoformat(),
            "raw_response": raw_response,
            "extracted_code": extracted_code,
            "code_length": len(extracted_code),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        # Also save the raw code as a .py file for easy viewing
        code_filename = f"iter_{self.iteration:03d}_candidate_{idx:03d}_code.py"
        code_filepath = task_dir / code_filename
        with open(code_filepath, "w") as f:
            f.write(f"# Task: {task_id}\n")
            f.write(f"# Iteration: {self.iteration}\n")
            f.write(f"# Candidate: {idx}\n")
            f.write(f"# Eval Mode: {self.eval_mode}\n")
            f.write(f"# Timestamp: {datetime.now().isoformat()}\n\n")
            f.write(extracted_code)

        self._write_summary(
            f"[{self.eval_mode}] LLM Response: {task_id} (code length: {len(extracted_code)})"
        )

    def log_kbeval_result(
        self,
        task_id: str,
        generated_code: str,
        result: Dict[str, Any],
        score: float,
        feedback: str,
    ):
        """Log KbEval evaluation result."""
        if not self.enabled:
            return

        task_dir = self._get_kbeval_task_dir(task_id)
        filename = f"iter_{self.iteration:03d}_candidate_{self.candidate_idx:03d}_kbeval.json"
        filepath = task_dir / filename

        data = {
            "iteration": self.iteration,
            "candidate_idx": self.candidate_idx,
            "task_id": task_id,
            "eval_mode": self.eval_mode,
            "timestamp": datetime.now().isoformat(),
            "score": score,
            "feedback": feedback,
            "kbeval_result": result,
            "generated_code_preview": generated_code[:500]
            + ("..." if len(generated_code) > 500 else ""),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        # Determine status emoji
        compiled = result.get("compiled", False) if result else False
        correct = result.get("correctness", False) if result else False
        if correct:
            status = "✅ SUCCESS"
        elif compiled:
            status = "⚠️ COMPILED (incorrect)"
        else:
            status = "❌ FAILED"

        self._write_summary(f"[{self.eval_mode}] KbEval: {task_id} - {status} (score: {score:.3f})")

    def log_reflection_input(
        self,
        iteration: int,
        candidate: Dict[str, str],
        feedback_data: Dict[str, Any],
    ):
        """Log reflection LLM input."""
        if not self.enabled:
            return

        filename = f"iter_{iteration:03d}_reflection_input.json"
        filepath = self.reflection_dir / filename

        data = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "current_prompt": candidate.get("system_prompt", ""),
            "feedback_data": feedback_data,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        self._write_summary(f"Reflection Input: iter={iteration}")

    def log_reflection_response(
        self,
        iteration: int,
        raw_response: str,
        new_prompt: str,
    ):
        """Log reflection LLM response and new prompt."""
        if not self.enabled:
            return

        # Save response JSON
        filename = f"iter_{iteration:03d}_reflection_response.json"
        filepath = self.reflection_dir / filename

        data = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "raw_response": raw_response,
            "new_prompt": new_prompt,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        # Also save the new prompt as a text file
        prompt_filename = f"iter_{iteration:03d}_new_prompt.txt"
        prompt_filepath = self.reflection_dir / prompt_filename
        with open(prompt_filepath, "w") as f:
            f.write(f"# Iteration: {iteration}\n")
            f.write(f"# Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n\n")
            f.write(new_prompt)

        self._write_summary(f"Reflection Response: iter={iteration} (prompt length: {len(new_prompt)})")

    def log_evaluation_batch(
        self,
        candidate: Dict[str, str],
        batch_size: int,
        scores: List[float],
        outputs: List[Any],
    ):
        """Log a complete evaluation batch summary."""
        if not self.enabled:
            return

        avg_score = sum(scores) / len(scores) if scores else 0.0
        success_count = sum(
            1 for o in outputs if getattr(o, "correctness_success", False)
        )

        self._write_summary(
            f"[{self.eval_mode}] Batch Eval: {batch_size} tasks, avg={avg_score:.3f}, "
            f"success={success_count}/{batch_size}"
        )

    def log_candidate_prompt(self, candidate_idx: int, candidate: Dict[str, str]):
        """Log a candidate prompt being evaluated."""
        if not self.enabled:
            return

        filename = f"candidate_{candidate_idx:03d}.txt"
        filepath = self.candidates_dir / filename

        with open(filepath, "w") as f:
            f.write(f"Candidate Index: {candidate_idx}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n\n")
            for key, value in candidate.items():
                f.write(f"[{key}]\n")
                f.write("-" * 40 + "\n")
                f.write(value + "\n\n")

    def finalize(self, best_score: float, best_candidate: Dict[str, str]):
        """Write final summary."""
        if not self.enabled:
            return

        self._write_summary("\n" + "=" * 60)
        self._write_summary("OPTIMIZATION COMPLETE")
        self._write_summary("=" * 60)
        self._write_summary(f"Total iterations: {self.iteration}")
        self._write_summary(f"Best score: {best_score:.4f}")

        # Save best prompt
        # Note: best_candidate may contain modular components instead of system_prompt
        # The main script now handles assembling the system_prompt, so we just save
        # whatever is in the candidate (either system_prompt or components)
        best_prompt_file = self.traces_dir / "best_prompt.txt"
        with open(best_prompt_file, "w") as f:
            f.write("Best System Prompt\n")
            f.write("=" * 60 + "\n\n")
            # If system_prompt key exists, use it; otherwise write all components
            if "system_prompt" in best_candidate:
                f.write(best_candidate.get("system_prompt", ""))
            else:
                # Write all components
                for key, value in best_candidate.items():
                    f.write(f"[{key}]\n")
                    f.write("-" * 40 + "\n")
                    f.write(value + "\n\n")


class TritonKernelAdapter(GEPAAdapter):
    """
    GEPA Adapter for Triton kernel generation optimization.

    This adapter:
    1. Uses InferenceClient to call your vLLM-hosted models
    2. Uses KbEvalClient to evaluate generated Triton kernels
    3. Provides rich feedback for prompt evolution
    4. Logs all traces to run_dir for examination
    5. Supports modular prompt components (role_description, task_description, etc.)
       where only optimizable components are evolved by GEPA
    """

    def __init__(
        self,
        model_name: str = "qwen3-32b",
        provider_name: str = "h100_8_5_a",
        config_file: str = "inferenceClient.yaml",
        max_tokens: int = 8192,
        # KbEvalClient settings
        eval_provider_name: str = "h8_4",
        eval_config_file: str = "kbEval.yaml",
        use_kb_eval: bool = True,
        # Reward function parameters (read from config)
        compile_score: float = 0.15,
        correct_score: float = 0.3,
        speedup_threshold: float = 1.0,
        speedup_score: float = 0.3,
        # Reference runtimes (pre-generated)
        reference_runtimes: Optional[Dict[str, Dict[str, Any]]] = None,
        # Tags for tracking
        run_tag: str = "gepa_optimization",
        model_tag: str = "gepa",
        # Logging settings
        run_dir: Optional[str] = None,
        enable_tracing: bool = True,
        # User prompt template (from triton.prompt.yaml)
        user_prompt_template: Optional[str] = None,
        # Concurrency settings
        max_concurrent_llm: int = 8,
        max_concurrent_eval: int = 4,
        # Modular prompt settings
        fixed_components: Optional[Dict[str, str]] = None,
        component_order: Optional[List[str]] = None,
        component_metadata: Optional[Dict[str, Dict[str, str]]] = None,
        reflection_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Triton Kernel Adapter.

        Args:
            model_name: Model name from inferenceClient.yaml
            provider_name: Provider name from inferenceClient.yaml
            config_file: Path to inferenceClient.yaml
            max_tokens: Maximum tokens for generation
            eval_provider_name: Provider for KbEvalClient
            eval_config_file: Path to kbEval.yaml
            use_kb_eval: Whether to use KbEvalClient for evaluation
            compile_score: Score for code that compiles but fails correctness (default: 0.15)
            correct_score: Base score for correct code (default: 0.3)
            speedup_threshold: Minimum speedup required to earn bonus (default: 1.0)
            speedup_score: Bonus score when speedup >= threshold (default: 0.3)
            reference_runtimes: Dict mapping task_id to pre-generated reference runtime results.
                                Each entry should have 'runtime' key with the reference runtime in μs.
                                Example: {"task_id": {"runtime": 45.23, "compiled": True, ...}}
            run_tag: Tag for tracking evaluation runs
            model_tag: Tag for tracking model
            run_dir: Directory to save traces (LLM responses, KbEval results, etc.)
            enable_tracing: Whether to enable trace logging
            user_prompt_template: Optional user prompt template with {reference_code} placeholder.
                                  If provided, uses this template instead of the default.
                                  Example: "Following is the reference code:\n```python\n{reference_code}\n```"
            max_concurrent_llm: Maximum concurrent LLM generation requests (default: 8)
            max_concurrent_eval: Maximum concurrent KbEval evaluation requests (default: 4)
            fixed_components: Dict of non-optimizable prompt components that are kept constant.
                              Example: {"examples": "Example code..."}
            component_order: List of component names in the order they should be assembled.
                             Example: ["role_description", "task_description", "task_instruction", "examples"]
        """
        self.model_name = model_name
        self.provider_name = provider_name
        self.config_file = config_file
        self.max_tokens = max_tokens

        # KbEvalClient settings
        self.eval_provider_name = eval_provider_name
        self.eval_config_file = eval_config_file
        self.use_kb_eval = use_kb_eval

        # Reward function parameters
        self.compile_score = compile_score
        self.correct_score = correct_score
        self.speedup_threshold = speedup_threshold
        self.speedup_score = speedup_score

        # Reference runtimes (pre-generated)
        self.reference_runtimes = reference_runtimes or {}

        # Tags
        self.run_tag = run_tag
        self.model_tag = model_tag

        # Logging
        self.run_dir = run_dir
        self.enable_tracing = enable_tracing
        self.trace_logger = TraceLogger(run_dir=run_dir, enabled=enable_tracing)
        self._candidate_idx = 0

        # User prompt template (from triton.prompt.yaml user_prompt.init)
        self.user_prompt_template = user_prompt_template

        # Concurrency settings
        self.max_concurrent_llm = max_concurrent_llm
        self.max_concurrent_eval = max_concurrent_eval

        # Modular prompt settings
        # fixed_components: components that are NOT evolved by GEPA
        # component_order: order to assemble the full system_prompt
        self.fixed_components = fixed_components or {}
        self.component_order = component_order or ["role_description", "task_description", "task_instruction", "examples"]
        self.component_metadata = component_metadata or {}
        self.reflection_config = reflection_config or {}

        # Reflection LM (set via set_reflection_lm)
        self._reflection_lm = None

        # Lazy load clients
        self._inference_client = None
        self._eval_client = None

    def _assemble_system_prompt(self, candidate: Dict[str, str]) -> str:
        """
        Assemble the full system_prompt from modular components.

        The candidate dict contains the optimizable components from GEPA.
        The fixed_components dict contains the non-optimizable components.
        Components are assembled in the order specified by component_order.

        Args:
            candidate: Dict of optimizable components from GEPA.
                       Can have individual keys like "role_description", "task_description", etc.
                       Or a single "system_prompt" key for backward compatibility.

        Returns:
            The assembled system_prompt string.
        """
        # Check if candidate has a single "system_prompt" key (backward compatibility)
        if "system_prompt" in candidate and len(candidate) == 1:
            return candidate["system_prompt"]

        # Assemble from modular components
        parts = []
        for component_name in self.component_order:
            # Try optimizable components first, then fixed components
            if component_name in candidate:
                parts.append(candidate[component_name])
            elif component_name in self.fixed_components:
                parts.append(self.fixed_components[component_name])

        return "\n".join(parts)

    @property
    def inference_client(self):
        """Lazy load the InferenceClient."""
        if self._inference_client is None:
            from inferenceClient import InferenceClient

            self._inference_client = InferenceClient(
                model_name=self.model_name,
                config_file=self.config_file,
            )
        return self._inference_client

    @property
    def eval_client(self):
        """Lazy load the KbEvalClient."""
        if self._eval_client is None and self.use_kb_eval:
            from kbEvalClient import KbEvalClient

            self._eval_client = KbEvalClient(
                config_file=self.eval_config_file,
            )
        return self._eval_client

    def _build_user_prompt(self, data: TritonKernelDataInst) -> str:
        """Build user prompt from data instance.

        Uses the custom user_prompt_template if provided (from triton.prompt.yaml user_prompt.init),
        otherwise falls back to the default template.
        """
        if self.user_prompt_template:
            # Use custom template from config
            prompt = self.user_prompt_template.format(
                reference_code=data.reference_code,
            )
        else:
            # Default template
            prompt = f"""Following is the reference PyTorch code, implement the complete new module with Triton (no testing code, no init code).

Reference code:
```python
{data.reference_code}
```

Be concise with your thinking. Please limit your reasoning and thinking within 600 words for each complete code writing."""
        return prompt

    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Try to find code blocks
        code_pattern = r"```(?:python)?\s*(.*?)```"
        matches = re.findall(code_pattern, response, re.DOTALL)

        if matches:
            # Return the last code block (usually the final answer)
            return matches[-1].strip()

        # If no code blocks, return the whole response
        return response.strip()

    def _format_kbeval_feedback(self, result: Dict[str, Any]) -> str:
        """
        Format the full kbeval result as a feedback string.

        Args:
            result: The full kbeval result dict

        Returns:
            Formatted feedback string with all relevant information
        """
        lines = []

        # Status header
        compiled = result.get("compiled", False)
        correctness = result.get("correctness", False)

        if correctness:
            lines.append("STATUS: SUCCESS")
        elif compiled:
            lines.append("STATUS: CORRECTNESS FAILED")
        else:
            lines.append("STATUS: COMPILATION FAILED")

        # Basic metrics
        lines.append(f"compiled: {compiled}")
        lines.append(f"correctness: {correctness}")

        runtime = result.get("runtime", -1.0)
        if runtime > 0:
            lines.append(f"runtime: {runtime}")

        ref_runtime = result.get("reference_runtime")
        if ref_runtime is not None:
            lines.append(f"reference_runtime: {ref_runtime}")

        speedup = result.get("speedup", 0.0)
        if speedup > 0:
            lines.append(f"speedup: {speedup}")

        # Metadata section
        metadata = result.get("metadata", {})
        if metadata:
            lines.append("")
            lines.append("metadata:")
            for key, value in metadata.items():
                if value:  # Only include non-empty values
                    lines.append(f"  {key}: {value}")

        # Runtime stats section
        runtime_stats = result.get("runtime_stats", {})
        if runtime_stats:
            lines.append("")
            lines.append("runtime_stats:")
            for key, value in runtime_stats.items():
                if value is not None:
                    lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    async def _generate_async(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate code using InferenceClient asynchronously."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        result = await self.inference_client.chat_completion(
            provider=self.provider_name,
            messages=messages,
            max_tokens=self.max_tokens,
        )

        if result is None:
            return ""

        content = result.get("content", "")
        reasoning = result.get("reasoning_content", "")

        # Combine reasoning and content if both present
        if reasoning and content:
            return f"{reasoning}\n\n{content}"
        return content or reasoning or ""

    async def _evaluate_with_kb_eval(
        self,
        data: TritonKernelDataInst,
        generated_code: str,
    ) -> tuple:
        """
        Evaluate generated Triton code using KbEvalClient.

        Returns:
            (score, output, feedback)
        """
        if not generated_code:
            return (
                0,
                TritonKernelOutput(
                    generated_code="",
                    compilation_success=False,
                    correctness_success=False,
                    error_message="Empty generation",
                ),
                "FAILED: Empty code generation. The model did not produce any code.",
            )

        # Call KbEvalClient with code_type="triton"
        result = await self.eval_client.kb_eval(
            provider=self.eval_provider_name,
            reference_code=data.reference_code,
            generated_code=generated_code,
            run_tag=self.run_tag,
            model_tag=self.model_tag,
            task_tag=data.task_id,
            eval_tag="gepa_eval",
            code_type="triton",  # KEY DIFFERENCE: Use "triton" instead of "cuda"
        )

        if result is None:
            return (
                0,
                TritonKernelOutput(
                    generated_code=generated_code,
                    compilation_success=False,
                    correctness_success=False,
                    error_message="Evaluation service unavailable",
                ),
                "FAILED: Could not reach evaluation service. Please check the kbEval server.",
            )

        # Parse results
        compiled = result.get("compiled", False)
        correctness = result.get("correctness", False)
        runtime = result.get("runtime", -1.0)
        metadata = result.get("metadata", {})
        runtime_stats = result.get("runtime_stats", {})

        # Get reference runtime from pre-loaded data or fallback to result
        ref_runtime = None
        if data.task_id in self.reference_runtimes:
            ref_data = self.reference_runtimes[data.task_id]
            ref_runtime = ref_data.get("runtime", -1.0)
            if ref_runtime <= 0:
                ref_runtime = None

        # Fallback to result's reference_runtime if pre-loaded not available
        if ref_runtime is None:
            ref_runtime = result.get("reference_runtime")

        # Calculate speedup
        speedup = 0.0
        if runtime > 0 and ref_runtime is not None and ref_runtime > 0:
            speedup = ref_runtime / runtime

        # Add reference_runtime and speedup to result for feedback
        result["reference_runtime"] = ref_runtime
        result["speedup"] = speedup

        # Calculate score using the configurable reward function
        score = calculate_reward(
            compiled=compiled,
            correctness=correctness,
            speedup=speedup,
            speedup_threshold=self.speedup_threshold,
            compile_score=self.compile_score,
            correct_score=self.correct_score,
            speedup_score=self.speedup_score,
        )

        # Build feedback directly from kbeval result
        feedback = self._format_kbeval_feedback(result)

        output = TritonKernelOutput(
            generated_code=generated_code,
            compilation_success=compiled,
            correctness_success=correctness,
            speedup=speedup if speedup > 0 else None,
            runtime=runtime if runtime > 0 else None,
            error_message=metadata.get("error") if not correctness else None,
            eval_metadata=result,  # Store the full KB eval result including metadata
        )

        return score, output, feedback

    def _evaluate_code_simple(
        self,
        data: TritonKernelDataInst,
        generated_code: str,
    ) -> tuple:
        """
        Simple evaluation without KbEvalClient (fallback).

        Returns:
            (score, output, feedback)
        """
        if not generated_code:
            return (
                self.failure_score,
                TritonKernelOutput(
                    generated_code="",
                    compilation_success=False,
                    correctness_success=False,
                    error_message="Empty generation",
                ),
                "FAILED: Empty code generation",
            )

        # Simple heuristic checks for Triton code
        has_model_new = "class ModelNew" in generated_code
        has_triton_kernel = "@triton.jit" in generated_code
        has_triton_import = "import triton" in generated_code

        if has_model_new and has_triton_kernel and has_triton_import:
            score = self.correct_score
            feedback = (
                "LIKELY CORRECT: Code contains ModelNew class, "
                "Triton kernel (@triton.jit), and triton import. "
                "(Note: Not verified by actual compilation/execution)"
            )
            output = TritonKernelOutput(
                generated_code=generated_code,
                compilation_success=True,
                correctness_success=True,
            )
        elif has_model_new and (has_triton_kernel or has_triton_import):
            score = self.compile_only_score
            missing = []
            if not has_triton_kernel:
                missing.append("Triton kernel (@triton.jit)")
            if not has_triton_import:
                missing.append("triton import")
            feedback = (
                f"PARTIAL: Code has ModelNew but missing: {', '.join(missing)}. "
                "Ensure you include both a Triton kernel and triton import."
            )
            output = TritonKernelOutput(
                generated_code=generated_code,
                compilation_success=False,
                correctness_success=False,
                error_message=f"Missing: {', '.join(missing)}",
            )
        else:
            score = self.failure_score
            feedback = (
                "FAILED: Code is missing required components.\n"
                "Required:\n"
                "1. class ModelNew - the optimized model class\n"
                "2. @triton.jit - Triton kernel decorator\n"
                "3. import triton - Triton library import"
            )
            output = TritonKernelOutput(
                generated_code=generated_code,
                compilation_success=False,
                correctness_success=False,
                error_message="Missing ModelNew class",
            )

        return score, output, feedback

    def evaluate(
        self,
        batch: List[TritonKernelDataInst],
        candidate: Dict[str, str],
        capture_traces: bool = False,
        eval_mode: Optional[str] = None,
    ) -> EvaluationBatch:
        """
        Evaluate a batch of inputs with the given prompt candidate.

        Args:
            batch: List of TritonKernelDataInst
            candidate: Dict with prompt components (either modular components like
                       'role_description', 'task_description', etc. or a single
                       'system_prompt' key for backward compatibility)
            capture_traces: Whether to capture execution traces
            eval_mode: Evaluation mode - "train" or "val". If None, auto-detect
                       based on whether capture_traces is True (train) or False (val).

        Returns:
            EvaluationBatch with outputs, scores, and optionally traces
        """
        # Assemble the full system_prompt from modular components
        system_prompt = self._assemble_system_prompt(candidate)

        # Auto-detect eval_mode if not provided
        # GEPA calls with capture_traces=True for train, False for val
        if eval_mode is None:
            eval_mode = "train" if capture_traces else "val"

        # Set eval mode for trace logger
        self.trace_logger.set_eval_mode(eval_mode)

        # Log candidate prompt
        self._candidate_idx += 1
        self.trace_logger.set_candidate_idx(self._candidate_idx)
        self.trace_logger.log_candidate_prompt(self._candidate_idx, candidate)

        outputs: List[TritonKernelOutput] = []
        scores: List[float] = []
        trajectories: List[TritonKernelTrajectory] = [] if capture_traces else None

        async def process_batch():
            """
            Process batch with pipeline parallelism and semaphore-based rate limiting.

            Instead of running all LLM calls first, then all evals:
            - Each task runs LLM generation -> evaluation as a pipeline
            - All task pipelines run concurrently
            - Semaphores limit concurrent LLM and eval requests separately

            Timeline visualization:
                Task 1: [LLM ----] [Eval ----]
                Task 2:   [LLM ----] [Eval ----]
                Task 3:     [LLM ----] [Eval ----]
                              ↑ Eval starts immediately when LLM finishes
            """
            # Create semaphores for rate limiting
            llm_semaphore = asyncio.Semaphore(self.max_concurrent_llm)
            eval_semaphore = asyncio.Semaphore(self.max_concurrent_eval)

            async def process_single_task(data: TritonKernelDataInst):
                """
                Process a single task through the full pipeline:
                LLM generation -> code extraction -> KbEval evaluation
                """
                user_prompt = self._build_user_prompt(data)

                # Log LLM request
                self.trace_logger.log_llm_request(
                    task_id=data.task_id,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    candidate_idx=self._candidate_idx,
                )

                # Step 1: LLM Generation (with concurrency limit)
                async with llm_semaphore:
                    try:
                        response = await self._generate_async(system_prompt, user_prompt)
                    except Exception as e:
                        print(f"Error generating code for {data.task_id}: {e}")
                        response = ""

                # Extract code from response
                generated_code = self._extract_code(response)

                # Log LLM response
                self.trace_logger.log_llm_response(
                    task_id=data.task_id,
                    raw_response=response if isinstance(response, str) else "",
                    extracted_code=generated_code,
                    candidate_idx=self._candidate_idx,
                )

                # Step 2: KbEval Evaluation (with separate concurrency limit)
                # This starts immediately after LLM generation completes for this task
                async with eval_semaphore:
                    try:
                        if self.use_kb_eval and self.eval_client is not None:
                            score, output, feedback = await self._evaluate_with_kb_eval(
                                data, generated_code
                            )
                        else:
                            score, output, feedback = self._evaluate_code_simple(
                                data, generated_code
                            )
                    except Exception as e:
                        print(f"Error evaluating code for {data.task_id}: {e}")
                        score = 0
                        output = TritonKernelOutput(
                            generated_code=generated_code,
                            compilation_success=False,
                            correctness_success=False,
                            error_message=str(e),
                        )
                        feedback = f"FAILED: Evaluation error - {e}"

                # Log KbEval result
                kbeval_result = (
                    output.eval_metadata
                    if output.eval_metadata
                    else {
                        "compiled": output.compilation_success,
                        "correctness": output.correctness_success,
                        "speedup": output.speedup,
                        "runtime": output.runtime,
                        "error": output.error_message,
                    }
                )
                self.trace_logger.log_kbeval_result(
                    task_id=data.task_id,
                    generated_code=generated_code,
                    result=kbeval_result,
                    score=score,
                    feedback=feedback,
                )

                return (data, user_prompt, generated_code, score, output, feedback)

            # Run all task pipelines concurrently
            tasks = [process_single_task(data) for data in batch]
            results_raw = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and collect results
            results = []
            for result in results_raw:
                if isinstance(result, Exception):
                    print(f"Task pipeline error: {result}")
                    continue
                results.append(result)

            return results

        # Run the async batch
        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(process_batch())
        finally:
            loop.close()

        # Collect results
        for data, user_prompt, generated_code, score, output, feedback in results:
            outputs.append(output)
            scores.append(score)

            if trajectories is not None:
                trajectories.append(
                    TritonKernelTrajectory(
                        data=data,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        generated_code=generated_code,
                        compilation_result=feedback,
                        correctness_result=feedback,
                        speedup=output.speedup,
                        eval_metadata=output.eval_metadata,
                    )
                )

        # Log batch summary
        self.trace_logger.log_evaluation_batch(
            candidate=candidate,
            batch_size=len(batch),
            scores=scores,
            outputs=outputs,
        )

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: List[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """
        Build reflective dataset for prompt evolution.

        This provides rich feedback to the reflection LM to improve prompts.
        """
        result: Dict[str, List[Dict[str, Any]]] = {}

        for component in components_to_update:
            items: List[Dict[str, Any]] = []

            if eval_batch.trajectories is None:
                continue

            for traj, score, output in zip(
                eval_batch.trajectories,
                eval_batch.scores,
                eval_batch.outputs,
            ):
                # Use kbeval result directly as feedback (cleaner and more informative)
                if output.eval_metadata:
                    feedback = json.dumps(output.eval_metadata, indent=2, default=str)
                    # Highlight speedup with different levels
                    speedup = output.eval_metadata.get("speedup", 0)
                    if speedup and speedup > 0:
                        if speedup >= 1.5:
                            feedback = f"🚀🚀 SUPER SPEEDUP: {speedup:.2f}x\n\n{feedback}"
                        elif speedup >= 1.0:
                            feedback = f"🚀 SPEEDUP: {speedup:.2f}x\n\n{feedback}"
                        else:
                            feedback = f"🐢 SLOWER THAN REFERENCE: {speedup:.2f}x \n\n{feedback}"
                else:
                    # Fallback if no eval_metadata
                    fallback_data = {
                        "compiled": output.compilation_success,
                        "correctness": output.correctness_success,
                        "runtime": output.runtime,
                        "speedup": output.speedup,
                        "error": output.error_message,
                    }
                    feedback = json.dumps(fallback_data, indent=2, default=str)
                    # Highlight speedup with different levels
                    if output.speedup and output.speedup > 0:
                        if output.speedup >= 1.5:
                            feedback = f"🚀🚀 SUPER SPEEDUP: {output.speedup:.2f}x\n\n{feedback}"
                        elif output.speedup >= 1.0:
                            feedback = f"🚀 SPEEDUP: {output.speedup:.2f}x\n\n{feedback}"
                        else:
                            feedback = f"🐢 SLOWER THAN REFERENCE: {output.speedup:.2f}x \n\n{feedback}"

                # Truncate code for reflection
                ref_code = traj.data.reference_code
                gen_code = traj.generated_code
                max_code_len = 1500

                if len(ref_code) > max_code_len:
                    ref_code = ref_code[:max_code_len] + "\n... (truncated)"
                if len(gen_code) > max_code_len:
                    gen_code = gen_code[:max_code_len] + "\n... (truncated)"

                item = {
                    "Inputs": f"Task: {traj.data.task_id}\n\nReference code:\n{ref_code}",
                    "Generated Outputs": gen_code,
                    "Feedback": feedback,
                    "Score": score,
                }
                items.append(item)

            result[component] = items

        return result

    def set_reflection_lm(self, reflection_lm):
        """Set the reflection LM callable for custom propose_new_texts."""
        self._reflection_lm = reflection_lm

    def propose_new_texts(
        self,
        candidate: Dict[str, str],
        reflective_dataset: Dict[str, List[Dict[str, Any]]],
        components_to_update: List[str],
    ) -> Dict[str, str]:
        """
        Override GEPA's default propose_new_texts with custom component-aware reflection.

        Uses the configurable reflection prompt template from the yaml config.
        """
        if self._reflection_lm is None:
            raise ValueError("Reflection LM not set. Call set_reflection_lm() first.")

        new_texts = {}

        for component_name in components_to_update:
            if component_name not in candidate:
                continue

            current_text = candidate[component_name]

            # Get component metadata
            metadata = self.component_metadata.get(component_name, {})
            label = metadata.get("label", component_name.upper().replace("_", " "))
            description = metadata.get("description", "")

            # Get feedback items for this component
            feedback_items = reflective_dataset.get(component_name, [])
            formatted_feedback = self._format_reflection_feedback(feedback_items)

            # Build the reflection prompt using configurable template
            reflection_prompt = self._build_reflection_prompt(
                component_name=component_name,
                label=label,
                description=description,
                current_text=current_text,
                formatted_feedback=formatted_feedback,
            )

            # Call reflection LM
            response = self._reflection_lm(reflection_prompt)

            # Extract improved text from response
            improved_text = self._extract_component_from_response(response, label)
            new_texts[component_name] = improved_text

        return new_texts

    def _build_reflection_prompt(
        self,
        component_name: str,
        label: str,
        description: str,
        current_text: str,
        formatted_feedback: str,
    ) -> str:
        """Build the reflection prompt using configurable template from yaml."""
        config = self.reflection_config
        template = config.get("template", self._get_default_template())
        sections = config.get("sections", {})
        components_overview = config.get("components_overview", self._get_default_components_overview())

        # Variable substitutions for all sections
        variables = {
            "component_name": component_name,
            "component_label": label,
            "component_purpose": description,
            "current_content": current_text,
            "evaluation_results": formatted_feedback,
            "components_overview": components_overview,
        }

        # First, substitute variables in each section
        processed_sections = {}
        for section_name, section_content in sections.items():
            processed_sections[section_name] = self._substitute_variables(section_content, variables)

        # Then, substitute sections in template
        result = template
        for section_name, section_content in processed_sections.items():
            placeholder = "{section:" + section_name + "}"
            result = result.replace(placeholder, section_content)

        # Substitute any remaining variables in the template
        result = self._substitute_variables(result, variables)

        return result

    def _substitute_variables(self, text: str, variables: Dict[str, str]) -> str:
        """Substitute {variable} placeholders with values."""
        result = text
        for var_name, var_value in variables.items():
            placeholder = "{" + var_name + "}"
            result = result.replace(placeholder, str(var_value))
        return result

    def _format_reflection_feedback(self, feedback_items: List[Dict[str, Any]]) -> str:
        """Format feedback items using configurable template."""
        if not feedback_items:
            return self.reflection_config.get("no_results_message", "No evaluation results available.")

        config = self.reflection_config
        example_template = config.get("example_template", self._get_default_example_template())
        separator = config.get("example_separator", "\n---\n")
        max_content_length = config.get("max_content_length", 800)

        formatted_examples = []
        for idx, item in enumerate(feedback_items, 1):
            inputs = item.get("Inputs", "")
            outputs = item.get("Generated Outputs", "")
            feedback = item.get("Feedback", "")
            score = item.get("Score", 0.0)

            # Truncate if needed
            if len(inputs) > max_content_length:
                inputs = inputs[:max_content_length] + "... (truncated)"
            if len(outputs) > max_content_length:
                outputs = outputs[:max_content_length] + "... (truncated)"

            # Format using template
            example = example_template
            example = example.replace("{index}", str(idx))
            example = example.replace("{score}", f"{score:.3f}")
            example = example.replace("{inputs}", inputs)
            example = example.replace("{outputs}", outputs)
            example = example.replace("{feedback}", feedback)

            formatted_examples.append(example)

        return separator.join(formatted_examples)

    def _extract_component_from_response(self, response: str, label: str) -> str:
        """Extract improved component text from reflection LM response."""
        # First, strip <think>...</think> tags (used by some LLMs for reasoning)
        cleaned_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

        # Try to find content after specific markers
        markers = [
            f"## Improved {label}:",
            f"## {label}:",
            f"**{label}:**",
            "```",
        ]

        for marker in markers:
            if marker in cleaned_response:
                parts = cleaned_response.split(marker, 1)
                if len(parts) > 1:
                    extracted = parts[1].strip()
                    # If it's a code block, extract content
                    if marker == "```" and "```" in extracted:
                        extracted = extracted.split("```")[0].strip()
                    return extracted

        # No marker found, return trimmed response
        return cleaned_response

    def _get_default_template(self) -> str:
        """Default reflection prompt template."""
        return """{section:intro}

{section:component_context}

{section:current_content}

{section:evaluation_results}

{section:task}

{section:output_format}"""

    def _get_default_example_template(self) -> str:
        """Default template for formatting feedback examples."""
        return """### Example {index} (Score: {score})

**Input:**
{inputs}

**Generated Output:**
{outputs}

**Feedback:**
{feedback}"""

    def _get_default_components_overview(self) -> str:
        """Default overview of system prompt components."""
        return """This component is ONE PART of a larger system prompt. The full system prompt is assembled from multiple components:
1. ROLE DESCRIPTION - Defines the AI's persona and expertise
2. TASK DESCRIPTION - Describes the overall task and objective
3. TASK INSTRUCTIONS - Provides specific guidelines and constraints
4. EXAMPLES - Shows example input/output pairs (not being optimized)"""


def create_triton_kernel_adapter(
    model_name: str = "qwen3-32b",
    provider_name: str = "h100_8_5_a",
    config_file: str = "inferenceClient.yaml",
    eval_provider_name: str = "local",
    eval_config_file: str = "kbEval.yaml",
    use_kb_eval: bool = True,
    **kwargs,
) -> TritonKernelAdapter:
    """Factory function to create a TritonKernelAdapter."""
    return TritonKernelAdapter(
        model_name=model_name,
        provider_name=provider_name,
        config_file=config_file,
        eval_provider_name=eval_provider_name,
        eval_config_file=eval_config_file,
        use_kb_eval=use_kb_eval,
        **kwargs,
    )
