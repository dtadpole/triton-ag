"""
CUDA Kernel Generation Adapter for GEPA.

This adapter integrates GEPA with the triton-ag InferenceClient
to optimize prompts for CUDA kernel generation tasks.

It uses KbEvalClient to evaluate generated CUDA kernels for:
- Compilation success
- Correctness (numerical accuracy)
- Performance (runtime/speedup)
"""

import asyncio
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

# Append /workspace to sys.path at the END (after site-packages)
# This ensures installed packages (like gepa) take precedence,
# while custom modules in /workspace are still importable.
if "/workspace" not in sys.path:
    sys.path.append("/workspace")

from gepa.core.adapter import EvaluationBatch, GEPAAdapter


@dataclass
class CudaKernelDataInst:
    """Data instance for CUDA kernel generation task."""

    task_id: str
    reference_code: str
    expected_output: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CudaKernelTrajectory:
    """Trajectory for CUDA kernel generation."""

    data: CudaKernelDataInst
    system_prompt: str
    user_prompt: str
    generated_code: str
    compilation_result: Optional[str] = None
    correctness_result: Optional[str] = None
    speedup: Optional[float] = None
    eval_metadata: Optional[Dict[str, Any]] = None


@dataclass
class CudaKernelOutput:
    """Output from CUDA kernel generation."""

    generated_code: str
    compilation_success: bool
    correctness_success: bool
    speedup: Optional[float] = None
    runtime: Optional[float] = None
    error_message: Optional[str] = None
    eval_metadata: Optional[Dict[str, Any]] = None


class TraceLogger:
    """Logger for tracking LLM responses, KbEval results, and execution traces."""

    def __init__(self, run_dir: Optional[str] = None, enabled: bool = True):
        self.enabled = enabled and run_dir is not None
        self.run_dir = Path(run_dir) if run_dir else None
        self.step_count = 0
        self.eval_count = 0

        if self.enabled and self.run_dir:
            # Create log directories
            self.traces_dir = self.run_dir / "traces"
            self.llm_responses_dir = self.traces_dir / "llm_responses"
            self.kbeval_results_dir = self.traces_dir / "kbeval_results"
            self.evaluations_dir = self.traces_dir / "evaluations"

            for d in [
                self.traces_dir,
                self.llm_responses_dir,
                self.kbeval_results_dir,
                self.evaluations_dir,
            ]:
                d.mkdir(parents=True, exist_ok=True)

            # Create summary log file
            self.summary_log = self.traces_dir / "summary.log"
            with open(self.summary_log, "w") as f:
                f.write(f"GEPA CUDA Kernel Optimization Trace Log\n")
                f.write(f"Started: {datetime.now().isoformat()}\n")
                f.write("=" * 60 + "\n\n")

    def _write_summary(self, message: str):
        """Append to summary log."""
        if not self.enabled:
            return
        with open(self.summary_log, "a") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")

    def log_llm_request(
        self,
        task_id: str,
        system_prompt: str,
        user_prompt: str,
        candidate_idx: Optional[int] = None,
    ):
        """Log LLM request (prompts)."""
        if not self.enabled:
            return

        self.step_count += 1
        filename = f"{self.step_count:04d}_{task_id}_request.json"
        filepath = self.llm_responses_dir / filename

        data = {
            "step": self.step_count,
            "task_id": task_id,
            "candidate_idx": candidate_idx,
            "timestamp": datetime.now().isoformat(),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        self._write_summary(f"LLM Request: {task_id} (step {self.step_count})")

    def log_llm_response(
        self,
        task_id: str,
        raw_response: str,
        extracted_code: str,
        candidate_idx: Optional[int] = None,
    ):
        """Log LLM response (generated code)."""
        if not self.enabled:
            return

        filename = f"{self.step_count:04d}_{task_id}_response.json"
        filepath = self.llm_responses_dir / filename

        data = {
            "step": self.step_count,
            "task_id": task_id,
            "candidate_idx": candidate_idx,
            "timestamp": datetime.now().isoformat(),
            "raw_response": raw_response,
            "extracted_code": extracted_code,
            "code_length": len(extracted_code),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        # Also save the raw code as a .py file for easy viewing
        code_filename = f"{self.step_count:04d}_{task_id}_code.py"
        code_filepath = self.llm_responses_dir / code_filename
        with open(code_filepath, "w") as f:
            f.write(f"# Task: {task_id}\n")
            f.write(f"# Step: {self.step_count}\n")
            f.write(f"# Timestamp: {datetime.now().isoformat()}\n\n")
            f.write(extracted_code)

        self._write_summary(
            f"LLM Response: {task_id} (code length: {len(extracted_code)})"
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

        self.eval_count += 1
        filename = f"{self.eval_count:04d}_{task_id}_kbeval.json"
        filepath = self.kbeval_results_dir / filename

        data = {
            "eval_count": self.eval_count,
            "step": self.step_count,
            "task_id": task_id,
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

        self._write_summary(f"KbEval: {task_id} - {status} (score: {score:.3f})")

    def log_evaluation_batch(
        self,
        candidate: Dict[str, str],
        batch_size: int,
        scores: List[float],
        outputs: List[Any],
    ):
        """Log a complete evaluation batch."""
        if not self.enabled:
            return

        filename = f"batch_{self.eval_count:04d}.json"
        filepath = self.evaluations_dir / filename

        avg_score = sum(scores) / len(scores) if scores else 0.0
        success_count = sum(
            1 for o in outputs if getattr(o, "correctness_success", False)
        )

        data = {
            "eval_count": self.eval_count,
            "timestamp": datetime.now().isoformat(),
            "candidate_prompt_preview": candidate.get("system_prompt", "")[:200],
            "batch_size": batch_size,
            "avg_score": avg_score,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "success_count": success_count,
            "scores": scores,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        self._write_summary(
            f"Batch Eval: {batch_size} tasks, avg={avg_score:.3f}, "
            f"success={success_count}/{batch_size}"
        )

    def log_candidate_prompt(self, candidate_idx: int, candidate: Dict[str, str]):
        """Log a candidate prompt being evaluated."""
        if not self.enabled:
            return

        filename = f"candidate_{candidate_idx:04d}.txt"
        filepath = self.evaluations_dir / filename

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
        self._write_summary(f"Total LLM calls: {self.step_count}")
        self._write_summary(f"Total evaluations: {self.eval_count}")
        self._write_summary(f"Best score: {best_score:.4f}")

        # Save best prompt
        best_prompt_file = self.traces_dir / "best_prompt.txt"
        with open(best_prompt_file, "w") as f:
            f.write("Best System Prompt\n")
            f.write("=" * 60 + "\n\n")
            f.write(best_candidate.get("system_prompt", ""))


class CudaKernelAdapter(GEPAAdapter):
    """
    GEPA Adapter for CUDA kernel generation optimization.

    This adapter:
    1. Uses InferenceClient to call your vLLM-hosted models
    2. Uses KbEvalClient to evaluate generated CUDA kernels
    3. Provides rich feedback for prompt evolution
    4. Logs all traces to run_dir for examination
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
        # Scoring settings
        failure_score: float = 0.0,
        compile_only_score: float = 0.3,
        correct_score: float = 0.7,
        speedup_bonus_weight: float = 0.3,
        # Tags for tracking
        run_tag: str = "gepa_optimization",
        model_tag: str = "gepa",
        # Logging settings
        run_dir: Optional[str] = None,
        enable_tracing: bool = True,
    ):
        """
        Initialize the CUDA Kernel Adapter.

        Args:
            model_name: Model name from inferenceClient.yaml
            provider_name: Provider name from inferenceClient.yaml
            config_file: Path to inferenceClient.yaml
            max_tokens: Maximum tokens for generation
            eval_provider_name: Provider for KbEvalClient
            eval_config_file: Path to kbEval.yaml
            use_kb_eval: Whether to use KbEvalClient for evaluation
            failure_score: Score for failed generation
            compile_only_score: Score for code that compiles but fails tests
            correct_score: Base score for correct code
            speedup_bonus_weight: Weight for speedup bonus (0-1)
            run_tag: Tag for tracking evaluation runs
            model_tag: Tag for tracking model
            run_dir: Directory to save traces (LLM responses, KbEval results, etc.)
            enable_tracing: Whether to enable trace logging
        """
        self.model_name = model_name
        self.provider_name = provider_name
        self.config_file = config_file
        self.max_tokens = max_tokens

        # KbEvalClient settings
        self.eval_provider_name = eval_provider_name
        self.eval_config_file = eval_config_file
        self.use_kb_eval = use_kb_eval

        # Scoring settings
        self.failure_score = failure_score
        self.compile_only_score = compile_only_score
        self.correct_score = correct_score
        self.speedup_bonus_weight = speedup_bonus_weight

        # Tags
        self.run_tag = run_tag
        self.model_tag = model_tag

        # Logging
        self.run_dir = run_dir
        self.enable_tracing = enable_tracing
        self.trace_logger = TraceLogger(run_dir=run_dir, enabled=enable_tracing)
        self._candidate_idx = 0

        # Lazy load clients
        self._inference_client = None
        self._eval_client = None

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

    def _build_user_prompt(self, data: CudaKernelDataInst) -> str:
        """Build user prompt from data instance."""
        prompt = f"""Following is the reference PyTorch code, implement the complete new module with CUDA (no testing code, no other code).

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
        data: CudaKernelDataInst,
        generated_code: str,
    ) -> tuple:
        """
        Evaluate generated CUDA code using KbEvalClient.

        Returns:
            (score, output, feedback)
        """
        if not generated_code:
            return (
                self.failure_score,
                CudaKernelOutput(
                    generated_code="",
                    compilation_success=False,
                    correctness_success=False,
                    error_message="Empty generation",
                ),
                "FAILED: Empty code generation. The model did not produce any code.",
            )

        # Call KbEvalClient
        result = await self.eval_client.kb_eval(
            provider=self.eval_provider_name,
            reference_code=data.reference_code,
            generated_code=generated_code,
            run_tag=self.run_tag,
            model_tag=self.model_tag,
            task_tag=data.task_id,
            eval_tag="gepa_eval",
            code_type="cuda",
        )

        if result is None:
            return (
                self.failure_score,
                CudaKernelOutput(
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

        # Calculate score and build feedback from full kbeval result
        if not compiled:
            score = self.failure_score
        elif not correctness:
            score = self.compile_only_score
        else:
            # Success! Calculate score with speedup bonus
            score = self.correct_score
            # Add speedup bonus if runtime is valid
            if runtime > 0:
                ref_runtime = result.get("reference_runtime", runtime)
                if ref_runtime > 0:
                    speedup = ref_runtime / runtime
                    if speedup > 1.0:
                        bonus = min(speedup - 1.0, 1.0) * self.speedup_bonus_weight
                        score = min(score + bonus, 1.0)

        # Build feedback directly from kbeval result
        feedback = self._format_kbeval_feedback(result)

        output = CudaKernelOutput(
            generated_code=generated_code,
            compilation_success=compiled,
            correctness_success=correctness,
            speedup=runtime_stats.get("speedup") if runtime_stats else None,
            runtime=runtime if runtime > 0 else None,
            error_message=metadata.get("error") if not correctness else None,
            eval_metadata=result,  # Store the full KB eval result including metadata
        )

        return score, output, feedback

    def _evaluate_code_simple(
        self,
        data: CudaKernelDataInst,
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
                CudaKernelOutput(
                    generated_code="",
                    compilation_success=False,
                    correctness_success=False,
                    error_message="Empty generation",
                ),
                "FAILED: Empty code generation",
            )

        # Simple heuristic checks
        has_model_new = "class ModelNew" in generated_code
        has_cuda_kernel = "__global__" in generated_code
        has_load_inline = "load_inline" in generated_code

        if has_model_new and has_cuda_kernel and has_load_inline:
            score = self.correct_score
            feedback = (
                "LIKELY CORRECT: Code contains ModelNew class, "
                "CUDA kernel (__global__), and load_inline. "
                "(Note: Not verified by actual compilation/execution)"
            )
            output = CudaKernelOutput(
                generated_code=generated_code,
                compilation_success=True,
                correctness_success=True,
            )
        elif has_model_new and (has_cuda_kernel or has_load_inline):
            score = self.compile_only_score
            missing = []
            if not has_cuda_kernel:
                missing.append("CUDA kernel (__global__)")
            if not has_load_inline:
                missing.append("load_inline")
            feedback = (
                f"PARTIAL: Code has ModelNew but missing: {', '.join(missing)}. "
                "Ensure you include both a CUDA kernel and load_inline."
            )
            output = CudaKernelOutput(
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
                "2. __global__ - CUDA kernel function\n"
                "3. load_inline - to compile CUDA code"
            )
            output = CudaKernelOutput(
                generated_code=generated_code,
                compilation_success=False,
                correctness_success=False,
                error_message="Missing ModelNew class",
            )

        return score, output, feedback

    def evaluate(
        self,
        batch: List[CudaKernelDataInst],
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """
        Evaluate a batch of inputs with the given prompt candidate.

        Args:
            batch: List of CudaKernelDataInst
            candidate: Dict with 'system_prompt' key
            capture_traces: Whether to capture execution traces

        Returns:
            EvaluationBatch with outputs, scores, and optionally traces
        """
        system_prompt = candidate.get("system_prompt", "")

        # Log candidate prompt
        self._candidate_idx += 1
        self.trace_logger.log_candidate_prompt(self._candidate_idx, candidate)

        outputs: List[CudaKernelOutput] = []
        scores: List[float] = []
        trajectories: List[CudaKernelTrajectory] = [] if capture_traces else None

        async def process_batch():
            # Step 1: Generate code for all inputs
            gen_tasks = []
            for data in batch:
                user_prompt = self._build_user_prompt(data)

                # Log LLM request
                self.trace_logger.log_llm_request(
                    task_id=data.task_id,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    candidate_idx=self._candidate_idx,
                )

                task = self._generate_async(system_prompt, user_prompt)
                gen_tasks.append((data, user_prompt, task))

            gen_results = []
            for data, user_prompt, task in gen_tasks:
                response = await task
                generated_code = self._extract_code(response)

                # Log LLM response
                self.trace_logger.log_llm_response(
                    task_id=data.task_id,
                    raw_response=response,
                    extracted_code=generated_code,
                    candidate_idx=self._candidate_idx,
                )

                gen_results.append((data, user_prompt, response, generated_code))

            # Step 2: Evaluate all generated code
            eval_results = []
            for data, user_prompt, response, generated_code in gen_results:
                if self.use_kb_eval and self.eval_client is not None:
                    score, output, feedback = await self._evaluate_with_kb_eval(
                        data, generated_code
                    )
                else:
                    score, output, feedback = self._evaluate_code_simple(
                        data, generated_code
                    )

                # Log KbEval result - use full result if available from KB eval
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

                eval_results.append(
                    (data, user_prompt, generated_code, score, output, feedback)
                )

            return eval_results

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
                    CudaKernelTrajectory(
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
                # Build detailed feedback
                if output.compilation_success and output.correctness_success:
                    if output.speedup and output.speedup > 1.0:
                        feedback = (
                            f"SUCCESS with speedup {output.speedup:.2f}x! "
                            f"The code compiles, produces correct results, "
                            f"and runs faster than the reference. "
                            f"Runtime: {output.runtime:.2f}μs."
                        )
                    else:
                        feedback = (
                            "SUCCESS: Code compiles and passes all correctness tests. "
                            "Consider optimizing for better performance."
                        )
                elif output.compilation_success:
                    feedback = (
                        f"CORRECTNESS FAILED: Code compiles but fails correctness.\n"
                        f"Error: {output.error_message or 'Unknown error'}\n\n"
                        "Tips:\n"
                        "- Check tensor dimensions and shapes\n"
                        "- Verify data types (float32, etc.)\n"
                        "- Ensure thread/block indexing is correct\n"
                        "- Check for race conditions in parallel code"
                    )
                else:
                    feedback = (
                        f"COMPILATION FAILED:\n"
                        f"Error: {output.error_message or 'Unknown error'}\n\n"
                        "Common issues:\n"
                        "- Missing includes or headers\n"
                        "- Syntax errors in CUDA code\n"
                        "- Type mismatches\n"
                        "- Missing ModelNew class definition\n"
                        "- Incorrect load_inline usage"
                    )

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


def create_cuda_kernel_adapter(
    model_name: str = "qwen3-32b",
    provider_name: str = "h100_8_5_a",
    config_file: str = "inferenceClient.yaml",
    eval_provider_name: str = "local",
    eval_config_file: str = "kbEval.yaml",
    use_kb_eval: bool = True,
    **kwargs,
) -> CudaKernelAdapter:
    """Factory function to create a CudaKernelAdapter."""
    return CudaKernelAdapter(
        model_name=model_name,
        provider_name=provider_name,
        config_file=config_file,
        eval_provider_name=eval_provider_name,
        eval_config_file=eval_config_file,
        use_kb_eval=use_kb_eval,
        **kwargs,
    )
