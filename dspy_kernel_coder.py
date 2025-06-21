import dspy
import argparse
from dspy_util import load_lm
from util import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from kbEvalTest.kbeval import KernelExecResult


class BenchmarkReference(dspy.Signature):
    """ReferenceBenchmark"""
    reference_code: str = dspy.InputField(description="Reference code")
    reference_result: KernelExecResult = dspy.OutputField(description="Reference code execution result")

    def forward(self) -> KernelExecResult:
        pass
    

class KernelCoder(dspy.Signature):
    """KernelCoder"""
    reference_code: str = dspy.InputField(description="Reference code")
    
    reference_result: KernelExecResult = dspy.OutputField(description="Reference code execution result")
    generated_code: str = dspy.OutputField(description="Generated code")
    generated_summary: str = dspy.OutputField(description="Generated code summary")
    generated_result: KernelExecResult = dspy.OutputField(description="Generated code execution result")

    def forward(self, reference_code: str, reference_result: KernelExecResult) -> KernelExecResult:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, default="anthropic")
    parser.add_argument("--model", type=str, default="claude-4-sonnet")
    args = parser.parse_args()

    lm = load_lm(args.provider, args.model)
    dspy.configure(lm=lm)
    logger.info(f"Loaded model: {lm.model}")

if __name__ == "__main__":
    main()