import dspy
import json
import os
import shutil
from datetime import datetime
import asyncio
import argparse
from dspy_util import load_lm, load_instructions_for_module
from util import logger, get_next_run_folder
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from kbEvalTest.kbeval import KernelExecResult
import mlflow

AGENT_NAME = "DSPy_KernelCoder"
AGENT_SHORT_NAME = "KC"
AGENT_VERSION = "0.1"

# define a DSPy module for reference measurement
class ReferenceMeasurement(dspy.Module):
    """Reference Measurement"""

    def __init__(self, lm: dspy.LM, current_wd: str, model_tag: str, task_tag: str, time_tag: str, dspy_tools: list[dspy.Tool]):
        super().__init__()
        self.lm = lm
        self.current_wd = current_wd
        self.model_tag = model_tag
        self.task_tag = task_tag
        self.time_tag = time_tag
        self.dspy_tools = dspy_tools
        # initialize the react agent
        self.react = dspy.ReAct("user_request, current_wd, model_tag, task_tag, time_tag, reference_filename -> reference_result: KernelExecResult", tools=self.dspy_tools)

    async def forward(self, reference_filename: str) -> KernelExecResult:
        # run the react agent
        result = await self.react.acall(
            user_request="Run kernel bench evaluation for the reference code",
            current_wd=self.current_wd,
            model_tag=self.model_tag, 
            task_tag=self.task_tag,
            time_tag=self.time_tag,
            reference_filename=os.path.join(self.current_wd, reference_filename)
        )
        logger.info(f"[reference] {json.dumps(result.get("reference_result", {}).model_dump(), indent=4)}")
        # logger.info(f"result: {result}")
        # logger.info(json.dumps(result.get("reasoning", {}), indent=4))
        # logger.info(json.dumps(result.get("trajectory", {}), indent=4))
        # logger.info(json.dumps(react.history, indent=4))
        return result.get("reference_result", {})
    

class CUDAIterativeCoder(dspy.Module):
    """CUDA Iteration Coder"""

    def __init__(self, lm: dspy.LM, current_wd: str, model_tag: str, task_tag: str, time_tag: str, dspy_tools: list[dspy.Tool]):
        super().__init__()
        self.lm = lm
        self.current_wd = current_wd
        self.model_tag = model_tag
        self.task_tag = task_tag
        self.time_tag = time_tag
        self.dspy_tools = dspy_tools
        
        configs = load_instructions_for_module("CUDAIterativeCoder")
        if "custom_instruction" in configs:
            self.custom_instruction = configs["custom_instruction"]

        if "examples" in configs:
            idx = 0
            for example in configs["examples"]:
                idx += 1
                self.custom_instruction += f"\n\nExample #{idx}:"
                self.custom_instruction += f"\n\nInput/reference_code:\n{example['reference_code']}\n"
                self.custom_instruction += f"\n\nOutput/generated_code:\n{example['generated_code']}\n"

        # initialize the signature
        self.signature = dspy.Signature("current_wd: str, model_tag: str, task_tag: str, time_tag: str, eval_tag: str, reference_code: str, prev_generated_code: str, prev_iteration_result: KernelExecResult -> generated_code: str, generated_result: KernelExecResult", instructions=self.custom_instruction)
        # initialize the react agent
        self.react = dspy.ReAct(self.signature, tools=self.dspy_tools)

    async def forward(self, eval_tag: str, reference_code: str, prev_generated_code: str = None, prev_iteration_result: KernelExecResult = None) -> tuple[str, KernelExecResult]:
        # run the react agent
        result = await self.react.acall(
            current_wd=self.current_wd,
            model_tag=self.model_tag,
            task_tag=self.task_tag,
            time_tag=self.time_tag,
            eval_tag=eval_tag,
            reference_code=reference_code,
            prev_generated_code=None, # prev_generated_code,
            prev_iteration_result=prev_iteration_result,
        )
        # logger.info(result.get("generated_code", ""))
        logger.info(f"[{eval_tag}] {json.dumps(result.get("generated_result", {}).model_dump(), indent=4)}")
        return result.get("generated_code", ""), result.get("generated_result", {})


class CUDARolloutPlanner(dspy.Module):
    """CUDARolloutPlanner"""
    
    def __init__(self, lm: dspy.LM, num_rollout: int, num_iter: int, current_wd: str, model_tag: str, task_tag: str, time_tag: str, dspy_tools: list[dspy.Tool]):
        super().__init__()
        self.lm = lm
        self.num_rollout = num_rollout
        self.num_iter = num_iter
        self.current_wd = current_wd
        self.model_tag = model_tag
        self.task_tag = task_tag
        self.time_tag = time_tag
        self.dspy_tools = dspy_tools

        self.reference_measurement = ReferenceMeasurement(lm, current_wd, model_tag, task_tag, time_tag, dspy_tools)

        self.cuda_iterative_coder = CUDAIterativeCoder(lm, current_wd, model_tag, task_tag, time_tag, dspy_tools)

    async def forward(self, reference_code: str) -> tuple[KernelExecResult, str, KernelExecResult]:

        # write reference code to the run folder
        reference_filename = os.path.join(self.current_wd, "reference_code.py")
        with open(reference_filename, "w") as f:
            f.write(reference_code)

        # run the reference measurement
        reference_result = await self.reference_measurement.forward(reference_filename)

        best_generated_code: str = None
        best_generated_result: KernelExecResult = None

        prev_generated_code: str = None
        prev_generated_result: KernelExecResult = None
        for i in range(self.num_iter):
            eval_tag = f"r01_i{i+1:02d}"
            generated_code, generated_result = await self.cuda_iterative_coder.forward(
                eval_tag,
                reference_filename,
                prev_generated_code=prev_generated_code,
                prev_iteration_result=prev_generated_result
            )
            prev_generated_code = generated_code
            prev_generated_result = generated_result

            if generated_result.compiled and generated_result.correctness and generated_result.runtime > 0:
                if best_generated_result is None or generated_result.runtime < best_generated_result.runtime:
                    best_generated_code = generated_code
                    best_generated_result = generated_result

        return reference_result, best_generated_code, best_generated_result



async def run(lm: dspy.LM, args: argparse.Namespace):

    # get the next run folder
    run_folder = get_next_run_folder()
    os.makedirs(run_folder, exist_ok=True)

    # Create server parameters for stdio connection
    kb_eval_params = StdioServerParameters(
        command="uv",
        args=["run", "--with", "mcp", "mcp", "run", "kbEvalMCPServer.py"],
        env=os.environ,
    )

    file_server_params = StdioServerParameters(
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            os.path.join(os.getcwd(), run_folder),
        ],
        env=os.environ,
    )

    # Connect to both MCP servers
    async with stdio_client(kb_eval_params) as (kb_read, kb_write), \
               stdio_client(file_server_params) as (file_read, file_write):
        
        async with ClientSession(kb_read, kb_write) as kb_session, \
                   ClientSession(file_read, file_write) as file_session:
            
            # Initialize both connections
            await kb_session.initialize()
            await file_session.initialize()
            
            # List available tools from both servers
            kb_tools = await kb_session.list_tools()
            file_tools = await file_session.list_tools()

            # Convert MCP tools to DSPy tools from both servers
            dspy_tools = []
            
            # Add tools from kernel benchmark evaluation server
            for tool in kb_tools.tools:
                dspy_tools.append(dspy.Tool.from_mcp_tool(kb_session, tool))
            
            # Add tools from file server
            for tool in file_tools.tools:
                dspy_tools.append(dspy.Tool.from_mcp_tool(file_session, tool))

            # Print all available tools
            logger.info(f"loaded {len(kb_tools.tools)} tools from [kb_session]")
            logger.info(f"loaded {len(file_tools.tools)} tools from [file_session]")
            logger.info(f"Total tools: {len(dspy_tools)}")

            current_wd = run_folder
            model_tag = "deepseek/deepseek-chat"
            task_tag = "level1/1_Square_matrix_multiplication_"
            time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

            # read reference code from the original file
            original_reference_filename = os.path.join(os.getcwd(), "kernel_bench", "level1", "1_Square_matrix_multiplication_.py")
            with open(original_reference_filename, "r") as f:
                reference_code = f.read()

            # initialize the rollout planner
            rollout_planner = CUDARolloutPlanner(lm, num_rollout=args.num_rollout, num_iter=args.num_iter, current_wd=current_wd, model_tag=model_tag, task_tag=task_tag, time_tag=time_tag, dspy_tools=dspy_tools)

            # run the rollout planner
            rollout_result = await rollout_planner(reference_code)
            logger.info(json.dumps(rollout_result.get("reference_result", {}), indent=4))
            logger.info(json.dumps(rollout_result.get("best_generated_code", ""), indent=4))
            logger.info(json.dumps(rollout_result.get("best_generated_result", {}), indent=4))




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--provider", type=str, default="deepseek")
    parser.add_argument("-m", "--model", type=str, default="deepseek-chat")
    parser.add_argument("-n", "--num_rollout", type=int, default=1)
    parser.add_argument("-i", "--num_iter", type=int, default=4)
    args = parser.parse_args()

    mlflow.set_tracking_uri("http://127.0.0.1:5050")
    mlflow.set_experiment(f"{AGENT_SHORT_NAME}_{AGENT_VERSION}_{args.provider}_{args.model}_DSPy")
    mlflow.dspy.autolog()

    lm = load_lm(args.provider, args.model)
    dspy.configure(lm=lm)
    logger.info(f"Loaded model: {lm.model}")

    # Run the async function after setting up the language model
    asyncio.run(run(lm, args))

if __name__ == "__main__":
    main()
