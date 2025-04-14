import os
import re
import ast
import argparse
import json
from mcp.server.fastmcp import FastMCP
import asyncio
from dataclasses import dataclass
from pydantic import Field
from util import is_subfolder
from loguru import logger
from typing import Any
from collections import deque

server = FastMCP("codeAnalyzer")


def recursive_ast_walk(node, isRoot=False):
    classes = []
    functions = []
    variables = []

    for child in selective_walk(node):
        if isinstance(child, ast.ClassDef):
            class_info = {
                "name": child.name,
            }
            classes.append(class_info)
            child_classes, child_functions, child_variables = recursive_ast_walk(child)
            if child_classes:
                class_info["classes"] = child_classes
            if child_functions:
                class_info["functions"] = child_functions
            if child_variables:
                class_info["variables"] = child_variables
        elif isinstance(child, ast.FunctionDef) or isinstance(
            child, ast.AsyncFunctionDef
        ):
            function_info = {
                "name": child.name,
            }
            functions.append(function_info)
            child_classes, child_functions, child_variables = recursive_ast_walk(child)
            if child_classes:
                function_info["classes"] = child_classes
            if child_functions:
                function_info["functions"] = child_functions
            if child_variables:
                function_info["variables"] = child_variables
        elif isRoot and (
            isinstance(child, ast.Assign)
            or isinstance(child, ast.AnnAssign)
            or isinstance(child, ast.AugAssign)
        ):
            targets = child.targets if isinstance(child, ast.Assign) else [child.target]
            for target in targets:
                if isinstance(target, ast.Attribute) or isinstance(
                    target, ast.Subscript
                ):
                    continue
                if isinstance(target, ast.Tuple):
                    for t in target.elts:
                        variable_info = {
                            "name": t.id,
                        }
                        variables.append(variable_info)
                else:
                    variable_info = {
                        "name": target.id,
                    }
                    variables.append(variable_info)

    return classes, functions, variables


def selective_walk(node):
    """
    Recursively yield all descendant nodes in the tree starting at *node*
    (not including *node* itself, excluding the children of *node* of type ast.ClassDef,
    ast.FunctionDef, ast.AsyncFunctionDef, ast.Assign, ast.AnnAssign, ast.AugAssign),
    in no specified order.
    """
    todo = deque(ast.iter_child_nodes(node))
    while todo:
        child_node = todo.popleft()
        if isinstance(child_node, ast.ClassDef):
            yield child_node
        elif isinstance(child_node, ast.FunctionDef) or isinstance(
            child_node, ast.AsyncFunctionDef
        ):
            yield child_node
        elif (
            isinstance(child_node, ast.Assign)
            or isinstance(child_node, ast.AnnAssign)
            or isinstance(child_node, ast.AugAssign)
        ):
            if child_node.value:
                todo.extend([child_node.value])
            yield child_node
        else:
            todo.extend(ast.iter_child_nodes(child_node))
            yield child_node


@server.tool(
    name="analyze_code",
    description="Analyze the code and return a report",
)
@logger.catch
async def analyze_python_code(
    wd: str,
    include_patterns: list[str] = [".*\\.py"],
    exclude_patterns: list[str] = ["_.*\\.py", ".*_test.py"],
) -> dict[str, Any]:
    if not is_subfolder(parent_folder=os.getcwd(), child_folder=wd):
        raise ValueError(
            f"Working directory {wd} is not a subfolder of cwd {os.getcwd()}"
        )
    # get all files in the working directory
    files = os.listdir(wd)
    # filter files by include_patterns and exclude_patterns
    results = {}
    for file in files:
        add_file = False
        for include_pattern in include_patterns:
            # convert include_pattern to regex
            include_regex = re.compile(include_pattern)
            if include_regex.match(file):
                add_file = True
                break
        for exclude_pattern in exclude_patterns:
            exclude_regex = re.compile(exclude_pattern)
            if exclude_regex.match(file):
                add_file = False
                break
        if add_file:
            parsed = ast.parse(open(os.path.join(wd, file)).read())
            hierarchy = {}
            classes, functions, variables = recursive_ast_walk(parsed, isRoot=True)
            hierarchy["classes"] = classes
            hierarchy["functions"] = functions
            hierarchy["variables"] = variables
            results[file] = hierarchy
    # return the files
    logger.info(
        f"Found {len(results)} files to analyze in {wd}\n{json.dumps(results, indent=4)}"
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wd", type=str, default=".")
    parser.add_argument("--include", type=str, default=".*\\.py")
    parser.add_argument("--exclude", type=str, default="_.*\\.py|.*_test\\.py")
    args = parser.parse_args()

    # asyncio.run(
    #    analyze_python_code(args.wd, args.include.split("|"), args.exclude.split("|"))
    # )
    server.run(transport="stdio")
