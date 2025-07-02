#!/usr/bin/env python3
"""
Simple FastAPI REPL
A web-based Read-Eval-Print Loop using FastAPI with global context
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import sys
import io
import asyncio
import traceback
from globalRegistry import GlobalRegistry
from logger import logger
from replCommon import CodeRequest, CodeResponse, load_config

reg = GlobalRegistry()
fastapi: FastAPI = reg.get('reg.fastapi')
global_namespace = {'reg': reg}

def execute_code(code: str) -> tuple:
    """Execute Python code and capture output/errors"""
    output = io.StringIO()
    error = None
    result = None
    is_exit = False
    
    try:
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = output
        
        try:
            # Try as expression first
            try:
                result = eval(code, global_namespace)
            except SyntaxError:
                # Not an expression, execute as statement
                exec(code, global_namespace)
        finally:
            sys.stdout = old_stdout

    except SystemExit:
        is_exit = True
    except Exception as e:
        error = traceback.format_exc()
        logger.error(error)
    
    return result, output.getvalue(), error, is_exit


@fastapi.post("/execute")
async def execute(request: CodeRequest):
    """Execute Python code"""
    try:
        result, output, error, is_exit = execute_code(request.code)
        
        return {
            "result": str(result) if result is not None else None,
            "output": output if output else None,
            "error": error,
            "is_exit": is_exit
        }
    
    except Exception as e:
        return {
            "result": None,
            "output": None,
            "error": f"Server error: {str(e)}",
            "is_exit": False
        }


@fastapi.get("/vars")
def get_variables():
    """Get current variables"""
    user_vars = {k: str(v) for k, v in global_namespace.items() 
                if not k.startswith('_')}
    return {"variables": user_vars}


@fastapi.post("/reset")
def reset():
    """Reset the global namespace"""
    global_namespace.clear()
    global_namespace['reg'] = reg
    global_namespace['vars'] = get_variables
    global_namespace['reset'] = reset
    return {"message": "Namespace reset"}

# call reset
reset()

@fastapi.get("/", response_class=HTMLResponse)
async def web_repl():
    """Simple web interface for the REPL"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Simple Web REPL</title>
    <style>
        body { font-family: monospace; margin: 20px; background: #1e1e1e; color: #d4d4d4; }
        .container { max-width: 800px; margin: 0 auto; }
        .output { background: #2d2d30; padding: 10px; border-radius: 5px; margin: 10px 0; min-height: 200px; white-space: pre-wrap; }
        .input-area { display: flex; margin: 10px 0; }
        textarea { flex: 1; padding: 10px; background: #3c3c3c; border: 1px solid #555; color: white; font-family: monospace; resize: vertical; min-height: 60px; }
        button { padding: 10px 20px; background: #007acc; color: white; border: none; cursor: pointer; margin-left: 5px; }
        button:hover { background: #005a9e; }
        .error { color: #f44747; }
        .result { color: #4ec9b0; }
        .output-text { color: #d4d4d4; }
        .info { color: #608b4e; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🐍 Simple Web REPL</h1>
        
        <div id="output" class="output">Welcome to Simple Web REPL!
Type Python code below and press Execute.

>>> </div>
        
        <div class="input-area">
            <textarea id="code-input" placeholder="Enter Python code... (Shift+Enter, Ctrl+Enter, or Alt+Enter to execute, Enter for new line)" rows="3"></textarea>
            <button onclick="runCode(false)">Execute</button>
            <button onclick="clearScreen()">Clear Output</button>
            <button onclick="showVars()">Variables</button>
            <button onclick="resetAll()">Reset</button>
        </div>
        
        <div class="info">
            <small>Keyboard: Shift+Enter, Ctrl+Enter, or Alt+Enter (execute and clear) | Button: Execute (keep code) | Enter: new line</small>
        </div>
    </div>

    <script>
        function escapeHtml(text) {
            return text
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;"); // or &apos;
        }

        function runCode(shouldClear) {
            var input = document.getElementById('code-input');
            var output = document.getElementById('output');
            var code = input.value.trim();
            
            if (!code) return;
            
            // Show what we're executing (preserve formatting for multiline)
            var codeLines = code.split('\\n');
            for (var i = 0; i < codeLines.length; i++) {
                var prefix = i === 0 ? '' : '... ';
                output.innerHTML += prefix + codeLines[i] + '\\n';
            }
            
            // Send to server
            fetch('/execute', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ code: code })
            })
            .then(function(response) {
                return response.json();
            })
            .then(function(data) {
                if (data.output) {
                    output.innerHTML += escapeHtml(data.output);
                }
                if (data.result !== null && data.result !== undefined) {
                    output.innerHTML += '<span class="result">' + escapeHtml(data.result) + '</span>\\n';
                }
                if (data.error) {
                    output.innerHTML += '<span class="error">' + escapeHtml(data.error) + '</span>\\n';
                }
                
                output.innerHTML += '>>> ';
                output.scrollTop = output.scrollHeight;
            })
            .catch(function(error) {
                output.innerHTML += '<span class="error">Error: ' + escaleHtml(error.message) + '</span>\\n>>> ';
            });
            
            // Clear input if requested
            if (shouldClear) {
                input.value = '';
            }
        }
        
        function clearScreen() {
            document.getElementById('output').innerHTML = 'Welcome to Simple Web REPL!\\n\\n>>> ';
        }

        function showVars() {
            fetch('/vars')
            .then(function(response) {
                return response.json();
            })
            .then(function(data) {
                var output = document.getElementById('output');
                output.innerHTML += 'Variables:\\n';
                
                if (Object.keys(data.variables).length === 0) {
                    output.innerHTML += '  (no variables defined)\\n';
                } else {
                    for (var name in data.variables) {
                        output.innerHTML += '  ' + name + ' = ' + escapeHtml(data.variables[name]) + '\\n';
                    }
                }
                output.innerHTML += '>>> ';
                output.scrollTop = output.scrollHeight;
            });
        }
        
        function resetAll() {
            fetch('/reset', { method: 'POST' })
            .then(function(response) {
                return response.json();
            })
            .then(function(data) {
                var output = document.getElementById('output');
                output.innerHTML += data.message + '\\n>>> ';
                output.scrollTop = output.scrollHeight;
            });
        }
        
        // Key event handling - Clear input on execution shortcuts
        document.getElementById('code-input').addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                if (e.shiftKey) {
                    // Shift+Enter: execute code and clear input
                    e.preventDefault();
                    runCode(true);
                } else if (e.ctrlKey || e.metaKey) {
                    // Ctrl+Enter (or Cmd+Enter on Mac): execute code and clear input
                    e.preventDefault();
                    runCode(true);
                } else if (e.altKey) {
                    // Alt+Enter: execute code and clear input
                    e.preventDefault();
                    runCode(true);
                } else {
                    // Just Enter: allow new line (default behavior)
                    // Do nothing, let the default behavior happen
                }
            }
        });
    </script>
</body>
</html>
    """


if __name__ == "__main__":
    asyncio.run(reg.run())