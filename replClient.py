#!/usr/bin/env python3
"""
Simple REPL CLI Tool
A basic Read-Eval-Print Loop for Python
"""

import sys
import os
import requests
import ast
import traceback
import readline
from logger import logger
from replUtils import CodeRequest, CodeResponse, load_config


class SimpleRepl:
    def __init__(self):
        self.buffer = []
        self.config = load_config()
        self.url = self.config['client']['url']
        api_key_file = self.config['client']['api_key']
        # expand ${HOME} to the home directory
        api_key_file = api_key_file.replace('${HOME}', os.path.expanduser('~'))
        # read the api key from the file
        with open(api_key_file, 'r') as f:
            self.api_key = f.read().strip()

        # Setup readline for better input handling
        try:
            readline.parse_and_bind("tab: complete")
            readline.set_history_length(1000)
        except:
            pass  # readline might not be available on all systems

    def is_complete(self, code):
        """Check if code is complete or needs more input"""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            # If it's incomplete, we expect more input
            if "unexpected EOF" in str(e):
                return False
            elif "IndentationError" in str(e):
                return False
            else:
                # Other syntax errors are actual errors
                return False
    
    def execute(self, code):
        """Execute Python code and return result"""
        try:
            response = requests.post(self.url + '/repl/execute', json={'code': code}, headers={'Authorization': f'Bearer {self.api_key}'})
            response.raise_for_status()
            result = response.json()
            if result['result']:
                print(result['result'])
            if result['output']:
                logger.info(result['output'])
            if result['error']:
                logger.error(result['error'])
            if result['is_exit']:
                sys.exit(0)
        except SystemExit:
            print("SystemExit")
            sys.exit(0)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt")
            raise
        except Exception as e:
            logger.error(f"Exception: {e}")
            traceback.print_exc()
            raise e
    
    def run(self):
        """Main REPL loop"""
        print("Simple Python REPL - Type 'exit()' or Ctrl+D to quit")
        
        while True:
            try:
                # Choose prompt based on whether we're continuing input
                prompt = "... " if self.buffer else ">>> "
                
                try:
                    line = input(prompt)
                except EOFError:
                    print("\nGoodbye!")
                    break
                
                # Add line to buffer
                self.buffer.append(line)
                code = '\n'.join(self.buffer)

                # If user enters empty line and has buffered code
                if len(self.buffer) > 1 and not line.strip():
                    code_to_execute = '\n'.join(self.buffer)  # Exclude empty line
                    if code_to_execute.strip():
                        self.execute(code_to_execute)
                    self.buffer = []
                
                elif len(self.buffer) > 1 and line.strip():
                    continue
                
                # If we have complete code (not waiting for more)
                elif self.is_complete(code):
                    self.execute(code)
                    self.buffer = []

                # Otherwise, continue reading more lines
                
            except KeyboardInterrupt:
                print("\nKeyboardInterrupt")
                self.buffer = []
                continue


def main():
    repl = SimpleRepl()
    repl.run()


if __name__ == "__main__":
    main()