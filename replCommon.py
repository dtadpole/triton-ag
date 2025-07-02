from pydantic import BaseModel
import yaml

class CodeRequest(BaseModel):
    code: str

class CodeResponse(BaseModel):
    result: str = None
    output: str = None
    error: str = None
    is_exit: bool = False

def load_config(config_file: str = 'repl.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config
