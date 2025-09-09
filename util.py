import os

def is_devserver() -> bool:
    import socket

    hostname = socket.gethostname()
    return "facebook.com" in hostname

## Define the global data folders
if is_devserver():
    KB_EVAL_DIR = "shared/.kbeval"
else:
    KB_EVAL_DIR = os.path.join(os.path.expanduser("~"), ".kbeval")

if is_devserver():
    INFERENCE_DIR = "shared/.inference"
else:
    INFERENCE_DIR = os.path.join(os.path.expanduser("~"), ".inference")

if is_devserver():
    TRAINER_DIR = "shared/.trainer"
else:
    TRAINER_DIR = "~/.trainer"

if is_devserver():
    WORKFLOW_DIR = "shared/.workflow"
else:
    WORKFLOW_DIR = "~/.workflow"

if is_devserver():
    CONFIG_FOLDER = "shared/config/"
else:
    CONFIG_FOLDER = "./"
