import argparse
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    args = parser.parse_args()

    # check if input file exists
    if not os.path.exists(args.input_file):
        logger.error(f"❌ [eval] Input file [{args.input_file}] not found")
        return

    # read input file
    with open(args.input_file, "r") as f:
        generated_code = f.read()

    # get reference code
    reference_code_file = os.path.join(os.path.dirname(args.input_file), "reference_code.py")
    if not os.path.exists(reference_code_file):
        logger.error(f"❌ [eval] Reference code [{reference_code_file}] not found")
        return

    # read reference code
    with open(reference_code, "r") as f:
        reference_code = f.read()

    context = {}
    exec(reference_code, context)
    # get reference function
    input_func = context.get("get_inputs")
    if input_func is None:
        logger.error(f"❌ [eval] Input function not found")
        return

    init_input_func = context.get("get_init_inputs")
    if init_input_func is None:
        logger.error(f"❌ [eval] Init input function not found")
        return

    model = context.get("Model")
    if model is None:
        logger.error(f"❌ [eval] Model not found")
        return

    model_new = context.get("ModelNew")
    if model_new is None:
        logger.error(f"❌ [eval] ModelNew not found")
        return

if __name__ == "__main__":
    main()