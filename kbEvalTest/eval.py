import torch
import argparse
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
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
    with open(reference_code_file, "r") as f:
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

    code_obj = compile(generated_code, args.input_file, "exec")
    exec(code_obj, context)
    model_new = context.get("ModelNew")
    if model_new is None:
        logger.error(f"❌ [eval] ModelNew not found")
        return

    init_input_data = init_input_func()
    ref_module = model(*init_input_data).cuda(device=args.device)
    gen_module = model_new(*init_input_data).cuda(device=args.device)

    input_data = input_func()
    input_data = [
        x.cuda(device=args.device) if isinstance(x, torch.Tensor) else x
        for x in input_data
    ]

    ref_output = ref_module(*input_data)
    gen_output = gen_module(*input_data)
    print(f"ref_output: {ref_output.shape} {ref_output}")
    print(f"gen_output: {gen_output.shape} {gen_output}")

    # check if ref_output and gen_output are equal using torch.allclose
    if not torch.allclose(ref_output, gen_output, atol=1e-2, rtol=1e-2):
        logger.error(f"❌ [eval] Reference and generated outputs are not equal")
        return
    
    logger.info(f"✅ [eval] Reference and generated outputs are equal")


if __name__ == "__main__":
    main()