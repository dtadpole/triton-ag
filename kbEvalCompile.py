import argparse
import torch
import os
from kbEvalTest.kbeval import load_custom_model


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_src", type=str, required=True)
    args.add_argument("--build_dir", type=str, default=None)
    args.add_argument("-m", "--max_jobs", type=int, default=4)
    args = args.parse_args() 

    context = {}

    with open(args.model_src, "r") as f:
        model_src = f.read()

    assert torch.cuda.is_available(), "CUDA is not available, cannot run Eval"
    torch.set_printoptions(
        precision=4,  # Decimal places
        threshold=10,  # Total number of elements before truncating
        edgeitems=3,  # Number of elements at beginning and end of dimensions
        linewidth=80,  # Maximum width before wrapping
    )

    os.environ["MAX_JOBS"] = str(args.max_jobs)
    os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
    # os.environ["TORCH_COMPILE_DEBUG"] = "1"

    load_custom_model(model_src, context, build_directory=args.build_dir)
