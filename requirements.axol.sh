#!/bin/bash

uv pip install packaging setuptools wheel
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install awscli pydantic psutil

uv pip install --no-build-isolation axolotl[deepspeed]
uv pip install -U flash-attn --no-build-isolation
