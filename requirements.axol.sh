#!/bin/bash

uv pip install packaging setuptools wheel
uv pip install torch
uv pip install awscli pydantic psutil

uv pip install --no-build-isolation axolotl[deepspeed]
uv pip install flash-attn
