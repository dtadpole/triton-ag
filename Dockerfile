ARG WORK_DIR=none
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel

RUN apt update && apt install -y less nano git

# Install stable packages first for better caching
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    aiohttp \
    anthropic \
    autoawq \
    black \
    boto3 \
    flake8 \
    google-generativeai \
    h5py \
    isort \
    jupyter \
    jupyterlab \
    loguru \
    matplotlib \
    mcp \
    "mcp[cli]" \
    ninja \
    openai-agents \
    pandas \
    pre-commit \
    protobuf \
    pydantic \
    pyyaml \
    runpod \
    scikit-learn \
    sentencepiece \
    together \
    torch==2.4.0 \
    triton \
    unsloth

# Install frequently changed or version-pinned packages separately
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    "accelerate>=0.25.0" \
    "bitsandbytes>=0.41.0" \
    "datasets>=2.14.0" \
    "peft>=0.6.0" \
    "sglang[all]>=0.4.6.post5" \
    "transformers>=4.36.0" \
    "trl>=0.7.0"

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    mlflow

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Install Node.js (latest LTS) and npx using nvm
ENV NVM_DIR=/root/.nvm
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash && \
    . "$NVM_DIR/nvm.sh" && \
    nvm install --lts && \
    nvm use --lts && \
    nvm alias default 'lts/*' && \
    ln -s "$NVM_DIR/versions/node/$(nvm version)/bin/node" /usr/local/bin/node && \
    ln -s "$NVM_DIR/versions/node/$(nvm version)/bin/npm" /usr/local/bin/npm && \
    ln -s "$NVM_DIR/versions/node/$(nvm version)/bin/npx" /usr/local/bin/npx
