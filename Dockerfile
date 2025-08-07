FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel
ARG WORK_DIR=none
ARG HTTP_PROXY="http://fwdproxy:8080"
ARG HTTPS_PROXY="http://fwdproxy:8080"


RUN echo 'APT::Sandbox::User "root";' | tee -a /etc/apt/apt.conf.d/10sandbox


# add -o APT::Sandbox::User=root for proxy to work
RUN apt-get -o APT::Sandbox::User=root update && apt-get -o APT::Sandbox::User=root install -y less nano git

# Install stable packages first for better caching
RUN pip install --no-cache-dir \
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
    protobuf \
    pydantic \
    pyyaml \
    runpod \
    scikit-learn \
    sentencepiece \
    together \
    triton \
    unsloth

# Install frequently changed or version-pinned packages separately
RUN pip install \
    "accelerate>=0.25.0" \
    "bitsandbytes>=0.41.0" \
    "datasets>=2.14.0" \
    "peft>=0.6.0" \
    "transformers>=4.36.0" \
    "trl>=0.7.0"

RUN pip install mlflow


# RUN echo 'APT::Sandbox::User "root";' | tee -a /etc/apt/apt.conf.d/10sandbox
# The installer requires curl (and certificates) to download the release archive
RUN apt-get -o APT::Sandbox::User=root update && apt-get -o APT::Sandbox::User=root install -y --no-install-recommends curl ca-certificates


# Download the latest installer
# ADD https://astral.sh/uv/install.sh /uv-installer.sh

# # Run the installer then remove it
# RUN sh /uv-installer.sh && rm /uv-installer.sh


RUN curl -LsSf https://astral.sh/uv/install.sh | sh

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

RUN pip install \
"sglang[all]>=0.4.6.post5"


RUN npm config set proxy http://fwdproxy:8080
RUN npm config set https-proxy http://fwdproxy:8080

RUN pip install openai-agents==0.0.19
RUN pip install transformers==4.52.1
RUN pip install together==1.5.8
RUN pip install wandb
RUN pip install duckdb
RUN pip install autoawq
RUN pip install unsloth==2025.7.8
RUN pip install vllm==0.9.1
RUN pip install llmcompressor
