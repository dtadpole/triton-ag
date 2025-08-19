FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}


ARG WORK_DIR=none
ARG HTTP_PROXY="http://fwdproxy:8080"
ARG HTTPS_PROXY="http://fwdproxy:8080"


RUN echo 'APT::Sandbox::User "root";' | tee -a /etc/apt/apt.conf.d/10sandbox


# add -o APT::Sandbox::User=root for proxy to work
RUN apt-get -o APT::Sandbox::User=root update && apt-get -o APT::Sandbox::User=root install -y less nano git


# RUN echo 'APT::Sandbox::User "root";' | tee -a /etc/apt/apt.conf.d/10sandbox
# The installer requires curl (and certificates) to download the release archive
RUN apt-get -o APT::Sandbox::User=root update && apt-get -o APT::Sandbox::User=root install -y --no-install-recommends curl ca-certificates

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

RUN npm config set proxy http://fwdproxy:8080
RUN npm config set https-proxy http://fwdproxy:8080


RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    wget \
    git \
    build-essential \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

RUN pip install pip-tools

# Set working directory
WORKDIR /workspace

### Command to compile requirements_devserver.in
# COPY requirements_devserver.in .

# RUN if [ -f constraints.txt ]; then \
#         pip-compile requirements.in --constraint constraints.txt; \
#     else \
#         pip-compile requirements.in; \
#     fi
# RUN pip-compile
# RUN pip-compile --resolver=backtracking requirements_devserver.in


COPY requirements_devserver.txt .

RUN pip install --no-cache-dir \
    --ignore-installed \
    --force-reinstall \
    -r requirements_devserver.txt

# Install cron, move to the top next time when rearrange the Dockerfile
RUN apt-get -o APT::Sandbox::User=root update && apt-get -o APT::Sandbox::User=root install -y cron tmux sshfs fuse3
