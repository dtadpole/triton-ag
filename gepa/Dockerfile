# GEPA Extension for Triton-AG Docker Image
# ==========================================
# Uses the existing triton_ag image as base and adds GEPA dependencies.
#
# Build: docker build -t gepa:latest .
# Run:   docker run -it --rm -e OPENAI_API_KEY=$OPENAI_API_KEY gepa:latest
# Shell: docker run -it --rm -e OPENAI_API_KEY=$OPENAI_API_KEY gepa:latest bash

FROM localhost/triton_ag:latest

# Set working directory for GEPA
WORKDIR /workspace/

# Copy only requirements first for better caching
COPY requirements.txt .

# Install GEPA-specific packages (only what's not in base image)
RUN pip install --no-cache-dir \
    gepa>=0.0.22 \
    dspy>=2.6.0 \
    pyyaml \
    python-dotenv

# Copy the rest of the GEPA module
COPY . .

# Default command: run the dependency test
CMD ["python", "examples/test_dependencies.py"]
