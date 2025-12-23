# Polysome Generic Workflow Runner Container
# Using CUDA 12.6 runtime to match JamePeng wheel availability
FROM nvidia/cuda:12.6.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set container metadata
LABEL org.opencontainers.image.source="https://github.com/computationalpathologygroup/Polysome"
LABEL org.opencontainers.image.description="Polysome Generic Workflow Runner with CUDA support"
LABEL org.opencontainers.image.version="1.0.0"

# Install Python 3.11 and build dependencies for vLLM/Triton
RUN apt-get update && apt-get install -y \
  software-properties-common \
  curl \
  libgomp1 \
  libgcc-s1 \
  git \
  git-lfs \
  build-essential \
  gcc \
  g++ \
  make \
  && add-apt-repository ppa:deadsnakes/ppa \
  && apt-get update && apt-get install -y \
  python3.11 \
  python3.11-venv \
  python3.11-dev \
  python3.11-distutils \
  && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11 using get-pip.py
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Install uv for faster package management
RUN python3.11 -m pip install --no-cache-dir uv

# Create symbolic links for python (force overwrite existing links)
RUN ln -sf /usr/bin/python3.11 /usr/bin/python \
  && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Create required directories for standardized mounting
RUN mkdir -p /opt/algorithm /models /data /output /workflows /prompts /tmp

# Set working directory
WORKDIR /opt/algorithm

# Copy dependency files first for better caching
COPY pyproject.toml .
COPY src/ ./src/
COPY inference.py .

# Install llama-cpp-python with CUDA support from JamePeng fork (has Gemma3 support)
RUN python3.11 -m pip install --no-cache-dir \
  https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.9-cu126-AVX2-linux-20250701/llama_cpp_python-0.3.9-cp311-cp311-linux_x86_64.whl

# Install the package with all optional dependencies using UV
RUN uv pip install --system -e .[all]


# Model weights placeholder - these will be mounted at runtime
# The actual models will be available at /models/ when the container runs
# Expected model files example:
#   /models/gemma-3-12b-it-q4_0.gguf
#   /models/gemma-3-27b-it-q4_0.gguf
RUN echo "Model weights will be mounted at runtime at /models/" > /models/README.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV WORKFLOW_PATH=/workflows/default.json

# Set Python path to include the source directory
ENV PYTHONPATH=/opt/algorithm/src


# Set the entrypoint to the generic inference script
ENTRYPOINT ["python", "inference.py"]

# Health check to verify the container is working
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import sys; sys.path.insert(0, '/opt/algorithm/src'); import polysome; print('Container is healthy')" || exit 1

# Add helpful metadata for the generic runner
LABEL algorithm.mount_points="/models,/data,/output,/workflows"
LABEL algorithm.environment_variables="WORKFLOW_PATH"
LABEL algorithm.model_requirements="LLaMA/Gemma compatible models mounted at /models/"
LABEL algorithm.description="Generic Polysome workflow runner - configure via WORKFLOW_PATH"
