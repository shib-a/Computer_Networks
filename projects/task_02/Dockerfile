# CPU-only image — saves ~1.8 GB vs the default CUDA wheel
FROM python:3.11-slim

WORKDIR /app

# System deps (needed by torchvision)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (separate step → Docker layer cache hit on rebuilds)
RUN pip install --no-cache-dir \
    torch==2.1.2+cpu \
    torchvision==0.16.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.100.0 \
    "uvicorn[standard]>=0.20.0"

# Force NumPy 1.x last — PyTorch 2.1.x was compiled against 1.x ABI;
# transitive deps may upgrade it to 2.x, so we pin it down here.
RUN pip install --no-cache-dir --force-reinstall "numpy==1.26.4"

# Copy application code (late stage → fast rebuilds when only code changes)
COPY . .

# Default entrypoint — override via docker-compose `command:`
CMD ["python3", "-m", "server.main"]
