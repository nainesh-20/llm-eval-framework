FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .
RUN pip install --no-cache-dir -e .

# Create results directory
RUN mkdir -p results

# Default: show help. Override with:
#   docker run --env-file .env llm-eval run --config configs/eval.yaml
ENTRYPOINT ["llm-eval"]
CMD ["--help"]
