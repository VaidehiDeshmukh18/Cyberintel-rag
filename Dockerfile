# syntax=docker/dockerfile:1.6
# CyberIntel-RAG dashboard — single-stage build with uv for speed.
#
# Build:   docker build -t cyberintel-rag .
# Run:     docker run -p 8501:8501 cyberintel-rag
# Compose: see docker-compose.yml (recommended — brings up Qdrant + Ollama too)

FROM python:3.11.9-slim

# System dependencies: git for pip-from-git, build-essential for native wheels
# (shap, numpy, scikit-learn all need a C compiler on slim images).
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv — 10x faster than pip for the full install.
RUN pip install --no-cache-dir uv

WORKDIR /app

# Install dependencies first (cached layer).
COPY pyproject.toml ./
RUN uv pip install --system --no-cache \
      "streamlit>=1.35" \
      "pandas>=2.0" \
      "pydantic>=2.0" \
      "loguru" \
      "httpx" \
      "qdrant-client" \
      "sentence-transformers" \
      "transformers<4.42" \
      "torch" \
      "torchvision" \
      "rank_bm25" \
      "scikit-learn" \
      "joblib" \
      "shap" \
      "numpy"

# Copy source last (changes most often).
COPY src/ ./src/
COPY app/ ./app/
COPY scripts/ ./scripts/

# Data is mounted at runtime via docker-compose (don't bake into image).

EXPOSE 8501

# Streamlit needs --server.address 0.0.0.0 to be reachable from outside the container.
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV LLM_PROVIDER=ollama
ENV OLLAMA_URL=http://ollama:11434
ENV LLM_MODEL=llama3.2:latest

CMD ["streamlit", "run", "app/dashboard.py"]