FROM python:3.14-slim-bookworm

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files and install
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application code
COPY fabricai_inference_server/ fabricai_inference_server/

EXPOSE 8000

ENV PYTHON_GIL=0
CMD ["uv", "run", "uvicorn", "fabricai_inference_server.app:app", "--host", "0.0.0.0", "--port", "8000"]
