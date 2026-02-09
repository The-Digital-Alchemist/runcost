# Modern Python Docker image with UV
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Copy dependency files
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .

# Copy source code
COPY costplan/ costplan/

# Install dependencies with UV (much faster than pip)
RUN uv pip install --system -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create non-root user
RUN useradd -m -u 1000 costplan && \
    chown -R costplan:costplan /app

USER costplan

# Default command
ENTRYPOINT ["costplan"]
CMD ["--help"]
