FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency definitions
COPY pyproject.toml ./
COPY README.md ./
RUN mkdir -p app && touch app/__init__.py

# Install dependencies
RUN uv pip install --system -e "."

# Production stage
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app/ ./app/
COPY mcp_servers/ ./mcp_servers/
COPY data/ ./data/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY prompts/ ./prompts/
COPY boot.sh ./boot.sh

# Create data directory
RUN mkdir -p data/raw data/artifacts
RUN chmod +x ./boot.sh

ENV PYTHONPATH=/app
ENV DATABASE_URL=sqlite+aiosqlite:///./data/pulse.db

EXPOSE 8000

CMD ["bash", "./boot.sh"]
