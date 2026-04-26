# Pulse Agent — Server

AI-powered weekly product review analysis. Backend built with FastAPI, LangGraph, and async Python.

## Architecture

```
app/
├── core/           # Domain models, types, exceptions
├── api/            # FastAPI REST endpoints
├── agent/          # LangGraph state machine
│   ├── nodes/      # Ingest, cluster, summarize, render, publish
│   └── prompts/    # Version-controlled LLM prompts
├── ingestion/      # App Store + Play Store scrapers
├── clustering/     # Embeddings + UMAP + HDBSCAN
├── summarization/  # LLM client + validators + cost tracker
├── renderer/       # Google Docs tree + email HTML/text
├── mcp_client/     # MCP client for Google Workspace
├── db/             # SQLAlchemy models + async sessions
└── observability/  # OpenTelemetry + metrics
```

## Quick Start

```bash
# Install dependencies
uv pip install -e ".[dev]"

# Initialize database
uv run pulse init-db-cmd

# Run smoke tests
uv run pytest tests/unit -v

# Verify the configured real LLM before a full run
uv run python -c "import asyncio; from app.summarization.llm_client import LLMClient; print(asyncio.run(LLMClient().ensure_ready()))"

# Run full pipeline (requires a valid real LLM key; embeddings are local by default)
uv run pulse run --product groww --weeks 10
```

## Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_PROVIDER` | Yes | `groq`, `openai`, `custom`, or `auto` |
| `GROQ_API_KEY` | If Groq | For Groq summarization and action generation |
| `OPENAI_API_KEY` | If OpenAI | For OpenAI-compatible summarization, or OpenAI embeddings |
| `LLM_ALLOW_HEURISTIC_FALLBACK` | No | Local-only deterministic fallback when no remote key is configured |
| `LLM_ALLOW_HEURISTIC_AUTH_FALLBACK` | No | Keep `false` for demos/prod so invalid remote keys fail fast |
| `DATABASE_URL` | No | Defaults to SQLite |
| `EMBEDDING_BACKEND` | No | Defaults to `huggingface-local` |
| `EMBEDDING_MODEL` | No | Defaults to `Qwen/Qwen3-Embedding-0.6B` |
| `DOCS_MCP_URL` | Phase 5+ | Google Docs MCP server |
| `GMAIL_MCP_URL` | Phase 6+ | Gmail MCP server |
| `CONFIRM_SEND` | No | Set `true` to actually send emails |

The default embedding path uses the free Apache-2.0 Hugging Face model
`Qwen/Qwen3-Embedding-0.6B`. The first clustering run downloads that model into
the local Hugging Face cache, so the first run is slower than later runs.

## MCP Servers

Run these in separate terminals from the `server` directory before starting a
non-dry-run publish flow:

```powershell
uv run uvicorn mcp_servers.docs_server:app --host 0.0.0.0 --port 5000
uv run uvicorn mcp_servers.smtp_server:app --host 127.0.0.1 --port 5001
```

Use `DOCS_MCP_URL=http://localhost:5000` and
`GMAIL_MCP_URL=http://localhost:5001` in `.env`. Docs publishing needs either
`GOOGLE_APPLICATION_CREDENTIALS` or `GOOGLE_SERVICE_ACCOUNT_JSON`. Email delivery
needs `GMAIL_SENDER` and a Gmail app password in `GMAIL_APP_PASSWORD`.

## Quality Gates

```bash
# Lint
ruff check app tests

# Type check
mypy app tests

# Test
pytest tests/unit -v --cov=app
```
