"""Prometheus metrics for the Pulse Agent."""

from prometheus_client import Counter, Enum, Gauge, Histogram

# Core pipeline metrics
REVIEWS_INGESTED = Counter(
    "pulse_reviews_ingested_total",
    "Reviews ingested",
    ["product", "source"],
)

CLUSTERS_FORMED = Gauge(
    "pulse_clusters_formed",
    "Clusters formed",
    ["product", "run_id"],
)

LLM_TOKENS = Counter(
    "pulse_llm_tokens_total",
    "LLM tokens used",
    ["model", "node"],
)

LLM_COST_USD = Counter(
    "pulse_llm_cost_usd_total",
    "LLM cost in USD",
    ["model"],
)

MCP_LATENCY = Histogram(
    "pulse_mcp_call_latency_seconds",
    "MCP call latency",
    ["server", "tool"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

PUBLISH_STATUS = Enum(
    "pulse_publish_status",
    "Publish status",
    ["product", "target"],
    states=["success", "failure", "skipped"],
)

RUN_DURATION = Histogram(
    "pulse_run_duration_seconds",
    "Full run duration",
    ["product"],
    buckets=[30, 60, 120, 180, 300, 600],
)
