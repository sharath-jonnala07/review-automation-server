#!/usr/bin/env sh
set -eu

PORT="${PORT:-8000}"

if [ -n "${GOOGLE_SERVICE_ACCOUNT_JSON:-}" ] && [ ! -f /tmp/service-account.json ]; then
  printf '%s' "$GOOGLE_SERVICE_ACCOUNT_JSON" > /tmp/service-account.json
  export GOOGLE_APPLICATION_CREDENTIALS=/tmp/service-account.json
fi

if [ -f alembic.ini ]; then
  alembic upgrade head
fi

exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"