#!/usr/bin/env bash
set -euo pipefail

echo "Checking Ollama on http://localhost:11434 ..."
if ! curl -fsS http://localhost:11434/api/tags >/dev/null; then
  echo "Ollama is not reachable. Start Ollama first, then run this script again." >&2
  exit 1
fi

echo "Pulling recommended local models if missing ..."
ollama pull qwen3:0.6b
ollama pull mxbai-embed-large:latest

echo "Starting local Docker stack ..."
docker compose -f docker-compose.local.yml up -d --build

echo ""
echo "Frontend:        http://localhost:8080"
echo "Backend API:     http://localhost:8001/docs"
echo "MinIO Console:   http://localhost:9101  (minioadmin / minioadmin)"
echo "Default login:   admin / admin12345"
