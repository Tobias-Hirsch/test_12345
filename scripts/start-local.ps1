$ErrorActionPreference = "Stop"

Write-Host "Checking Ollama on http://localhost:11434 ..."
try {
  Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 5 | Out-Null
} catch {
  Write-Host "Ollama is not reachable. Start Ollama first, then run this script again." -ForegroundColor Red
  exit 1
}

Write-Host "Pulling recommended local models if missing ..."
ollama pull qwen3:0.6b
ollama pull mxbai-embed-large:latest

Write-Host "Starting local Docker stack ..."
docker compose -f docker-compose.local.yml up -d --build

Write-Host ""
Write-Host "Frontend:        http://localhost:8080"
Write-Host "Backend API:     http://localhost:8001/docs"
Write-Host "MinIO Console:   http://localhost:9101  (minioadmin / minioadmin)"
Write-Host "Default login:   admin / admin12345"
