# Lokales Setup mit Docker + lokalem Ollama

Diese lokale Variante startet Datenbanken, Backend, Frontend und Gateway per Docker. Ollama läuft **nicht** im Docker-Stack, sondern auf deinem lokalen Rechner. Dadurch nutzt Ollama deine vorhandene CUDA-/GPU-Installation direkt.

## Voraussetzungen

- Docker / Docker Compose
- Ollama lokal installiert und gestartet
- Optional für GPU-Container: NVIDIA Container Toolkit bzw. GPU-Unterstützung in Docker Desktop

## 1. Ollama starten und Modelle laden

PowerShell oder Terminal:

```bash
ollama serve
ollama pull qwen3:0.6b
ollama pull mxbai-embed-large:latest
```

Falls `ollama serve` meldet, dass der Port bereits belegt ist, läuft Ollama wahrscheinlich schon.

## 2. Stack starten

### Windows PowerShell

```powershell
.\scripts\start-local.ps1
```

### Linux/macOS/Git Bash

```bash
./scripts/start-local.sh
```

Oder manuell:

```bash
docker compose -f docker-compose.local.yml up -d --build
```

## 3. Aufrufen

- Frontend: http://localhost:8080
- Backend API Docs: http://localhost:8001/docs
- MinIO Console: http://localhost:9101
  - Benutzer: `minioadmin`
  - Passwort: `minioadmin`
- Standard-Login der App:
  - Benutzer: `admin`
  - Passwort: `admin12345`

## Optional: OCR-Services mit GPU starten

Die OCR-Services sind standardmäßig deaktiviert, damit der erste lokale Start einfacher ist. Wenn Docker deine NVIDIA-GPU verwenden kann:

```bash
docker compose -f docker-compose.local.yml --profile gpu-ocr up -d --build
```

Danach kannst du in `.env.local` diese Werte setzen, wenn die App OCR aktiv nutzen soll:

```env
PADDLEOCR_API_URL=http://paddleocr:8080
LATEXOCR_API_URL=http://latexocr:8002/predict
```

Anschließend Backend neu starten:

```bash
docker compose -f docker-compose.local.yml restart backend
```

## Wichtige Änderungen gegenüber der ursprünglichen Server-Konfiguration

- Keine festen Serverpfade wie `/home/cjb/...`
- Keine festen Server-IP-Adressen
- Keine festen GPU-IDs wie `0,1,2,3`
- Ollama wird über `http://host.docker.internal:11434/api` angesprochen
- MinerU-SGLang ist lokal deaktiviert; PDF-Verarbeitung läuft im Pipeline-Modus
- MinIO, MySQL, MongoDB, Redis und Milvus verwenden lokale Docker-Volumes

## Nützliche Befehle

Logs anzeigen:

```bash
docker compose -f docker-compose.local.yml logs -f backend
```

Stack stoppen:

```bash
docker compose -f docker-compose.local.yml down
```

Stack stoppen und lokale Daten löschen:

```bash
docker compose -f docker-compose.local.yml down -v
```

Ollama-Verbindung aus dem Backend-Container prüfen:

```bash
docker compose -f docker-compose.local.yml exec backend curl http://host.docker.internal:11434/api/tags
```
