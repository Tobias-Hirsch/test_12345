# Änderungen: deutsche Lokalisierung und OCR

Diese Version führt die deutsche Bereinigung der lokalen Projektversion fort.

## Bereits geprüft / umgesetzt

- Systemprompt für direkten Chat ist auf Deutsch gesetzt: `backend/app/services/chat_response_service.py` → `DIRECT_CHAT_SYSTEM_PROMPT`
- RAG-Systemprompt ist auf Deutsch gesetzt: `backend/app/services/chat_response_service.py` → `_construct_rag_system_prompt(...)`
- RAG-Q&A-Systemprompt ist auf Deutsch gesetzt: `backend/app/rag_knowledge/generic_knowledge.py`
- Frontend-Sprachen sind auf Deutsch/Englisch reduziert
- Alte chinesische Locale-Datei `frontend/src/locales/zh.json` wurde entfernt
- Standard-/Fallback-Sprache ist Deutsch
- Login-Sprachumschaltung nutzt denselben localStorage-Key wie die restliche App: `language`
- PaddleOCR ist auf Deutsch voreingestellt und über `PADDLEOCR_LANG` konfigurierbar
- MinerU-Pipeline ist lokal auf `lang="german"` voreingestellt
- `.env.local` verweist auf die lokalen OCR-Services im Docker-Netzwerk
- `docker-compose.local.yml` setzt `PADDLEOCR_LANG="german"` für den PaddleOCR-Service
- Einige sichtbare Platzhaltertexte in der ABAC-/Policy-Oberfläche wurden bereinigt

## Nicht übersetzt

- `models/paddlex-models/.../inference.yml` enthält OCR-Modellvokabular. Diese Datei darf nicht manuell übersetzt werden, da sonst das OCR-Modell beschädigt werden kann.

## Lokaler Start

Ohne OCR-Profil:

```bash
docker compose -f docker-compose.local.yml up -d --build
```

Mit OCR-Profil:

```bash
docker compose -f docker-compose.local.yml --profile gpu-ocr up -d --build
```

Nach Änderungen an `.env.local` sollte der Backend-Container neu erzeugt werden:

```bash
docker compose -f docker-compose.local.yml up -d --force-recreate backend
```
