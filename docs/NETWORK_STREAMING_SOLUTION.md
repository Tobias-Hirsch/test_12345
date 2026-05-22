# Kommentar

## 🚨 Kommentar

Hinweis`ERR_INCOMPLETE_CHUNKED_ENCODING` FehlerJaHinweis

1. **ExcelHinweis**: 12KBHinweis
2. **Hinweis**: map-reduceHinweis
3. **Hinweis**: Hinweis
4. **Hinweis**: Hinweis

## 🛠️ Kommentar

### 1. Kommentar

#### A. Kommentar

**Kommentar**: `backend/app/tools/document_processor.py`

Hinweis
- Hinweis
- Hinweis
- Hinweis

```python
# Kommentar
def _is_structured_data(text: str) -> bool:
    table_indicators = ['Arbeitsblatt:', 'Unnamed:', '|', '\t', '---']
    nan_count = text.count('NaN')
    total_lines = len(text.split('\n'))
    return nan_count > total_lines * 0.3 or any(indicator in text for indicator in table_indicators)

# Kommentar
def _extract_table_summary(text: str) -> str:
    # Kommentar
    # Kommentar
```

#### B. Kommentar

**Kommentar**: `backend/app/services/chat_response_service.py`

Hinweis
- Hinweis
- Hinweis
- Hinweis

```python
# Kommentar
if current_time - last_heartbeat > 10:
    yield {"event": "heartbeat", "data": f"Processing... ({chunk_count} chunks processed)"}

if chunk_count % 3 == 0:
    yield {"event": "progress", "data": f"Summarizing document... ({chunk_count} sections processed)"}
```

### 2. Kommentar

#### A. NginxKommentar

**Kommentar**: `nginx/nginx.conf`

Hinweis
```nginx
# Kommentar
proxy_read_timeout 900s;           # 15Hinweis
proxy_buffering off;               # Hinweis
proxy_request_buffering off;       # Hinweis

# Kommentar
location ~ ^/api/chat/conversations/.+/messages/ {
    proxy_pass http://backend;
    proxy_read_timeout 900s;
    proxy_buffering off;
}
```

#### B. Docker ComposeKommentar

```yaml
services:
  nginx:
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/conf.d/default.conf
    depends_on:
      - backend
      - frontend
```

### 3. Kommentar

#### A. Kommentar

**Kommentar**: `frontend/src/composables/useStreamingWithRetry.ts`

Hinweis
- Hinweis
- Hinweis
- Hinweis
- Hinweis

#### B. Kommentar

**Kommentar**: `frontend/src/services/resilientApiService.ts`

Hinweis
- Hinweis
- Hinweis
- Fehlerhinweis
- Hinweis

#### C. Kommentar

**Kommentar**: `frontend/src/components/ResilientChatInterface.vue`

BenutzerHinweis
- Hinweis
- Hinweisäge
- Hinweis
- Fehlerhinweis

## 📋 Kommentar

### 1. Kommentar

```bash
# 1. Kommentar
cp backend/app/tools/document_processor.py.new backend/app/tools/document_processor.py
cp backend/app/services/chat_response_service.py.new backend/app/services/chat_response_service.py

# 2. Kommentar
docker-compose build backend

# 3. Kommentar
docker-compose restart backend
```

### 2. Kommentar

```bash
# 1. Kommentar
cp frontend/src/composables/useStreamingWithRetry.ts frontend/src/composables/
cp frontend/src/services/resilientApiService.ts frontend/src/services/
cp frontend/src/composables/useResilientChatSending.ts frontend/src/composables/

# 2. Kommentar
# Kommentar

# 3. Kommentar
docker-compose build frontend

# 4. Kommentar
docker-compose restart frontend
```

### 3. Kommentar

```bash
# 1. Kommentar
cp nginx/nginx.conf /path/to/nginx/conf.d/default.conf

# 2. Kommentar
docker-compose exec nginx nginx -s reload

# OderKommentar
docker-compose restart nginx
```

## 🔧 Kommentar

### Kommentar

| Hinweis| Hinweis| Hinweis| Hinweis|
|------|------|---------|------|
| Nginx | proxy_read_timeout | 900s | 15Hinweis|
| Hinweis| timeoutMs | 900000 | 15Hinweis|
| Hinweis| heartbeat_interval | 10s | Hinweis|

### Kommentar

| Hinweis| Hinweis| Hinweis|
|------|---------|------|
| maxRetries | 3 | Hinweis|
| retryDelay | 2000ms | Hinweis|
| backoffMultiplier | 1.5 | Hinweis|

## 📊 Kommentar

### Kommentar

| Hinweis| Hinweis| Hinweis| Hinweis|
|----------|--------|--------|---------|
| ExcelHinweis| 3Hinweis| 20-30Hinweis| 85% ⬇️ |
| Hinweis| Hinweis| 1Hinweis| 90% ⬇️ |
| Hinweis| Hinweis| Hinweis| Hinweis⬆️ |

### BenutzerKommentar

- ✅ Hinweis
- ✅ Hinweisäge  
- ✅ Hinweis
- ✅ Hinweis
- ✅ Hinweis

### Kommentar

- ✅ Hinweis
- ✅ Hinweis
- ✅ Hinweis
- ✅ Hinweis

## 🚦 Kommentar

### 1. Kommentar

```bash
# Kommentar
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"content":"Hinweis","attachments":[...]}' \
  http://your-server/api/chat/conversations/test/messages/
```

### 2. Kommentar

```bash
# Kommentar
# Kommentar
```

### 3. Kommentar

```bash
# HochladenKommentar
# Bestätigen15Kommentar
```

## 🔍 Kommentar

### Kommentarüsselwörter

- `Detected structured data` - Hinweis
- `Processing... (X chunks processed)` - Hinweis
- `Retrying stream (attempt X/3)` - Hinweis
- `Attempting to recover partial response` - Hinweis

### Kommentar

1. **Hinweis**: Hinweis
2. **Hinweis**: Hinweis
3. **Hinweis**: Hinweis
4. **Hinweis**: BestätigenHinweis

## 🎯 Kommentar

1. **WebSocketHinweis**: Hinweis
2. **Hinweis**: Hinweis
3. **Hinweis**: Hinweis
4. **Hinweis**: Hinweis

Hinweis%Hinweis