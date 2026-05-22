# MinerUKommentar

## Kommentar

HinweisägeHinweis

## Kommentar

1. **Hinweis**: Hinweis
2. **Hinweis**: Hinweis
3. **Hinweis**: Hinweis
4. **Hinweis**: Hinweis

## Kommentar

### Kommentar

```
app/services/
├── mineru_service.py              # Hinweis
├── mineru_service_hybrid.py       # Hinweis
├── mineru_service_optimized.py    # Hinweis
└── mineru_vlm_optimized.py        # VLMHinweis
```

### Kommentar

```
app/services/
├── mineru_unified_service.py      # Hinweis
├── mineru_service.py              # Hinweis
├── mineru_service_hybrid.py       # Hinweis
├── mineru_service_optimized.py    # Hinweis
└── mineru_vlm_optimized.py        # Hinweis

app/core/
└── mineru_config.py               # Hinweis

app/utils/
└── mineru_error_handler.py        # FehlerFehler bei der Verarbeitung
```

## Kommentar

### 1. UnifiedMinerUProcessor (mineru_unified_service.py)

**Kommentar**
- Hinweis
- Hinweis
- Hinweis
- Hinweis

**Kommentar**
```python
async def process_document_bytes(file_bytes: bytes, filename: str, strategy: Optional[str] = None) -> Optional[Dict[str, Any]]
```

**Kommentar**
- **Hinweis**: Hinweis
- **Hinweis**: Hinweis
- **Hinweis**: Hinweis
- **Hinweis**: Hinweis

### 2. MinerUConfigManager (mineru_config.py)

**Kommentar**
- Hinweis
- Hinweis
- Hinweis
- Hinweis

**Kommentar**
```python
@dataclass
class ProcessingConfig:
    strategy: str
    max_retries: int
    timeout_seconds: int
    enable_preprocessing: bool
    fallback_enabled: bool
```

### 3. MinerUErrorHandler (mineru_error_handler.py)

**Kommentar**
- Fehlerhinweis
- Hinweis
- Hinweis
- Hinweis

**Kommentar**
- Hinweis
- Hinweis
- Hinweis
- Hinweis

## Kommentar

### Kommentaräge

```bash
# Kommentar
MINERU_FORCE_MODE=sglang  # sglang, vlm, pipeline, fallback

# Kommentar
MINERU_MAX_RETRIES=3
MINERU_TIMEOUT_SECONDS=600
MINERU_MAX_CONCURRENT_JOBS=3
MINERU_MEMORY_LIMIT_MB=2048

# Kommentar
MINERU_CACHE_ENABLED=true
MINERU_METRICS_ENABLED=true
```

### Kommentaräge

```bash
# Kommentar
ENVIRONMENT=production
MINERU_NIGHTTIME_HOURS=22-6
MINERU_SGLANG_SERVER_URL=http://1.116.119.85:8908
PDF_PROCESSING_STRATEGY=smart
```

## Kommentar

### Kommentar

1. ✅ Hinweis
2. ✅ Hinweis`app/tools/pdf.py`Hinweis
3. ✅ Hinweis
4. ✅ Hinweis

### Kommentar

1. Hinweis
2. Hinweis
3. Hinweis
4. Hinweis

### Kommentar

1. Hinweis
2. Hinweis
3. Hinweis
4. Hinweis

## APIKommentar

### Kommentar

```python
# Kommentar
from app.services.mineru_service import get_mineru_processor
processor = get_mineru_processor()
result = await processor.process_document_bytes(file_bytes, filename)

# Kommentar
from app.services.mineru_unified_service import get_unified_mineru_processor
processor = get_unified_mineru_processor()
result = await processor.process_document_bytes(file_bytes, filename, strategy="auto")
```

### Kommentar

```python
# Kommentar
processor = get_unified_mineru_processor()
stats = processor.get_performance_summary()

# Kommentar
from app.core.mineru_config import get_mineru_configuration_summary
config = get_mineru_configuration_summary()

# Kommentar
from app.utils.mineru_error_handler import get_mineru_error_handler
error_handler = get_mineru_error_handler()
error_stats = error_handler.get_error_statistics()
```

## Kommentar

### FehlerKommentar

- **Hinweis**: Hinweis
- **Hinweis**: Hinweis
- **Fehlerhinweis**: Hinweis

### Kommentar

- **Hinweis**: Hinweisäge
- **Hinweis**: Hinweis
- **Hinweis**: Hinweis

### Kommentar

- **Hinweis**: Hinweis
- **Hinweis**: HinweisößeHinweis
- **Hinweis**: Hinweis

## Kommentar

### 1. Kommentar

```bash
# Kommentar
curl -X POST "http://localhost:8000/api/rag/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.pdf"
```

### 2. Kommentar

```python
from app.core.mineru_config import validate_mineru_configuration
validation_result = validate_mineru_configuration()
print(validation_result)
```

### 3. Kommentar

```python
from app.services.mineru_unified_service import get_unified_mineru_processor
processor = get_unified_mineru_processor()
stats = processor.get_performance_summary()
print(f"Hinweis{stats['strategy_stats']['sglang']['success_rate']}")
```

## Kommentar

### Kommentar

1. **SGLangHinweis**
   - Hinweis`MINERU_SGLANG_SERVER_URL` Hinweis
   - Hinweis
   - Hinweis

2. **Hinweis**
   - Hinweis
   - Hinweis`MINERU_NIGHTTIME_HOURS`
   - BestätigenHinweis

3. **Hinweis**
   - Hinweis
   - Hinweis
   - Hinweis

### Kommentar

```bash
# Kommentar
grep "MinerU\|mineru" logs/app.log

# Kommentar
grep "error_handler" logs/app.log

# Kommentar
grep "performance" logs/app.log
```

## Kommentar

### Kommentar

1. Hinweis
2. Hinweis
3. Hinweis
4. Hinweis

### MittelKommentar

1. Hinweis
2. Hinweis
3. Hinweis
4. BenutzerHinweis

### Kommentar

1. Hinweis
2. Hinweis
3. HochHinweis
4. Hinweis

## Kommentar

HinweisägeHinweis

Hinweis
- **Hinweis**: Hinweis
- **Hinweis**: Hinweis
- **Hinweis**: Hinweis
- **Hinweis**: Hinweis
- **Hinweis**: Hinweis

Hinweis