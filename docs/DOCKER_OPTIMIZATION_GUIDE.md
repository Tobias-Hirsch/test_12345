# Docker-Kommentar

Hinweis

## 🚀 Kommentar

### 1. Kommentar

Hinweis
- `requirements.core.txt` - Hinweis
- `requirements.extras.txt` - Hinweis

```bash
# Kommentar
echo "new-stable-package==1.0.0" >> backend/requirements.core.txt

# Kommentar
echo "docx2txt==0.8" >> backend/requirements.extras.txt
echo "openpyxl==3.1.2" >> backend/requirements.extras.txt
```

### 2. Kommentar

Hinweis`Dockerfile.optimized`Hinweis
- **base-dependencies**: Hinweis
- **core-dependencies**: Hinweis
- **variable-dependencies**: Hinweis
- **final**: Hinweis

### 3. Kommentar

Hinweis
```bash
# Kommentar
./scripts/build-optimized.sh --push --version v1.2.3

# Kommentar
# Kommentar✅
# Kommentar🔄
# Kommentar⚡
```

## 📦 Kommentar

| Hinweis| Hinweis| Hinweis| Hinweis|
|------|---------|---------|---------|
| Hinweis| 15-20Hinweis| 3-5Hinweis| 70-75% |
| Hinweis| 15-20Hinweis| 1-2Hinweis| 85-90% |
| Hinweis| 15-20Hinweis| 8-12Hinweis| 40-50% |

## 🛠️ Kommentar

### Kommentar

```bash
# Kommentar
./scripts/build-optimized.sh

# Kommentar
echo "new-package==1.0.0" >> backend/requirements.extras.txt
./scripts/build-optimized.sh  # Hinweis

# Kommentar
./scripts/build-optimized.sh  # Hinweis
```

### Kommentar

```bash
# Kommentar
./scripts/build-optimized.sh --push --version $(date +%Y%m%d-%H%M%S) --registry your-registry.com
```

### CI/CDKommentar

Hinweis`.github/workflows/docker-optimized.yml`Hinweis
- Hinweis
- Hinweis
- Hinweis
- Hinweis

## 🔧 HochKommentar

### 1. Kommentar

```bash
# AktivDocker BuildKitKommentar
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Kommentar
docker build --cache-from type=local,src=/tmp/.buildx-cache
```

### 2. Kommentar

```bash
# Kommentar
docker build --target core-dependencies --tag rosti-backend:cache-core .

# Kommentar
docker pull your-registry.com/rosti-backend:cache-core
docker pull your-registry.com/rosti-backend:cache-extras
```

### 3. Kommentar

```bash
# Kommentar
docker buildx build --platform linux/amd64,linux/arm64 --push .
```

## 📋 Kommentar

### Kommentar

```bash
# Kommentar
docker system prune -f

# Kommentar
docker builder prune -f
```

### Kommentar

```bash
# Kommentar
docker system df

# Kommentar
docker build --progress=plain . 2>&1 | grep -E "(CACHED|DONE)"
```

### Kommentar

1. **Hinweis**: Hinweis`requirements.core.txt`
2. **Hinweis**: Hinweis
3. **Hinweis**: Hinweis

## 🚨 Kommentar

### Kommentar

```bash
# Kommentar
docker build --no-cache .

# Kommentar
docker buildx prune --filter type=exec.cachemount
```

### Kommentar

```bash
# Kommentar
docker build --progress=plain .

# Kommentar
docker build --target core-dependencies .
```

## 📈 Kommentar

Hinweis

- **Hinweis%+**: Hinweis
- **CI/CDHinweis**: Hinweis
- **RessourceHinweis**: Hinweis
- **Hinweis**: Hinweis

## 🔄 Kommentar

1. Hinweis`requirements.txt`
2. Hinweis`requirements.core.txt`Hinweis`requirements.extras.txt`
3. Hinweis
4. Hinweis
5. Hinweis
6. Hinweis