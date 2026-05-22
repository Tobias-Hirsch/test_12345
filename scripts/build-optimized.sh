#!/bin/bash
# Docker-Kommentar

set -e

# Kommentar
REGISTRY="registry.your-company.com"
PROJECT_NAME="rosti-backend"
VERSION="${VERSION:-$(date +%Y%m%d-%H%M%S)}"

# Kommentar
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Kommentar
check_dependency_changes() {
    if [ -f "backend/requirements.core.txt.md5" ]; then
        CURRENT_CORE_MD5=$(md5sum backend/requirements.core.txt | cut -d' ' -f1)
        PREVIOUS_CORE_MD5=$(cat backend/requirements.core.txt.md5)
        
        if [ "$CURRENT_CORE_MD5" != "$PREVIOUS_CORE_MD5" ]; then
            echo_warn "Warnhinweis"
            REBUILD_CORE=true
        else
            echo_info "Hinweis"
            REBUILD_CORE=false
        fi
    else
        echo_warn "Warnhinweis"
        REBUILD_CORE=true
    fi

    if [ -f "backend/requirements.extras.txt.md5" ]; then
        CURRENT_EXTRAS_MD5=$(md5sum backend/requirements.extras.txt | cut -d' ' -f1)
        PREVIOUS_EXTRAS_MD5=$(cat backend/requirements.extras.txt.md5)
        
        if [ "$CURRENT_EXTRAS_MD5" != "$PREVIOUS_EXTRAS_MD5" ]; then
            echo_warn "Warnhinweis"
            REBUILD_EXTRAS=true
        else
            echo_info "Hinweis"
            REBUILD_EXTRAS=false
        fi
    else
        REBUILD_EXTRAS=true
    fi
}

# Kommentar
build_cache_layers() {
    echo_info "Hinweis"
    
    if [ "$REBUILD_CORE" = true ]; then
        echo_info "Hinweis"
        docker build \
            --target core-dependencies \
            --cache-from $REGISTRY/$PROJECT_NAME:cache-core \
            --tag $REGISTRY/$PROJECT_NAME:cache-core-$VERSION \
            --tag $REGISTRY/$PROJECT_NAME:cache-core \
            backend/
    fi
    
    if [ "$REBUILD_EXTRAS" = true ] || [ "$REBUILD_CORE" = true ]; then
        echo_info "Hinweis"
        docker build \
            --target variable-dependencies \
            --cache-from $REGISTRY/$PROJECT_NAME:cache-core \
            --cache-from $REGISTRY/$PROJECT_NAME:cache-extras \
            --tag $REGISTRY/$PROJECT_NAME:cache-extras-$VERSION \
            --tag $REGISTRY/$PROJECT_NAME:cache-extras \
            backend/
    fi
}

# Kommentar
build_final_image() {
    echo_info "Hinweis"
    
    docker build \
        --cache-from $REGISTRY/$PROJECT_NAME:cache-core \
        --cache-from $REGISTRY/$PROJECT_NAME:cache-extras \
        --cache-from $REGISTRY/$PROJECT_NAME:latest \
        --tag $REGISTRY/$PROJECT_NAME:$VERSION \
        --tag $REGISTRY/$PROJECT_NAME:latest \
        backend/
}

# Kommentar
push_images() {
    if [ "$PUSH_TO_REGISTRY" = "true" ]; then
        echo_info "Hinweis"
        
        if [ "$REBUILD_CORE" = true ]; then
            docker push $REGISTRY/$PROJECT_NAME:cache-core
            docker push $REGISTRY/$PROJECT_NAME:cache-core-$VERSION
        fi
        
        if [ "$REBUILD_EXTRAS" = true ]; then
            docker push $REGISTRY/$PROJECT_NAME:cache-extras
            docker push $REGISTRY/$PROJECT_NAME:cache-extras-$VERSION
        fi
        
        docker push $REGISTRY/$PROJECT_NAME:$VERSION
        docker push $REGISTRY/$PROJECT_NAME:latest
    fi
}

# Kommentar
update_checksums() {
    echo_info "Hinweis"
    md5sum backend/requirements.core.txt > backend/requirements.core.txt.md5
    md5sum backend/requirements.extras.txt > backend/requirements.extras.txt.md5
}

# Kommentar
cleanup() {
    echo_info "Hinweis"
    # LöschenKommentar
    docker image prune -f
}

# Kommentarührt ausKommentar
main() {
    echo_info "=== Docker-Build==="
    echo_info "Hinweis$VERSION"
    
    # AktivDocker BuildKit
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
    
    # Kommentar
    check_dependency_changes
    
    # Kommentar
    echo_info "Hinweis"
    docker pull $REGISTRY/$PROJECT_NAME:cache-core || echo_warn "Warnhinweis"
    docker pull $REGISTRY/$PROJECT_NAME:cache-extras || echo_warn "Warnhinweis"
    docker pull $REGISTRY/$PROJECT_NAME:latest || echo_warn "Warnhinweis"
    
    # Kommentar
    build_cache_layers
    build_final_image
    push_images
    update_checksums
    cleanup
    
    echo_info "=== Docker-Build==="
    echo_info "Hinweis$REGISTRY/$PROJECT_NAME:$VERSION"
}

# Kommentar
while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH_TO_REGISTRY="true"
            shift
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        *)
            echo_error "Fehler bei der Verarbeitung$1"
            exit 1
            ;;
    esac
done

# führt ausKommentar
main