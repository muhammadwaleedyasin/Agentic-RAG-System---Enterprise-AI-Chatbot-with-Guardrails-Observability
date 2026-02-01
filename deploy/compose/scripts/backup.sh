#!/bin/bash

# Enterprise RAG Chatbot - Backup Script
# This script creates backups of all persistent data

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKUP_DIR="./backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="rag_backup_${TIMESTAMP}"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Create backup directory
create_backup_dir() {
    log "Creating backup directory..."
    mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"
}

# Backup databases
backup_databases() {
    log "Backing up databases..."
    
    # Check if services are running
    if ! docker-compose ps | grep -q "rag_zep_db.*Up"; then
        error "Zep database is not running"
        return 1
    fi
    
    if ! docker-compose ps | grep -q "rag_langfuse_db.*Up"; then
        error "Langfuse database is not running"
        return 1
    fi
    
    # Backup Zep database
    log "Backing up Zep database..."
    docker-compose exec -T zep_db pg_dump -U zep_user -d zep_db --clean --if-exists > "${BACKUP_DIR}/${BACKUP_NAME}/zep_db.sql"
    
    # Backup Langfuse database
    log "Backing up Langfuse database..."
    docker-compose exec -T langfuse_db pg_dump -U langfuse_user -d langfuse_db --clean --if-exists > "${BACKUP_DIR}/${BACKUP_NAME}/langfuse_db.sql"
    
    log "Database backups completed"
}

# Backup Weaviate data
backup_weaviate() {
    log "Backing up Weaviate data..."
    
    # Create backup using Docker volume
    docker run --rm \
        -v "$(docker-compose ps -q weaviate | head -1 | xargs docker inspect --format '{{ range .Mounts }}{{ if eq .Destination "/var/lib/weaviate" }}{{ .Name }}{{ end }}{{ end }}')":/data \
        -v "$(pwd)/${BACKUP_DIR}/${BACKUP_NAME}":/backup \
        alpine tar czf /backup/weaviate_data.tar.gz -C /data .
    
    log "Weaviate backup completed"
}

# Backup vLLM models (if exists)
backup_vllm_models() {
    log "Backing up vLLM models..."
    
    if docker-compose ps | grep -q "rag_vllm.*Up"; then
        # Create backup of model cache
        docker run --rm \
            -v "$(docker-compose ps -q vllm | head -1 | xargs docker inspect --format '{{ range .Mounts }}{{ if eq .Destination "/root/.cache/huggingface" }}{{ .Name }}{{ end }}{{ end }}')":/data \
            -v "$(pwd)/${BACKUP_DIR}/${BACKUP_NAME}":/backup \
            alpine tar czf /backup/vllm_models.tar.gz -C /data .
        
        log "vLLM models backup completed"
    else
        info "vLLM service not running, skipping model backup"
    fi
}

# Backup application data
backup_app_data() {
    log "Backing up application data..."
    
    # Backup uploads
    if [ -d "./data/uploads" ]; then
        tar czf "${BACKUP_DIR}/${BACKUP_NAME}/uploads.tar.gz" -C ./data uploads
        log "Uploads backup completed"
    fi
    
    # Backup cache
    if [ -d "./data/cache" ]; then
        tar czf "${BACKUP_DIR}/${BACKUP_NAME}/cache.tar.gz" -C ./data cache
        log "Cache backup completed"
    fi
    
    # Backup logs
    if [ -d "./logs" ]; then
        tar czf "${BACKUP_DIR}/${BACKUP_NAME}/logs.tar.gz" logs
        log "Logs backup completed"
    fi
}

# Backup configuration
backup_config() {
    log "Backing up configuration..."
    
    # Copy .env file (excluding sensitive data)
    if [ -f ".env" ]; then
        # Remove passwords and keys for security
        sed 's/=.*_PASSWORD=.*/=***REDACTED***/g; s/=.*_KEY=.*/=***REDACTED***/g; s/=.*_SECRET=.*/=***REDACTED***/g' .env > "${BACKUP_DIR}/${BACKUP_NAME}/env_template"
    fi
    
    # Copy docker-compose files
    cp docker-compose.yml "${BACKUP_DIR}/${BACKUP_NAME}/"
    [ -f docker-compose.override.yml ] && cp docker-compose.override.yml "${BACKUP_DIR}/${BACKUP_NAME}/"
    [ -f docker-compose.prod.yml ] && cp docker-compose.prod.yml "${BACKUP_DIR}/${BACKUP_NAME}/"
    
    # Copy configuration files
    if [ -d "./deploy/configs" ]; then
        cp -r ./deploy/configs "${BACKUP_DIR}/${BACKUP_NAME}/"
    fi
    
    log "Configuration backup completed"
}

# Create backup manifest
create_manifest() {
    log "Creating backup manifest..."
    
    cat > "${BACKUP_DIR}/${BACKUP_NAME}/manifest.txt" << EOF
Enterprise RAG Chatbot Backup
=============================

Backup Date: $(date)
Backup Name: ${BACKUP_NAME}
Docker Compose Project: $(docker-compose config --services | head -1 | cut -d'_' -f1)

Contents:
---------
$(ls -la "${BACKUP_DIR}/${BACKUP_NAME}" | tail -n +2)

Services Status at Backup:
--------------------------
$(docker-compose ps)

Volume Information:
------------------
$(docker volume ls | grep "$(basename $(pwd))")

Network Information:
-------------------
$(docker network ls | grep "$(basename $(pwd))")
EOF
    
    log "Backup manifest created"
}

# Compress backup
compress_backup() {
    log "Compressing backup..."
    
    cd "${BACKUP_DIR}"
    tar czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}"
    rm -rf "${BACKUP_NAME}"
    
    log "Backup compressed: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
}

# Clean old backups
clean_old_backups() {
    local retention_days=${1:-7}
    
    log "Cleaning backups older than ${retention_days} days..."
    
    find "${BACKUP_DIR}" -name "rag_backup_*.tar.gz" -type f -mtime +${retention_days} -delete
    
    log "Old backups cleaned"
}

# Verify backup
verify_backup() {
    log "Verifying backup..."
    
    if [ -f "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" ]; then
        # Test if backup can be extracted
        tar tzf "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" > /dev/null
        
        local backup_size=$(du -h "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" | cut -f1)
        log "Backup verification successful - Size: ${backup_size}"
        
        return 0
    else
        error "Backup verification failed - file not found"
        return 1
    fi
}

# Full backup
full_backup() {
    log "Starting full backup..."
    
    create_backup_dir
    backup_databases
    backup_weaviate
    backup_vllm_models
    backup_app_data
    backup_config
    create_manifest
    compress_backup
    
    if verify_backup; then
        log "Full backup completed successfully: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
    else
        error "Backup verification failed"
        exit 1
    fi
}

# Quick backup (databases only)
quick_backup() {
    log "Starting quick backup (databases only)..."
    
    BACKUP_NAME="rag_quick_backup_${TIMESTAMP}"
    create_backup_dir
    backup_databases
    create_manifest
    compress_backup
    
    if verify_backup; then
        log "Quick backup completed successfully: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
    else
        error "Quick backup verification failed"
        exit 1
    fi
}

# List backups
list_backups() {
    log "Available backups:"
    
    if [ -d "${BACKUP_DIR}" ]; then
        ls -lh "${BACKUP_DIR}"/rag_backup_*.tar.gz 2>/dev/null | while read -r line; do
            echo "  ${line}"
        done
    else
        info "No backups found"
    fi
}

# Show backup info
show_backup_info() {
    local backup_file=$1
    
    if [ -z "$backup_file" ]; then
        error "Please specify backup file"
        exit 1
    fi
    
    if [ ! -f "$backup_file" ]; then
        error "Backup file not found: $backup_file"
        exit 1
    fi
    
    log "Backup information for: $backup_file"
    
    # Extract and show manifest
    tar xzf "$backup_file" -O "$(basename $backup_file .tar.gz)/manifest.txt"
}

# Main execution
case "${1:-full}" in
    "full")
        full_backup
        clean_old_backups 7
        ;;
    "quick")
        quick_backup
        ;;
    "databases")
        BACKUP_NAME="rag_db_backup_${TIMESTAMP}"
        create_backup_dir
        backup_databases
        compress_backup
        verify_backup
        ;;
    "list")
        list_backups
        ;;
    "clean")
        clean_old_backups "${2:-7}"
        ;;
    "info")
        show_backup_info "$2"
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [command] [options]"
        echo ""
        echo "Commands:"
        echo "  full        Create full backup (default)"
        echo "  quick       Create quick backup (databases only)"
        echo "  databases   Backup databases only"
        echo "  list        List available backups"
        echo "  clean [days] Clean backups older than X days (default: 7)"
        echo "  info <file> Show backup information"
        echo "  help        Show this help message"
        ;;
    *)
        error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac