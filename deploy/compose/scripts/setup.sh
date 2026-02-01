#!/bin/bash

# Enterprise RAG Chatbot - Setup Script
# This script initializes the Docker environment and performs initial setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if Docker and Docker Compose are installed
check_dependencies() {
    log "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    log "Dependencies check passed"
}

# Check for GPU support
check_gpu_support() {
    log "Checking GPU support..."
    
    if command -v nvidia-smi &> /dev/null; then
        info "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
        
        if docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
            log "Docker GPU support confirmed"
            export GPU_AVAILABLE=true
        else
            warn "Docker GPU support not available. Install nvidia-docker2 for GPU acceleration."
            export GPU_AVAILABLE=false
        fi
    else
        warn "No NVIDIA GPU detected. Local LLM inference will use CPU (slower)."
        export GPU_AVAILABLE=false
    fi
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    mkdir -p data/uploads
    mkdir -p data/cache
    mkdir -p logs
    mkdir -p deploy/configs/nginx/ssl
    
    # Set appropriate permissions
    chmod 755 data/uploads data/cache logs
    
    log "Directories created successfully"
}

# Setup environment file
setup_environment() {
    log "Setting up environment configuration..."
    
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            cp .env.example .env
            log "Created .env from .env.example"
            warn "Please edit .env file with your API keys and configuration"
        else
            error ".env.example not found. Cannot create environment file."
            exit 1
        fi
    else
        info ".env file already exists"
    fi
}

# Generate secure passwords for databases
generate_passwords() {
    log "Generating secure database passwords..."
    
    if ! grep -q "POSTGRES_PASSWORD=" .env || grep -q "postgres_password" .env; then
        POSTGRES_PASS=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
        ZEP_DB_PASS=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
        LANGFUSE_DB_PASS=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
        
        # Update .env file
        sed -i "s/POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=${POSTGRES_PASS}/" .env
        sed -i "s/ZEP_DB_PASSWORD=.*/ZEP_DB_PASSWORD=${ZEP_DB_PASS}/" .env
        sed -i "s/LANGFUSE_DB_PASSWORD=.*/LANGFUSE_DB_PASSWORD=${LANGFUSE_DB_PASS}/" .env
        
        log "Database passwords generated and updated in .env"
    else
        info "Database passwords already configured"
    fi
}

# Pull Docker images
pull_images() {
    log "Pulling Docker images..."
    
    # Determine which profile to use
    if [ "$GPU_AVAILABLE" = true ]; then
        docker-compose --profile local-llm pull
    else
        docker-compose pull
    fi
    
    log "Docker images pulled successfully"
}

# Initialize databases
init_databases() {
    log "Initializing databases..."
    
    # Start only database services first
    docker-compose up -d zep_db langfuse_db
    
    # Wait for databases to be ready
    log "Waiting for databases to be ready..."
    sleep 30
    
    # Check database connectivity
    docker-compose exec -T zep_db pg_isready -U zep_user
    docker-compose exec -T langfuse_db pg_isready -U langfuse_user
    
    log "Databases initialized successfully"
}

# Start services
start_services() {
    log "Starting services..."
    
    if [ "$GPU_AVAILABLE" = true ]; then
        docker-compose --profile local-llm up -d
    else
        docker-compose up -d
    fi
    
    log "Services started successfully"
}

# Health check
health_check() {
    log "Performing health checks..."
    
    # Wait for services to be ready
    sleep 60
    
    # Check service health
    local services=("weaviate:8080" "zep:8000" "langfuse:3000")
    
    if [ "$GPU_AVAILABLE" = true ]; then
        services+=("vllm:8000")
    fi
    
    for service in "${services[@]}"; do
        IFS=':' read -r name port <<< "$service"
        if curl -f -s "http://localhost:${port}/health" > /dev/null 2>&1 || \
           curl -f -s "http://localhost:${port}/healthz" > /dev/null 2>&1 || \
           curl -f -s "http://localhost:${port}/v1/.well-known/ready" > /dev/null 2>&1; then
            log "${name} service is healthy"
        else
            warn "${name} service health check failed"
        fi
    done
}

# Display service URLs
display_urls() {
    log "Setup completed! Service URLs:"
    echo ""
    echo -e "${BLUE}Main Services:${NC}"
    echo "  • API Documentation: http://localhost:8000/docs"
    echo "  • Application Health: http://localhost:8000/health"
    echo ""
    echo -e "${BLUE}Management Interfaces:${NC}"
    echo "  • Weaviate Console: http://localhost:8080/v1/meta"
    echo "  • Langfuse Dashboard: http://localhost:3000"
    echo "  • Zep API: http://localhost:8002"
    
    if [ "$GPU_AVAILABLE" = true ]; then
        echo "  • vLLM API: http://localhost:8001/v1/models"
    fi
    
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "  1. Edit .env file with your API keys"
    echo "  2. Test the API: curl http://localhost:8000/health"
    echo "  3. Upload documents via the API"
    echo "  4. Start querying your documents!"
    echo ""
}

# Main execution
main() {
    log "Starting Enterprise RAG Chatbot setup..."
    
    check_dependencies
    check_gpu_support
    create_directories
    setup_environment
    generate_passwords
    pull_images
    init_databases
    start_services
    health_check
    display_urls
    
    log "Setup completed successfully!"
}

# Handle script arguments
case "${1:-}" in
    "check")
        check_dependencies
        check_gpu_support
        ;;
    "pull")
        pull_images
        ;;
    "start")
        start_services
        ;;
    "health")
        health_check
        ;;
    "stop")
        log "Stopping services..."
        docker-compose down
        ;;
    "clean")
        log "Cleaning up..."
        docker-compose down -v
        docker system prune -f
        ;;
    *)
        main
        ;;
esac