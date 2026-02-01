# Enterprise RAG Chatbot

An enterprise-grade Retrieval-Augmented Generation (RAG) chatbot system built with FastAPI backend and Next.js frontend. This system enables intelligent document-based conversations with advanced security, compliance, and observability features.

## Features

### Core Capabilities
- **RAG Pipeline**: Intelligent document retrieval and AI-powered response generation
- **Multi-Provider Support**: Works with OpenAI, OpenRouter, vLLM, and local models
- **Vector Search**: Hybrid search with BM25 + vector similarity using Weaviate, Pinecone, Chroma, or Milvus
- **Document Processing**: Support for PDF, Word, text files with automatic chunking and embedding
- **Real-time Chat**: WebSocket-based streaming responses

### Security & Compliance
- **PII Detection**: Automatic detection and redaction of sensitive information
- **Content Filtering**: Topic moderation and safety checks
- **Citation Enforcement**: All responses include proper source attribution
- **Role-Based Access Control**: Granular user permissions
- **Audit Logging**: Complete activity tracking for compliance (GDPR, HIPAA, SOC2)

### Observability
- **Langfuse Integration**: AI model performance tracking
- **Structured Logging**: Security event correlation
- **Health Monitoring**: Real-time health checks and alerting

## Tech Stack

### Frontend
- Next.js 14 (App Router)
- React 18
- TypeScript
- Tailwind CSS
- React Query
- Socket.IO Client

### Backend
- FastAPI
- Python 3.9+
- Pydantic 2
- SQLAlchemy 2
- Uvicorn

### Infrastructure
- Weaviate (Vector Database)
- PostgreSQL (Relational Database)
- Zep (Memory Management)
- Langfuse (Observability)
- Docker & Docker Compose

## Project Structure

```
enterprise-rag-chatbot/
├── frontend/                 # Next.js Frontend Application
│   ├── src/app/             # App Router pages (login, chat, documents, admin)
│   ├── src/components/      # Reusable UI components
│   ├── src/contexts/        # React contexts (Auth, Theme)
│   ├── src/hooks/           # Custom React hooks
│   ├── src/lib/             # Utilities (API client, WebSocket)
│   └── src/types/           # TypeScript definitions
├── src/                      # Backend RAG System
│   ├── app/                 # FastAPI application
│   ├── core/                # Core RAG components
│   ├── guardrails/          # Security and compliance systems
│   ├── security/            # Authentication and authorization
│   ├── memory/              # Conversation memory management
│   ├── observability/       # Monitoring and logging
│   ├── providers/           # LLM and embedding providers
│   ├── vector_db/           # Vector database abstractions
│   ├── retrieval/           # Search and retrieval engine
│   └── ingestion/           # Document processing pipeline
├── deploy/compose/           # Docker Compose deployment files
├── config/                   # Configuration files
├── tests/                    # Test suites
├── scripts/                  # Utility scripts
└── alembic/                  # Database migrations
```

## Prerequisites

- Node.js 18.17.0+
- Python 3.9+
- Docker and Docker Compose
- Git

## Installation

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd enterprise-rag-chatbot

# Navigate to deployment folder
cd deploy/compose

# Create environment file
cp .env.example .env
# Edit .env with your API keys

# Start all services
docker-compose up -d

# Check service status
docker-compose ps
```

### Option 2: Local Development

**Backend Setup:**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
# Edit .env with your configuration

# Run database migrations
alembic upgrade head

# Start the backend server
python -m src.app.main
```

**Frontend Setup:**
```bash
cd frontend

# Install dependencies
npm install

# Create environment file
cp .env.example .env.local
# Edit .env.local with your configuration

# Start development server
npm run dev
```

## Configuration

### Essential Environment Variables

```env
# Application
ENVIRONMENT=development
LOG_LEVEL=INFO

# LLM Provider
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your-api-key

# Embedding Provider
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your-api-key

# Vector Database
VECTOR_DB_PROVIDER=weaviate
WEAVIATE_URL=http://localhost:8080

# Security
GUARDRAILS_ENABLED=true
PII_DETECTION_ENABLED=true
JWT_SECRET_KEY=your-secret-key

# Observability
OBSERVABILITY_PROVIDER=langfuse
LANGFUSE_SECRET_KEY=your-secret-key
```

## Usage

### Service Endpoints

After starting the application:

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost:3000 | Web application |
| Backend API | http://localhost:8000 | REST API |
| API Docs | http://localhost:8000/docs | Swagger documentation |
| Weaviate | http://localhost:8080 | Vector database |
| Langfuse | http://localhost:3001 | Observability dashboard |

### API Endpoints

- `POST /api/v1/chat` - Send a chat message
- `POST /api/v1/documents/upload` - Upload documents
- `GET /api/v1/documents` - List documents
- `GET /api/v1/health` - Health check
- `POST /api/v1/auth/login` - User authentication

## Development

### Running Tests

```bash
# Backend tests
pytest tests/ -v --cov=src

# Frontend tests
cd frontend && npm test

# End-to-end tests
cd frontend && npm run test:e2e
```

### Code Quality

```bash
# Backend
black src/ tests/           # Format code
isort src/ tests/           # Sort imports
ruff check src/ tests/      # Lint code
mypy src/                   # Type checking

# Frontend
cd frontend
npm run lint               # ESLint
npm run typecheck          # TypeScript check
```

### Building for Production

```bash
# Frontend build
cd frontend && npm run build

# Docker production build
cd deploy/compose
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Deployment

### Docker Compose Production

```bash
cd deploy/compose
cp .env.example .env
# Configure production environment variables

docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale services
docker-compose up -d --scale app=3 --scale frontend=2
```

### Health Checks

- Backend: `GET /api/v1/health`
- Frontend: `GET /`
- Memory Service: `GET /healthz`

## Troubleshooting

### Common Issues

**Backend won't start:**
- Check that all required environment variables are set
- Verify database connection settings
- Ensure required ports are available

**Vector database connection failed:**
- Verify Weaviate is running: `docker-compose ps`
- Check WEAVIATE_URL environment variable

**Frontend API errors:**
- Verify backend is running and accessible
- Check NEXT_PUBLIC_API_URL in frontend environment

**Document upload fails:**
- Check file size limits in configuration
- Verify supported file formats (PDF, DOCX, TXT)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make your changes with tests
4. Run the test suite
5. Submit a pull request

## License

This project is proprietary. All rights reserved.

## Support

For issues and questions, please open an issue in the repository.
