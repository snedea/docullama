# DocuLLaMA - Enterprise RAG System for Azure Container Apps

A production-ready Document + LLaMA RAG system optimized for Azure Container Apps with advanced AutoLLaMA features, Qdrant Cloud integration, and Azure OpenAI.

## 🚀 Features

### Core RAG Capabilities
- **Universal Document Processing** - Apache Tika support for PDF, Word, Excel, PowerPoint, code files, and more
- **Hybrid Search** - Vector similarity + keyword search with re-ranking
- **Azure OpenAI Integration** - GPT-4o-mini and text-embedding-3-small (768-dim for cost optimization)
- **Qdrant Cloud** - Production vector database with JWT authentication
- **OpenAI-Compatible API** - Drop-in replacement for OpenAI chat completions

### AutoLLaMA Advanced Features
- **Knowledge Graph** - Automatic concept extraction and relationship mapping
- **Research Intelligence** - AI-powered content analysis and research suggestions
- **Citation Formatting** - Academic citation support (MLA, APA, Chicago)
- **Content Gap Analysis** - Identify missing topics and research opportunities
- **Pattern Detection** - Discover trends and insights across content

### Azure Container Apps Optimized
- **Health Checks** - `/health`, `/ready`, `/live` endpoints for Container Apps probes
- **Managed Identity** - Azure Key Vault integration for secrets management
- **Auto-scaling** - KEDA-based scaling on CPU, memory, and HTTP requests
- **Monitoring** - Application Insights, structured logging, OpenTelemetry
- **Cost Tracking** - Real-time Azure OpenAI usage and cost monitoring

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Azure         │    │ Document         │    │   Qdrant        │
│   OpenAI        │ -> │ Processing       │ -> │   Cloud         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ↓                       ↑                       ↓
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Chat            │    │   Apache Tika    │    │ Knowledge       │
│ Completions     │    │   Server         │    │ Graph           │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ↓                       ↑                       ↓
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Research        │    │ Container Apps   │    │Research         │
│ Intelligence    │ -> │ Environment      │ -> │Intelligence     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                               ↑
                    ┌──────────────────┐
                    │ Azure Monitor    │
                    │ & Key Vault      │
                    └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Azure CLI (`az`)
- Docker
- Azure subscription with Container Apps enabled
- Qdrant Cloud account
- Azure OpenAI service

### 1. Clone and Configure

```bash
git clone https://github.com/snedea/docullama.git
cd docullama

# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

### 2. Deploy to Azure

```bash
# Set environment variables
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_OPENAI_ENDPOINT="https://your-openai.openai.azure.com/"
export QDRANT_HOST="your-cluster.qdrant.io"

# Deploy to staging
./deploy.sh deploy

# Deploy to production
ENVIRONMENT=production ./deploy.sh deploy
```

### 3. Verify Deployment

```bash
# Health check
./deploy.sh health

# View logs
az containerapp logs show --name docullama --resource-group docullama-rg
```

## 📊 API Documentation

### OpenAI-Compatible Endpoints

**Chat Completion with RAG**
```bash
curl -X POST "https://your-app.azurecontainerapps.io/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "What does the documentation say about deployment?"}
    ],
    "collection_name": "docullama_documents"
  }'
```

**Document Upload**
```bash
curl -X POST "https://your-app.azurecontainerapps.io/v1/documents/upload" \
  -H "X-API-Key: your-api-key" \
  -F "file=@document.pdf" \
  -F "collection_name=my_documents"
```

**Search Documents**
```bash
curl -X POST "https://your-app.azurecontainerapps.io/v1/search" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "machine learning best practices",
    "max_results": 10,
    "enable_reranking": true
  }'
```

### Health and Monitoring

```bash
# Application health
curl https://your-app.azurecontainerapps.io/health

# System metrics
curl https://your-app.azurecontainerapps.io/metrics

# API documentation
open https://your-app.azurecontainerapps.io/docs
```

### Knowledge Graph

```bash
# Get knowledge graph data
curl "https://your-app.azurecontainerapps.io/v1/knowledge-graph" \
  -H "X-API-Key: your-api-key"
```

### Research Intelligence

```bash
# Analyze content for research insights
curl -X POST "https://your-app.azurecontainerapps.io/v1/research/analyze" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "content": "Your research content here",
    "analysis_type": "comprehensive"
  }'
```

## ⚙️ Configuration

### Environment Variables

#### Required
- `AZURE_OPENAI_ENDPOINT` - Azure OpenAI service endpoint
- `AZURE_OPENAI_API_KEY` - API key (stored in Key Vault)
- `QDRANT_HOST` - Qdrant Cloud host
- `QDRANT_API_KEY` - Qdrant API key (stored in Key Vault)
- `JWT_SECRET_KEY` - JWT signing key (stored in Key Vault)

#### Optional
- `CHUNK_SIZE=512` - Document chunk size
- `SIMILARITY_THRESHOLD=0.7` - Vector search threshold
- `MAX_SEARCH_RESULTS=10` - Maximum search results
- `ENABLE_RESEARCH_INTELLIGENCE=true` - Enable research features
- `ENABLE_KNOWLEDGE_GRAPH=true` - Enable knowledge graph

### Qdrant Cloud Configuration

```env
QDRANT_HOST=c50e289b-40c6-46bc-b9ae-065b9a62502d.eastus-0.azure.cloud.qdrant.io
QDRANT_PORT=6333
QDRANT_USE_HTTPS=true
QDRANT_COLLECTION_NAME=docullama_documents
```

### Cost Optimization

- **768-dimension embeddings** - Reduced from 1536 for 50% cost savings
- **Response caching** - Redis-based caching for repeated queries
- **Batch processing** - Efficient bulk document processing
- **Cost tracking** - Real-time usage monitoring

## 🔒 Security

### Authentication
- JWT-based API authentication
- Azure Managed Identity for resource access
- API key header: `X-API-Key`

### Secrets Management
- Azure Key Vault for all secrets
- System-assigned managed identity
- No secrets in environment variables

### Network Security
- HTTPS-only ingress
- Private Key Vault access
- Secure container registry

## 📈 Monitoring

### Application Insights
- Request/response tracking
- Error monitoring
- Performance metrics
- Custom telemetry

### Health Checks
- Liveness: `/live`
- Readiness: `/ready`
- Health: `/health`

### Logs
```bash
# View application logs
az containerapp logs show --name docullama --resource-group docullama-rg

# Stream logs
az containerapp logs tail --name docullama --resource-group docullama-rg
```

### Metrics
- Request count and latency
- Error rates
- Cost tracking
- Resource utilization

## 🔧 Development

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start local development
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest

# Format code
black .
flake8 .
```

### Docker Development

```bash
# Build image
docker build -t docullama .

# Run locally
docker run -p 8000:8000 --env-file .env docullama
```

## 📋 Testing

### Unit Tests
```bash
pytest tests/unit/
```

### Integration Tests
```bash
pytest tests/integration/
```

### Performance Tests
```bash
k6 run tests/performance/load-test.js
```

## 🚀 Deployment

### CI/CD Pipeline
- GitHub Actions workflow
- Automated testing
- Security scanning
- Multi-environment deployment

### Manual Deployment
```bash
# Deploy to staging
ENVIRONMENT=staging ./deploy.sh deploy

# Deploy to production
ENVIRONMENT=production ./deploy.sh deploy
```

### Scaling
```bash
# Scale replicas
az containerapp update \
  --name docullama \
  --resource-group docullama-rg \
  --min-replicas 2 \
  --max-replicas 20
```

## 🔍 Troubleshooting

### Common Issues

**Container App not starting**
```bash
# Check logs
az containerapp logs show --name docullama --resource-group docullama-rg

# Check revision status
az containerapp revision list --name docullama --resource-group docullama-rg
```

**Qdrant connection issues**
```bash
# Test Qdrant connectivity
curl -H "api-key: your-key" https://your-cluster.qdrant.io:6333/collections
```

**OpenAI API errors**
```bash
# Check API key and endpoint
curl -H "api-key: your-key" "https://your-openai.openai.azure.com/openai/deployments?api-version=2024-02-01"
```

### Performance Optimization

**Slow responses**
- Check embedding cache hit rate
- Monitor Qdrant query performance
- Review chunk size settings

**High costs**
- Enable response caching
- Use 768-dimension embeddings
- Implement query optimization

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Submit a pull request

## 📞 Support

- **Issues**: GitHub Issues
- **Documentation**: `/docs` endpoint
- **Health Status**: `/health` endpoint

---

**Transform your documents into intelligent, searchable knowledge with DocuLLaMA on Azure Container Apps** 🚀
