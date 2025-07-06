# DocuLLaMA Setup Instructions for Claude Web

This document provides step-by-step instructions for Claude Web to deploy DocuLLaMA to Azure Container Apps.

## 🎯 Repository Overview

**DocuLLaMA** is a production-ready RAG (Retrieval-Augmented Generation) system optimized for Azure Container Apps with advanced AutoLLaMA features.

### Key Features
- **Universal Document Processing** - Apache Tika support for multiple formats
- **Azure OpenAI Integration** - GPT-4o-mini and text-embedding-3-small
- **Qdrant Cloud Vector Database** - Production vector storage
- **Knowledge Graph** - Automatic concept extraction and relationships
- **Research Intelligence** - AI-powered content analysis
- **Azure Container Apps Optimized** - Health checks, auto-scaling, monitoring

## 🗂️ Repository Structure

```
docullama/
├── app.py                          # Main FastAPI application
├── config.py                       # Configuration management
├── monitoring.py                   # Health checks and monitoring
├── document_processor.py           # Universal document processing
├── rag_engine.py                   # RAG engine with hybrid search
├── qdrant_client.py                # Qdrant Cloud integration
├── knowledge_graph.py              # Knowledge graph management
├── research_intelligence.py        # Research intelligence engine
├── requirements.txt                # Python dependencies
├── requirements-dev.txt            # Development dependencies
├── Dockerfile                      # Container build configuration
├── .env.example                    # Environment variable template
├── containerapp.yaml              # Azure Container Apps config
├── deploy.sh                       # Deployment script
├── .github/workflows/              # CI/CD pipeline
│   └── deploy-container-app.yml
├── tests/                          # Test suite
│   ├── conftest.py
│   └── test_health.py
├── README.md                       # Main documentation
├── DEPLOYMENT_GUIDE.md             # Detailed deployment guide
├── CONTRIBUTING.md                 # Contribution guidelines
├── LICENSE                         # MIT License
├── .gitignore                      # Git ignore rules
└── .dockerignore                   # Docker ignore rules
```

## 🚀 Quick Deploy to Azure Container Apps

### Prerequisites
1. **Azure Subscription** with Container Apps enabled
2. **Qdrant Cloud Account** - Get from [cloud.qdrant.io](https://cloud.qdrant.io)
3. **Azure OpenAI Service** - Deploy GPT-4o-mini and text-embedding-3-small models

### Step 1: Configure Qdrant Cloud
```bash
# Qdrant Cloud connection details
QDRANT_HOST="c50e289b-40c6-46bc-b9ae-065b9a62502d.eastus-0.azure.cloud.qdrant.io"
QDRANT_PORT="6333"
QDRANT_USE_HTTPS="true"
```

### Step 2: Configure Azure OpenAI
```bash
# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT="https://your-openai.openai.azure.com/"
AZURE_OPENAI_CHAT_MODEL="gpt-4o-mini"
AZURE_OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
AZURE_OPENAI_EMBEDDING_DIMENSIONS="768"  # Cost-optimized
```

### Step 3: Deploy with One Command
```bash
# Clone repository
git clone https://github.com/snedea/docullama.git
cd docullama

# Configure environment
cp .env.example .env
# Edit .env with your Azure and Qdrant credentials

# Deploy to Azure
./deploy.sh deploy
```

## 🔧 Environment Configuration

Create `.env` file with these required variables:

```env
# Azure Container Apps
PORT=8000
WORKERS=4
ENVIRONMENT=production

# Azure OpenAI (REQUIRED)
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_CHAT_MODEL=gpt-4o-mini
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-3-small
AZURE_OPENAI_EMBEDDING_DIMENSIONS=768

# Qdrant Cloud (REQUIRED)
QDRANT_HOST=your-cluster.qdrant.io
QDRANT_PORT=6333
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION_NAME=docullama_documents
QDRANT_USE_HTTPS=true

# Security (REQUIRED)
JWT_SECRET_KEY=your-jwt-secret-key
ENABLE_API_KEY_AUTH=true

# Advanced Features (OPTIONAL)
ENABLE_RESEARCH_INTELLIGENCE=true
ENABLE_KNOWLEDGE_GRAPH=true
ENABLE_MULTI_SOURCE_RAG=true
ENABLE_CITATION_FORMATTING=true

# Document Processing
MAX_FILE_SIZE_MB=100
CHUNK_SIZE=512
CHUNK_OVERLAP=50
QUALITY_THRESHOLD=0.6

# Monitoring
LOG_LEVEL=INFO
ENABLE_STRUCTURED_LOGGING=true
ENABLE_OPENTELEMETRY=true
ENABLE_COST_TRACKING=true
```

## 📋 Deployment Checklist

### Azure Resources Needed
- [ ] **Resource Group** - `docullama-rg`
- [ ] **Container Registry** - `docullama`
- [ ] **Key Vault** - `docullama-kv`
- [ ] **Log Analytics** - `docullama-logs`
- [ ] **Application Insights** - `docullama-insights`
- [ ] **Container Apps Environment** - `docullama-env`
- [ ] **Container App** - `docullama`

### External Services
- [ ] **Qdrant Cloud Cluster** - Vector database
- [ ] **Azure OpenAI Service** - Language models

### Configuration
- [ ] **Environment Variables** - All required variables set
- [ ] **Secrets in Key Vault** - API keys stored securely
- [ ] **Managed Identity** - System-assigned identity configured
- [ ] **Health Checks** - All endpoints responding

## 🧪 Testing the Deployment

### Health Check
```bash
curl https://your-app.azurecontainerapps.io/health
```

### Upload Document
```bash
curl -X POST "https://your-app.azurecontainerapps.io/v1/documents/upload" \
  -H "X-API-Key: your-api-key" \
  -F "file=@test_document.pdf"
```

### Chat with RAG
```bash
curl -X POST "https://your-app.azurecontainerapps.io/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "What information do you have about the uploaded documents?"}
    ]
  }'
```

### API Documentation
```bash
# Visit interactive API docs
open https://your-app.azurecontainerapps.io/docs
```

## 📊 Key Endpoints

- **Health**: `/health`, `/ready`, `/live`
- **Metrics**: `/metrics`
- **API Docs**: `/docs`
- **Chat**: `/v1/chat/completions`
- **Upload**: `/v1/documents/upload`
- **Search**: `/v1/search`
- **Knowledge Graph**: `/v1/knowledge-graph`
- **Research**: `/v1/research/analyze`

## 🔒 Security Features

- **JWT Authentication** with API key support
- **Azure Key Vault** integration for secrets
- **Managed Identity** for secure resource access
- **HTTPS-only** ingress
- **Input validation** and sanitization

## 📈 Production Features

- **Auto-scaling** based on CPU, memory, and HTTP requests
- **Health probes** for Container Apps
- **Structured logging** with Application Insights
- **Cost tracking** for Azure OpenAI usage
- **Performance monitoring** with OpenTelemetry

## 🚨 Troubleshooting

### Common Issues
1. **Container not starting** - Check logs with `az containerapp logs show`
2. **Qdrant connection failed** - Verify API key and host
3. **OpenAI errors** - Check model deployments and API key
4. **Health check failing** - Verify all dependencies are accessible

### Debug Commands
```bash
# View logs
az containerapp logs show --name docullama --resource-group docullama-rg

# Check container status
az containerapp show --name docullama --resource-group docullama-rg

# Test Qdrant connection
curl -H "api-key: $QDRANT_API_KEY" "https://$QDRANT_HOST:6333/collections"
```

## 💡 Claude Web Action Items

To deploy DocuLLaMA successfully:

1. **Review the repository structure** and understand the architecture
2. **Set up Azure OpenAI** with required models (gpt-4o-mini, text-embedding-3-small)
3. **Configure Qdrant Cloud** cluster and get connection details
4. **Run the deployment script** with proper environment variables
5. **Test all endpoints** to ensure everything is working
6. **Monitor the deployment** using Azure Application Insights

The system is designed to be production-ready with comprehensive monitoring, security, and scalability features optimized for Azure Container Apps.

---

**Ready to deploy DocuLLaMA and transform documents into intelligent, searchable knowledge!** 🚀