# DocuLLaMA Azure Container Apps Deployment Guide

This guide provides step-by-step instructions for deploying DocuLLaMA to Azure Container Apps.

## 📋 Prerequisites

### Required Services
- **Azure Subscription** with Container Apps enabled
- **Qdrant Cloud Account** - Vector database service
- **Azure OpenAI Service** - Language model and embeddings

### Local Tools
- Azure CLI (`az`) version 2.50+
- Docker Desktop
- Git
- Python 3.11+

## 🚀 Quick Deployment

### 1. Clone Repository
```bash
git clone https://github.com/snedea/docullama.git
cd docullama
```

### 2. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit with your values
nano .env
```

### 3. Deploy to Azure
```bash
# Login to Azure
az login

# Set subscription (optional)
export AZURE_SUBSCRIPTION_ID="your-subscription-id"

# Deploy to staging
./deploy.sh deploy

# Or deploy to production
ENVIRONMENT=production ./deploy.sh deploy
```

## 🔧 Detailed Setup

### Azure OpenAI Configuration

1. **Create Azure OpenAI Service**
   ```bash
   az cognitiveservices account create \
     --name "docullama-openai" \
     --resource-group "docullama-rg" \
     --kind "OpenAI" \
     --sku "S0" \
     --location "eastus"
   ```

2. **Deploy Required Models**
   ```bash
   # GPT-4o-mini for chat
   az cognitiveservices account deployment create \
     --name "docullama-openai" \
     --resource-group "docullama-rg" \
     --deployment-name "gpt-4o-mini" \
     --model-name "gpt-4o-mini" \
     --model-version "2024-07-18" \
     --sku-capacity 120 \
     --sku-name "Standard"

   # Text embedding model
   az cognitiveservices account deployment create \
     --name "docullama-openai" \
     --resource-group "docullama-rg" \
     --deployment-name "text-embedding-3-small" \
     --model-name "text-embedding-3-small" \
     --model-version "1" \
     --sku-capacity 120 \
     --sku-name "Standard"
   ```

### Qdrant Cloud Setup

1. **Create Qdrant Cloud Cluster**
   - Visit [cloud.qdrant.io](https://cloud.qdrant.io)
   - Create new cluster in Azure East US region
   - Note the cluster URL and API key

2. **Configure Connection**
   ```bash
   export QDRANT_HOST="your-cluster.qdrant.io"
   export QDRANT_API_KEY="your-api-key"
   ```

### Environment Variables

Required variables for `.env`:

```env
# Azure Container Apps
PORT=8000
WORKERS=4
ENVIRONMENT=production

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_CHAT_MODEL=gpt-4o-mini
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-3-small
AZURE_OPENAI_EMBEDDING_DIMENSIONS=768

# Qdrant Cloud
QDRANT_HOST=your-cluster.qdrant.io
QDRANT_PORT=6333
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_USE_HTTPS=true
QDRANT_COLLECTION_NAME=docullama_documents

# Security
JWT_SECRET_KEY=your-jwt-secret
ENABLE_API_KEY_AUTH=true

# Features
ENABLE_RESEARCH_INTELLIGENCE=true
ENABLE_KNOWLEDGE_GRAPH=true
ENABLE_MULTI_SOURCE_RAG=true
ENABLE_CITATION_FORMATTING=true
```

## 🏗️ Manual Azure Resources

If you prefer manual resource creation:

### 1. Resource Group
```bash
az group create \
  --name "docullama-rg" \
  --location "eastus"
```

### 2. Container Registry
```bash
az acr create \
  --resource-group "docullama-rg" \
  --name "docullama" \
  --sku Standard \
  --admin-enabled true
```

### 3. Key Vault
```bash
az keyvault create \
  --resource-group "docullama-rg" \
  --name "docullama-kv" \
  --location "eastus"
```

### 4. Log Analytics
```bash
az monitor log-analytics workspace create \
  --resource-group "docullama-rg" \
  --workspace-name "docullama-logs" \
  --location "eastus"
```

### 5. Container Apps Environment
```bash
az containerapp env create \
  --name "docullama-env" \
  --resource-group "docullama-rg" \
  --location "eastus"
```

## 🔐 Security Configuration

### Secrets Management
```bash
# Store secrets in Key Vault
az keyvault secret set \
  --vault-name "docullama-kv" \
  --name "azure-openai-api-key" \
  --value "your-api-key"

az keyvault secret set \
  --vault-name "docullama-kv" \
  --name "qdrant-api-key" \
  --value "your-qdrant-key"
```

### Managed Identity
```bash
# Enable system-assigned managed identity
az containerapp identity assign \
  --name "docullama" \
  --resource-group "docullama-rg" \
  --system-assigned

# Grant Key Vault access
az keyvault set-policy \
  --name "docullama-kv" \
  --object-id "<managed-identity-id>" \
  --secret-permissions get list
```

## 📊 Monitoring Setup

### Application Insights
```bash
az monitor app-insights component create \
  --app "docullama-insights" \
  --location "eastus" \
  --resource-group "docullama-rg" \
  --workspace "/subscriptions/<sub-id>/resourceGroups/docullama-rg/providers/Microsoft.OperationalInsights/workspaces/docullama-logs"
```

### Alerts
```bash
# High error rate alert
az monitor metrics alert create \
  --name "docullama-high-error-rate" \
  --resource-group "docullama-rg" \
  --scopes "/subscriptions/<sub-id>/resourceGroups/docullama-rg/providers/Microsoft.App/containerApps/docullama" \
  --condition "avg exceptions/server > 10" \
  --description "Alert when error rate is high"
```

## 🔄 CI/CD Pipeline

### GitHub Actions Setup

1. **Create Service Principal**
   ```bash
   az ad sp create-for-rbac \
     --name "docullama-deploy" \
     --role contributor \
     --scopes "/subscriptions/<subscription-id>/resourceGroups/docullama-rg" \
     --sdk-auth
   ```

2. **Add GitHub Secrets**
   - `AZURE_CREDENTIALS` - Service principal JSON
   - `AZURE_ACR_USERNAME` - Container registry username
   - `AZURE_ACR_PASSWORD` - Container registry password
   - `AZURE_OPENAI_ENDPOINT` - OpenAI endpoint
   - `QDRANT_HOST` - Qdrant cluster host

3. **Push to trigger deployment**
   ```bash
   git push origin main
   ```

## 🎯 Testing Deployment

### Health Checks
```bash
# Get application URL
APP_URL=$(az containerapp show \
  --name "docullama" \
  --resource-group "docullama-rg" \
  --query properties.configuration.ingress.fqdn -o tsv)

# Test health endpoint
curl https://$APP_URL/health

# Test API documentation
open https://$APP_URL/docs
```

### Upload Test Document
```bash
curl -X POST "https://$APP_URL/v1/documents/upload" \
  -H "X-API-Key: your-api-key" \
  -F "file=@test_document.pdf"
```

### Test Chat Completion
```bash
curl -X POST "https://$APP_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "What information do you have?"}
    ]
  }'
```

## 🔧 Troubleshooting

### Common Issues

**Container not starting**
```bash
# Check logs
az containerapp logs show \
  --name "docullama" \
  --resource-group "docullama-rg"

# Check revision status
az containerapp revision list \
  --name "docullama" \
  --resource-group "docullama-rg"
```

**Qdrant connection errors**
```bash
# Test connectivity
curl -H "api-key: $QDRANT_API_KEY" \
  "https://$QDRANT_HOST:6333/collections"
```

**OpenAI API errors**
```bash
# Verify deployment
az cognitiveservices account deployment list \
  --name "docullama-openai" \
  --resource-group "docullama-rg"
```

### Performance Optimization

**Scale replicas**
```bash
az containerapp update \
  --name "docullama" \
  --resource-group "docullama-rg" \
  --min-replicas 2 \
  --max-replicas 10
```

**Increase resources**
```bash
az containerapp update \
  --name "docullama" \
  --resource-group "docullama-rg" \
  --cpu 4.0 \
  --memory 8Gi
```

## 💰 Cost Optimization

### Right-sizing
- Start with 2 CPU, 4Gi memory
- Monitor usage and adjust
- Use staging environment for testing

### OpenAI Cost Control
- Enable response caching
- Use 768-dimension embeddings
- Monitor token usage in Azure portal

### Qdrant Cost Control
- Use appropriate cluster size
- Monitor vector storage usage
- Clean up old collections

## 🚀 Production Checklist

- [ ] Azure OpenAI models deployed
- [ ] Qdrant Cloud cluster configured
- [ ] Key Vault secrets configured
- [ ] Managed identity setup
- [ ] Application Insights enabled
- [ ] Health checks passing
- [ ] API authentication working
- [ ] Document upload/processing tested
- [ ] Chat completions working
- [ ] Monitoring and alerts configured
- [ ] Backup strategy defined
- [ ] CI/CD pipeline working

## 📞 Support

- **GitHub Issues** - Bug reports and feature requests
- **Azure Support** - Azure-specific issues
- **Qdrant Support** - Vector database issues

---

**Deploy DocuLLaMA to Azure Container Apps and start building intelligent document experiences!** 🚀