#!/bin/bash

# DocuLLaMA Azure Container Apps Deployment Script
# Production-ready deployment with Qdrant Cloud integration

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="docullama"
LOCATION="eastus"
SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID:-}"
ENVIRONMENT="${ENVIRONMENT:-staging}"

# Resource names
RESOURCE_GROUP="${PROJECT_NAME}-rg-${ENVIRONMENT}"
CONTAINER_APP_ENV="${PROJECT_NAME}-env-${ENVIRONMENT}"
CONTAINER_APP_NAME="${PROJECT_NAME}-${ENVIRONMENT}"
ACR_NAME="${PROJECT_NAME}acr${ENVIRONMENT}"
KEY_VAULT_NAME="${PROJECT_NAME}-kv-${ENVIRONMENT}"
LOG_ANALYTICS_NAME="${PROJECT_NAME}-logs-${ENVIRONMENT}"
APP_INSIGHTS_NAME="${PROJECT_NAME}-insights-${ENVIRONMENT}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Azure CLI
    if ! command -v az &> /dev/null; then
        log_error "Azure CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    # Check if logged in to Azure
    if ! az account show &> /dev/null; then
        log_error "Not logged in to Azure. Please run 'az login' first."
        exit 1
    fi
    
    # Set subscription if provided
    if [[ -n "${SUBSCRIPTION_ID}" ]]; then
        log_info "Setting Azure subscription to ${SUBSCRIPTION_ID}"
        az account set --subscription "${SUBSCRIPTION_ID}"
    fi
    
    log_success "Prerequisites check completed"
}

# Create resource group
create_resource_group() {
    log_info "Creating resource group ${RESOURCE_GROUP}..."
    
    if az group show --name "${RESOURCE_GROUP}" &> /dev/null; then
        log_warning "Resource group ${RESOURCE_GROUP} already exists"
    else
        az group create \
            --name "${RESOURCE_GROUP}" \
            --location "${LOCATION}" \
            --tags environment="${ENVIRONMENT}" project="${PROJECT_NAME}"
        log_success "Resource group ${RESOURCE_GROUP} created"
    fi
}

# Create Container Registry
create_container_registry() {
    log_info "Creating Azure Container Registry ${ACR_NAME}..."
    
    if az acr show --name "${ACR_NAME}" --resource-group "${RESOURCE_GROUP}" &> /dev/null; then
        log_warning "ACR ${ACR_NAME} already exists"
    else
        az acr create \
            --resource-group "${RESOURCE_GROUP}" \
            --name "${ACR_NAME}" \
            --sku Standard \
            --admin-enabled true
        log_success "ACR ${ACR_NAME} created"
    fi
    
    # Enable admin user
    az acr update --name "${ACR_NAME}" --admin-enabled true
}

# Create Key Vault
create_key_vault() {
    log_info "Creating Key Vault ${KEY_VAULT_NAME}..."
    
    if az keyvault show --name "${KEY_VAULT_NAME}" --resource-group "${RESOURCE_GROUP}" &> /dev/null; then
        log_warning "Key Vault ${KEY_VAULT_NAME} already exists"
    else
        az keyvault create \
            --resource-group "${RESOURCE_GROUP}" \
            --name "${KEY_VAULT_NAME}" \
            --location "${LOCATION}" \
            --sku standard \
            --enable-rbac-authorization false
        log_success "Key Vault ${KEY_VAULT_NAME} created"
    fi
}

# Create Log Analytics Workspace
create_log_analytics() {
    log_info "Creating Log Analytics workspace ${LOG_ANALYTICS_NAME}..."
    
    if az monitor log-analytics workspace show --resource-group "${RESOURCE_GROUP}" --workspace-name "${LOG_ANALYTICS_NAME}" &> /dev/null; then
        log_warning "Log Analytics workspace ${LOG_ANALYTICS_NAME} already exists"
    else
        az monitor log-analytics workspace create \
            --resource-group "${RESOURCE_GROUP}" \
            --workspace-name "${LOG_ANALYTICS_NAME}" \
            --location "${LOCATION}" \
            --sku PerGB2018
        log_success "Log Analytics workspace ${LOG_ANALYTICS_NAME} created"
    fi
}

# Create Application Insights
create_app_insights() {
    log_info "Creating Application Insights ${APP_INSIGHTS_NAME}..."
    
    if az monitor app-insights component show --app "${APP_INSIGHTS_NAME}" --resource-group "${RESOURCE_GROUP}" &> /dev/null; then
        log_warning "Application Insights ${APP_INSIGHTS_NAME} already exists"
    else
        az monitor app-insights component create \
            --app "${APP_INSIGHTS_NAME}" \
            --location "${LOCATION}" \
            --resource-group "${RESOURCE_GROUP}" \
            --workspace "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/${RESOURCE_GROUP}/providers/Microsoft.OperationalInsights/workspaces/${LOG_ANALYTICS_NAME}"
        log_success "Application Insights ${APP_INSIGHTS_NAME} created"
    fi
}

# Create Container Apps Environment
create_container_apps_environment() {
    log_info "Creating Container Apps Environment ${CONTAINER_APP_ENV}..."
    
    if az containerapp env show --name "${CONTAINER_APP_ENV}" --resource-group "${RESOURCE_GROUP}" &> /dev/null; then
        log_warning "Container Apps Environment ${CONTAINER_APP_ENV} already exists"
    else
        # Get Log Analytics workspace ID
        WORKSPACE_ID=$(az monitor log-analytics workspace show \
            --resource-group "${RESOURCE_GROUP}" \
            --workspace-name "${LOG_ANALYTICS_NAME}" \
            --query customerId -o tsv)
        
        WORKSPACE_KEY=$(az monitor log-analytics workspace get-shared-keys \
            --resource-group "${RESOURCE_GROUP}" \
            --workspace-name "${LOG_ANALYTICS_NAME}" \
            --query primarySharedKey -o tsv)
        
        az containerapp env create \
            --name "${CONTAINER_APP_ENV}" \
            --resource-group "${RESOURCE_GROUP}" \
            --location "${LOCATION}" \
            --logs-workspace-id "${WORKSPACE_ID}" \
            --logs-workspace-key "${WORKSPACE_KEY}"
        
        log_success "Container Apps Environment ${CONTAINER_APP_ENV} created"
    fi
}

# Build and push Docker image
build_and_push_image() {
    log_info "Building and pushing Docker image..."
    
    # Get ACR login server
    ACR_LOGIN_SERVER=$(az acr show --name "${ACR_NAME}" --resource-group "${RESOURCE_GROUP}" --query loginServer -o tsv)
    
    # Login to ACR
    az acr login --name "${ACR_NAME}"
    
    # Build image
    IMAGE_TAG="${ACR_LOGIN_SERVER}/${PROJECT_NAME}:$(date +%Y%m%d-%H%M%S)"
    
    log_info "Building image ${IMAGE_TAG}..."
    docker build -t "${IMAGE_TAG}" "${SCRIPT_DIR}"
    
    # Push image
    log_info "Pushing image to ACR..."
    docker push "${IMAGE_TAG}"
    
    log_success "Image ${IMAGE_TAG} built and pushed successfully"
    echo "${IMAGE_TAG}"
}

# Set up secrets in Key Vault
setup_secrets() {
    log_info "Setting up secrets in Key Vault..."
    
    # Prompt for secrets if not provided via environment variables
    if [[ -z "${AZURE_OPENAI_API_KEY:-}" ]]; then
        read -rsp "Enter Azure OpenAI API Key: " AZURE_OPENAI_API_KEY
        echo
    fi
    
    if [[ -z "${QDRANT_API_KEY:-}" ]]; then
        read -rsp "Enter Qdrant API Key: " QDRANT_API_KEY
        echo
    fi
    
    if [[ -z "${JWT_SECRET_KEY:-}" ]]; then
        JWT_SECRET_KEY=$(openssl rand -base64 32)
        log_info "Generated JWT secret key"
    fi
    
    # Store secrets
    az keyvault secret set --vault-name "${KEY_VAULT_NAME}" --name "azure-openai-api-key" --value "${AZURE_OPENAI_API_KEY}"
    az keyvault secret set --vault-name "${KEY_VAULT_NAME}" --name "qdrant-api-key" --value "${QDRANT_API_KEY}"
    az keyvault secret set --vault-name "${KEY_VAULT_NAME}" --name "jwt-secret-key" --value "${JWT_SECRET_KEY}"
    
    log_success "Secrets configured in Key Vault"
}

# Deploy Container App
deploy_container_app() {
    local image_tag="$1"
    
    log_info "Deploying Container App ${CONTAINER_APP_NAME}..."
    
    # Get Application Insights connection string
    APP_INSIGHTS_CONNECTION_STRING=$(az monitor app-insights component show \
        --app "${APP_INSIGHTS_NAME}" \
        --resource-group "${RESOURCE_GROUP}" \
        --query connectionString -o tsv)
    
    # Get ACR credentials
    ACR_USERNAME=$(az acr credential show --name "${ACR_NAME}" --query username -o tsv)
    ACR_PASSWORD=$(az acr credential show --name "${ACR_NAME}" --query passwords[0].value -o tsv)
    
    # Deploy Container App
    az containerapp create \
        --name "${CONTAINER_APP_NAME}" \
        --resource-group "${RESOURCE_GROUP}" \
        --environment "${CONTAINER_APP_ENV}" \
        --image "${image_tag}" \
        --registry-server "${ACR_NAME}.azurecr.io" \
        --registry-username "${ACR_USERNAME}" \
        --registry-password "${ACR_PASSWORD}" \
        --target-port 8000 \
        --ingress external \
        --min-replicas 1 \
        --max-replicas 10 \
        --cpu 2.0 \
        --memory 4Gi \
        --secrets \
            azure-openai-api-key="keyvaultref:https://${KEY_VAULT_NAME}.vault.azure.net/secrets/azure-openai-api-key,identityref:system" \
            qdrant-api-key="keyvaultref:https://${KEY_VAULT_NAME}.vault.azure.net/secrets/qdrant-api-key,identityref:system" \
            jwt-secret-key="keyvaultref:https://${KEY_VAULT_NAME}.vault.azure.net/secrets/jwt-secret-key,identityref:system" \
        --env-vars \
            "PORT=8000" \
            "WORKERS=4" \
            "ENVIRONMENT=${ENVIRONMENT}" \
            "AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT:-https://your-openai.openai.azure.com/}" \
            "AZURE_OPENAI_API_KEY=secretref:azure-openai-api-key" \
            "AZURE_OPENAI_API_VERSION=2024-02-01" \
            "AZURE_OPENAI_CHAT_MODEL=gpt-4o-mini" \
            "AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-3-small" \
            "AZURE_OPENAI_EMBEDDING_DIMENSIONS=768" \
            "QDRANT_HOST=${QDRANT_HOST:-c50e289b-40c6-46bc-b9ae-065b9a62502d.eastus-0.azure.cloud.qdrant.io}" \
            "QDRANT_PORT=6333" \
            "QDRANT_API_KEY=secretref:qdrant-api-key" \
            "QDRANT_COLLECTION_NAME=docullama_documents" \
            "QDRANT_USE_HTTPS=true" \
            "JWT_SECRET_KEY=secretref:jwt-secret-key" \
            "APPLICATIONINSIGHTS_CONNECTION_STRING=${APP_INSIGHTS_CONNECTION_STRING}" \
            "LOG_LEVEL=INFO" \
            "ENABLE_STRUCTURED_LOGGING=true" \
            "ENABLE_OPENTELEMETRY=true" \
            "ENABLE_COST_TRACKING=true" \
            "ENABLE_RESEARCH_INTELLIGENCE=true" \
            "ENABLE_KNOWLEDGE_GRAPH=true" \
            "ENABLE_MULTI_SOURCE_RAG=true" \
            "ENABLE_CITATION_FORMATTING=true"
    
    log_success "Container App ${CONTAINER_APP_NAME} deployed successfully"
}

# Configure managed identity and Key Vault access
configure_managed_identity() {
    log_info "Configuring managed identity and Key Vault access..."
    
    # Enable system-assigned managed identity
    az containerapp identity assign \
        --name "${CONTAINER_APP_NAME}" \
        --resource-group "${RESOURCE_GROUP}" \
        --system-assigned
    
    # Get the managed identity principal ID
    PRINCIPAL_ID=$(az containerapp identity show \
        --name "${CONTAINER_APP_NAME}" \
        --resource-group "${RESOURCE_GROUP}" \
        --query principalId -o tsv)
    
    # Grant Key Vault access to managed identity
    az keyvault set-policy \
        --name "${KEY_VAULT_NAME}" \
        --object-id "${PRINCIPAL_ID}" \
        --secret-permissions get list
    
    log_success "Managed identity configured with Key Vault access"
}

# Health check
health_check() {
    log_info "Performing health check..."
    
    # Get the application URL
    APP_URL=$(az containerapp show \
        --name "${CONTAINER_APP_NAME}" \
        --resource-group "${RESOURCE_GROUP}" \
        --query properties.configuration.ingress.fqdn -o tsv)
    
    log_info "Application URL: https://${APP_URL}"
    
    # Wait for deployment to be ready
    log_info "Waiting for application to be ready..."
    sleep 30
    
    # Health check
    if curl -f "https://${APP_URL}/health" &> /dev/null; then
        log_success "Health check passed! Application is running correctly."
        log_info "API Documentation: https://${APP_URL}/docs"
        log_info "Health Status: https://${APP_URL}/health"
        log_info "Metrics: https://${APP_URL}/metrics"
    else
        log_error "Health check failed! Please check the application logs."
        log_info "Check logs with: az containerapp logs show --name ${CONTAINER_APP_NAME} --resource-group ${RESOURCE_GROUP}"
        exit 1
    fi
}

# Display deployment summary
deployment_summary() {
    log_info "Deployment Summary:"
    echo "=================================="
    echo "Environment: ${ENVIRONMENT}"
    echo "Resource Group: ${RESOURCE_GROUP}"
    echo "Container App: ${CONTAINER_APP_NAME}"
    echo "ACR: ${ACR_NAME}"
    echo "Key Vault: ${KEY_VAULT_NAME}"
    echo ""
    
    APP_URL=$(az containerapp show \
        --name "${CONTAINER_APP_NAME}" \
        --resource-group "${RESOURCE_GROUP}" \
        --query properties.configuration.ingress.fqdn -o tsv)
    
    echo "Application URLs:"
    echo "  Main App: https://${APP_URL}"
    echo "  API Docs: https://${APP_URL}/docs"
    echo "  Health: https://${APP_URL}/health"
    echo "  Metrics: https://${APP_URL}/metrics"
    echo ""
    
    echo "Useful commands:"
    echo "  View logs: az containerapp logs show --name ${CONTAINER_APP_NAME} --resource-group ${RESOURCE_GROUP}"
    echo "  Update app: az containerapp update --name ${CONTAINER_APP_NAME} --resource-group ${RESOURCE_GROUP}"
    echo "  Scale app: az containerapp update --name ${CONTAINER_APP_NAME} --resource-group ${RESOURCE_GROUP} --min-replicas 1 --max-replicas 10"
    echo "=================================="
}

# Cleanup function
cleanup() {
    if [[ "${1:-}" == "full" ]]; then
        log_warning "Performing full cleanup (deleting all resources)..."
        read -p "Are you sure you want to delete all resources? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            az group delete --name "${RESOURCE_GROUP}" --yes --no-wait
            log_success "Resource group ${RESOURCE_GROUP} deletion initiated"
        fi
    else
        log_info "No cleanup performed. Use './deploy.sh cleanup full' for full cleanup."
    fi
}

# Main deployment function
main() {
    local command="${1:-deploy}"
    
    case "${command}" in
        "deploy")
            log_info "Starting DocuLLaMA deployment to ${ENVIRONMENT}..."
            check_prerequisites
            create_resource_group
            create_log_analytics
            create_app_insights
            create_container_registry
            create_key_vault
            create_container_apps_environment
            setup_secrets
            
            IMAGE_TAG=$(build_and_push_image)
            deploy_container_app "${IMAGE_TAG}"
            configure_managed_identity
            health_check
            deployment_summary
            ;;
        "cleanup")
            cleanup "${2:-}"
            ;;
        "health")
            health_check
            ;;
        *)
            echo "Usage: $0 [deploy|cleanup|health]"
            echo "  deploy  - Deploy DocuLLaMA to Azure Container Apps"
            echo "  cleanup - Clean up resources (use 'cleanup full' for complete removal)"
            echo "  health  - Perform health check on deployed application"
            exit 1
            ;;
    esac
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi